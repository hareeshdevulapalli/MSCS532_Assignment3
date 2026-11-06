"""Benchmarks for Assignment 3.

Part 1 times Randomized Quicksort against Deterministic Quicksort (first
element as pivot) on random, sorted, reverse-sorted, and repeated-element
arrays, and counts pivot comparisons. A randomized Quicksort that uses the
two-way Lomuto partition is included to show the effect of repeated keys.

Part 2 measures the hash table with chaining:
  * time per operation as the load factor alpha = n/m grows (fixed m),
  * time per operation as n grows, with and without dynamic resizing,
  * chain lengths under the universal hash versus h(k) = k mod m.

Times are medians over several runs measured with time.perf_counter.
CSV files and graphs are written to results/.

    python3 benchmark.py            # full run, a couple of minutes
    python3 benchmark.py --quick    # small sizes, a few seconds
"""

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time

from hash_table import HashTable
from quicksort import deterministic_quicksort, randomized_quicksort

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def randomized_quicksort_lomuto(arr, stats=None):
    return randomized_quicksort(arr, three_way=False, stats=stats)


SORTS = {
    "Randomized Quicksort": randomized_quicksort,
    "Deterministic Quicksort": deterministic_quicksort,
    "Randomized (Lomuto)": randomized_quicksort_lomuto,
}
DISTRIBUTIONS = ["random", "sorted", "reverse", "repeated"]
SKIP_AFTER_MS = 1000  # stop growing n for a case once one run takes longer


def make_array(kind, n, rng):
    if kind == "random":
        return [rng.randint(0, 10 * n) for _ in range(n)]
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "repeated":
        return [rng.randint(0, 9) for _ in range(n)]  # only 10 distinct keys
    raise ValueError(kind)


def median_ms(func, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------- Part 1

def bench_quicksort(sizes, repeats, rng):
    rows = []
    for dist in DISTRIBUTIONS:
        for name, sort in SORTS.items():
            for n in sizes:
                data = make_array(dist, n, rng)

                def run():
                    copy = list(data)
                    sort(copy)
                    return copy

                assert run() == sorted(data)
                ms = median_ms(run, repeats)
                stats = {"comparisons": 0}
                sort(list(data), stats=stats)
                rows.append({"distribution": dist, "algorithm": name, "n": n,
                             "time_ms": round(ms, 3),
                             "comparisons": stats["comparisons"]})
                print(f"  {dist:>8}  {name:<24} n={n:>6}  {ms:9.2f} ms"
                      f"  {stats['comparisons']:>11,} comparisons")
                if ms > SKIP_AFTER_MS:
                    break  # quadratic case: larger n would take too long
    return rows


def plot_quicksort(rows, plt):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, dist in zip(axes.flat, DISTRIBUTIONS):
        for name in SORTS:
            pts = [(r["n"], r["time_ms"]) for r in rows
                   if r["distribution"] == dist and r["algorithm"] == name]
            ax.plot(*zip(*pts), marker="o", label=name)
        ax.set_title(f"{dist.capitalize()} input")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("time (ms)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend()
    fig.suptitle("Quicksort running time (median, log-log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "quicksort_time.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for dist in DISTRIBUTIONS:
        pts = [(r["n"], r["comparisons"] / (r["n"] * math.log2(r["n"])))
               for r in rows
               if r["distribution"] == dist and r["algorithm"] == "Randomized Quicksort"]
        ax.plot(*zip(*pts), marker="o", label=f"randomized, {dist}")
    pts = [(r["n"], r["comparisons"] / (r["n"] * math.log2(r["n"])))
           for r in rows
           if r["distribution"] == "sorted" and r["algorithm"] == "Deterministic Quicksort"]
    ax.plot(*zip(*pts), marker="s", linestyle="--", label="deterministic, sorted")
    ns = sorted({r["n"] for r in rows})
    exact = [(2 * (n + 1) * harmonic(n) - 4 * n) / (n * math.log2(n)) for n in ns]
    ax.plot(ns, exact, color="black", linestyle=":", label="theory: 2(n+1)H_n - 4n")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("comparisons / (n lg n)")
    ax.set_title("Pivot comparisons divided by n lg n")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "quicksort_comparisons.png"), dpi=150)
    plt.close(fig)


def harmonic(n):
    return sum(1.0 / k for k in range(1, n + 1))


# ---------------------------------------------------------------- Part 2

class DivisionHashTable(HashTable):
    """Same table, but with the fixed division hash h(k) = k mod m."""

    def _index(self, key):
        return self._to_int(key) % len(self._slots)


def per_op_us(total_ms, count):
    return total_ms * 1000 / count


def chain_costs(table):
    """Expected items examined: (successful, unsuccessful), from chain lengths."""
    lengths = table.chain_lengths()
    n = sum(lengths)
    successful = sum(length * (length + 1) / 2 for length in lengths) / n
    unsuccessful = n / len(lengths)
    return successful, unsuccessful


def bench_load_factor(num_slots, alphas, repeats, rng):
    rows = []
    for alpha in alphas:
        n = int(alpha * num_slots)
        keys = rng.sample(range(10**12), 2 * n)
        present, absent = keys[:n], keys[n:]

        def build():
            table = HashTable(capacity=num_slots, resize=False, rng=random.Random(1))
            for key in present:
                table.insert(key, key)
            return table

        table = build()
        insert_ms = median_ms(build, repeats)
        hit_ms = median_ms(lambda: [table.search(k) for k in present], repeats)
        miss_ms = median_ms(lambda: [table.search(k) for k in absent], repeats)

        def delete_all():
            t = build()
            start = time.perf_counter()
            for key in present:
                t.delete(key)
            return (time.perf_counter() - start) * 1000

        delete_ms = statistics.median(delete_all() for _ in range(repeats))
        succ, unsucc = chain_costs(table)
        row = {"alpha": alpha, "n": n, "m": num_slots,
               "insert_us": round(per_op_us(insert_ms, n), 4),
               "search_hit_us": round(per_op_us(hit_ms, n), 4),
               "search_miss_us": round(per_op_us(miss_ms, n), 4),
               "delete_us": round(per_op_us(delete_ms, n), 4),
               "examined_hit": round(succ, 3),
               "theory_hit": round(1 + alpha / 2 - alpha / (2 * n), 3),
               "examined_miss": round(unsucc, 3),
               "max_chain": max(table.chain_lengths())}
        rows.append(row)
        print(f"  alpha={alpha:>5}  insert {row['insert_us']:.3f} us  "
              f"hit {row['search_hit_us']:.3f} us  miss {row['search_miss_us']:.3f} us  "
              f"delete {row['delete_us']:.3f} us  "
              f"examined hit {succ:.2f} (theory {row['theory_hit']:.2f})  "
              f"max chain {row['max_chain']}")
    return rows


def bench_scaling(sizes, fixed_slots, fixed_limit, repeats, rng):
    rows = []
    for n in sizes:
        keys = rng.sample(range(10**12), n)
        for mode in ("resizing", "fixed"):
            if mode == "fixed" and n > fixed_limit:
                continue

            def build():
                if mode == "resizing":
                    table = HashTable(rng=random.Random(1))
                else:
                    table = HashTable(capacity=fixed_slots, resize=False,
                                      rng=random.Random(1))
                for key in keys:
                    table.insert(key, key)
                return table

            table = build()
            insert_ms = median_ms(build, repeats)
            search_ms = median_ms(lambda: [table.search(k) for k in keys], repeats)
            row = {"mode": mode, "n": n, "final_slots": table.num_slots,
                   "final_alpha": round(table.load_factor, 3),
                   "insert_us": round(per_op_us(insert_ms, n), 4),
                   "search_us": round(per_op_us(search_ms, n), 4)}
            rows.append(row)
            print(f"  {mode:>8}  n={n:>7}  m={table.num_slots:>7}  "
                  f"alpha={row['final_alpha']:>7}  insert {row['insert_us']:.3f} us"
                  f"  search {row['search_us']:.3f} us")
    return rows


def bench_chains(num_slots, rng):
    """Chain-length histogram at alpha = 1 for three hash/key combinations."""
    random_keys = rng.sample(range(10**12), num_slots)
    strided_keys = [64 * i for i in range(num_slots)]  # e.g. aligned addresses
    cases = {
        "universal, random keys": (HashTable, random_keys),
        "universal, keys = 64i": (HashTable, strided_keys),
        "k mod m, keys = 64i": (DivisionHashTable, strided_keys),
    }
    results = {}
    for label, (cls, keys) in cases.items():
        table = cls(capacity=num_slots, resize=False, rng=random.Random(2))
        for key in keys:
            table.insert(key, key)
        lengths = table.chain_lengths()
        succ, _ = chain_costs(table)
        results[label] = lengths
        print(f"  {label:<24} empty slots {lengths.count(0):>5}  "
              f"max chain {max(lengths):>4}  avg items examined (hit) {succ:7.2f}")
    return results


def plot_hash(load_rows, scale_rows, chains, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    alphas = [r["alpha"] for r in load_rows]
    for key, label in [("insert_us", "insert"), ("search_hit_us", "successful search"),
                       ("search_miss_us", "unsuccessful search"), ("delete_us", "delete")]:
        ax1.plot(alphas, [r[key] for r in load_rows], marker="o", label=label)
    ax1.set_xlabel("load factor alpha = n / m")
    ax1.set_ylabel("time per operation (microseconds)")
    ax1.set_title(f"Fixed table, m = {load_rows[0]['m']}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(alphas, [r["examined_hit"] for r in load_rows], marker="o",
             label="successful: measured")
    ax2.plot(alphas, [1 + a / 2 for a in alphas], linestyle="--",
             label="successful: 1 + alpha/2")
    ax2.plot(alphas, [r["examined_miss"] for r in load_rows], marker="s",
             label="unsuccessful: measured")
    ax2.plot(alphas, alphas, linestyle=":", label="unsuccessful: alpha")
    ax2.set_xlabel("load factor alpha")
    ax2.set_ylabel("keys examined per search")
    ax2.set_title("Keys examined vs. theory")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "hash_load_factor.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, style in (("resizing", "-"), ("fixed", "--")):
        for key, label in (("insert_us", "insert"), ("search_us", "search")):
            pts = [(r["n"], r[key]) for r in scale_rows if r["mode"] == mode]
            ax.plot(*zip(*pts), marker="o", linestyle=style, label=f"{mode}: {label}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of keys n")
    ax.set_ylabel("time per operation (microseconds)")
    ax.set_title("Dynamic resizing vs. a fixed table of 1024 slots")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "hash_scaling.png"), dpi=150)
    plt.close(fig)

    top = max(max(lengths) for lengths in chains.values())
    poisson = [math.exp(-1) / math.factorial(k) for k in range(top + 1)]
    fig, axes = plt.subplots(1, len(chains), figsize=(13, 4), sharey=True, sharex=True)
    for ax, (label, lengths) in zip(axes, chains.items()):
        counts = [lengths.count(k) for k in range(top + 1)]
        ax.bar(range(top + 1), [c / len(lengths) for c in counts], label="measured")
        ax.plot(range(top + 1), poisson, color="black", marker=".",
                label="Poisson(1)")
        ax.set_title(f"{label}\nmax chain = {max(lengths)}")
        ax.set_xlabel("chain length")
        ax.set_xticks(range(0, top + 1, 2))
    axes[0].set_ylabel("fraction of slots")
    axes[0].legend()
    fig.suptitle("Chain lengths at alpha = 1")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "hash_chains.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- driver

def write_csv(name, rows):
    with open(os.path.join(RESULTS_DIR, name), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small sizes only")
    args = parser.parse_args()

    random.seed(532)  # makes the random pivots repeatable too
    rng = random.Random(532)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.quick:
        sort_sizes, sort_repeats = [250, 500, 1000, 2000], 3
        alphas, load_slots = [0.5, 1, 2, 4, 8], 500
        scale_sizes, hash_repeats = [1000, 4000, 16000], 3
        chain_slots = 2000
    else:
        sort_sizes, sort_repeats = [1000, 2000, 4000, 8000, 16000, 32000, 64000], 5
        alphas, load_slots = [0.25, 0.5, 1, 2, 4, 8, 16], 2000
        scale_sizes, hash_repeats = [2**k for k in range(10, 20)], 5
        chain_slots = 10000

    print(f"Python {sys.version.split()[0]}\n")
    print("Part 1: Quicksort")
    sort_rows = bench_quicksort(sort_sizes, sort_repeats, rng)
    print("\nPart 2a: hash table, growing load factor (no resizing)")
    load_rows = bench_load_factor(load_slots, alphas, hash_repeats, rng)
    print("\nPart 2b: hash table, growing n")
    scale_rows = bench_scaling(scale_sizes, 1024, 2**17, hash_repeats, rng)
    print("\nPart 2c: chain lengths at alpha = 1")
    chains = bench_chains(chain_slots, rng)

    write_csv("quicksort_results.csv", sort_rows)
    write_csv("hash_load_factor.csv", load_rows)
    write_csv("hash_scaling.csv", scale_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; skipping graphs.")
        return
    plot_quicksort(sort_rows, plt)
    plot_hash(load_rows, scale_rows, chains, plt)
    print(f"\nCSV files and graphs written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
