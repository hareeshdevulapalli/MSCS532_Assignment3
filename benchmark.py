"""Timing experiments for Assignment 3.

Part 1: times Randomized Quicksort vs Deterministic Quicksort (first item as
pivot) on random, sorted, reverse sorted and lots-of-duplicates lists, and
counts how many comparisons each one makes. I also run a randomized version
with the two-way Lomuto partition to show what duplicates do to it.

Part 2: hash table with chaining
  - time per operation as the load factor n/m goes up (m stays fixed)
  - time per operation as n goes up, with resizing on and off
  - how long the buckets get with the universal hash vs plain k mod m

Every time is the median of a few runs (time.perf_counter). CSVs and graphs
go into results/.

    python3 benchmark.py            # full run, ~3 minutes
    python3 benchmark.py --quick    # small sizes, few seconds
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


def lomuto_quicksort(arr, stats=None):
    return randomized_quicksort(arr, three_way=False, stats=stats)


SORT_FUNCS = {
    "Randomized Quicksort": randomized_quicksort,
    "Deterministic Quicksort": deterministic_quicksort,
    "Randomized (Lomuto)": lomuto_quicksort,
}
INPUT_KINDS = ["random", "sorted", "reverse", "repeated"]
MAX_RUN_MS = 1000  # if one run takes longer than this, skip the bigger sizes


def build_input(kind, n, rng):
    if kind == "random":
        return [rng.randint(0, 10 * n) for _ in range(n)]
    if kind == "sorted":
        return list(range(n))
    if kind == "reverse":
        return list(range(n, 0, -1))
    if kind == "repeated":
        return [rng.randint(0, 9) for _ in range(n)]  # just 10 different values
    raise ValueError(kind)


def median_time_ms(func, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


# ===== Part 1: quicksort =====

def time_sorts(sizes, repeats, rng):
    rows = []
    for dist in INPUT_KINDS:
        for name, sort in SORT_FUNCS.items():
            for n in sizes:
                data = build_input(dist, n, rng)

                def run():
                    copy = list(data)
                    sort(copy)
                    return copy

                assert run() == sorted(data)
                ms = median_time_ms(run, repeats)
                stats = {"comparisons": 0}
                sort(list(data), stats=stats)
                rows.append({"distribution": dist, "algorithm": name, "n": n,
                             "time_ms": round(ms, 3),
                             "comparisons": stats["comparisons"]})
                print(f"  {dist:>8}  {name:<24} n={n:>6}  {ms:9.2f} ms"
                      f"  {stats['comparisons']:>11,} comparisons")
                if ms > MAX_RUN_MS:
                    break  # n^2 case, bigger n would take forever
    return rows


def plot_quicksort(rows, plt):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, dist in zip(axes.flat, INPUT_KINDS):
        for name in SORT_FUNCS:
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
    for dist in INPUT_KINDS:
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


# ===== Part 2: hash table =====

class ModHashTable(HashTable):
    """Same table but hashing with plain k mod m (for comparison)."""

    def _bucket_of(self, key):
        return self._key_to_int(key) % len(self._buckets)


def micros_per_op(total_ms, count):
    return total_ms * 1000 / count


def avg_keys_checked(ht):
    """Average keys checked per (hit, miss) search, worked out from bucket sizes."""
    lengths = ht.bucket_sizes()
    n = sum(lengths)
    successful = sum(length * (length + 1) / 2 for length in lengths) / n
    unsuccessful = n / len(lengths)
    return successful, unsuccessful


def measure_load_factor(bucket_count, alphas, repeats, rng):
    rows = []
    for alpha in alphas:
        n = int(alpha * bucket_count)
        keys = rng.sample(range(10**12), 2 * n)
        present, absent = keys[:n], keys[n:]

        def build():
            ht = HashTable(capacity=bucket_count, resize=False, rng=random.Random(1))
            for key in present:
                ht.insert(key, key)
            return ht

        ht = build()
        insert_ms = median_time_ms(build, repeats)
        hit_ms = median_time_ms(lambda: [ht.search(k) for k in present], repeats)
        miss_ms = median_time_ms(lambda: [ht.search(k) for k in absent], repeats)

        def delete_all():
            t = build()
            start = time.perf_counter()
            for key in present:
                t.delete(key)
            return (time.perf_counter() - start) * 1000

        delete_ms = statistics.median(delete_all() for _ in range(repeats))
        succ, unsucc = avg_keys_checked(ht)
        row = {"alpha": alpha, "n": n, "m": bucket_count,
               "insert_us": round(micros_per_op(insert_ms, n), 4),
               "search_hit_us": round(micros_per_op(hit_ms, n), 4),
               "search_miss_us": round(micros_per_op(miss_ms, n), 4),
               "delete_us": round(micros_per_op(delete_ms, n), 4),
               "examined_hit": round(succ, 3),
               "theory_hit": round(1 + alpha / 2 - alpha / (2 * n), 3),
               "examined_miss": round(unsucc, 3),
               "max_chain": max(ht.bucket_sizes())}
        rows.append(row)
        print(f"  alpha={alpha:>5}  insert {row['insert_us']:.3f} us  "
              f"hit {row['search_hit_us']:.3f} us  miss {row['search_miss_us']:.3f} us  "
              f"delete {row['delete_us']:.3f} us  "
              f"examined hit {succ:.2f} (theory {row['theory_hit']:.2f})  "
              f"max chain {row['max_chain']}")
    return rows


def measure_growth(sizes, fixed_buckets, fixed_limit, repeats, rng):
    rows = []
    for n in sizes:
        keys = rng.sample(range(10**12), n)
        for mode in ("resizing", "fixed"):
            if mode == "fixed" and n > fixed_limit:
                continue

            def build():
                if mode == "resizing":
                    ht = HashTable(rng=random.Random(1))
                else:
                    ht = HashTable(capacity=fixed_buckets, resize=False,
                                   rng=random.Random(1))
                for key in keys:
                    ht.insert(key, key)
                return ht

            ht = build()
            insert_ms = median_time_ms(build, repeats)
            search_ms = median_time_ms(lambda: [ht.search(k) for k in keys], repeats)
            row = {"mode": mode, "n": n, "final_slots": ht.num_buckets,
                   "final_alpha": round(ht.load_factor, 3),
                   "insert_us": round(micros_per_op(insert_ms, n), 4),
                   "search_us": round(micros_per_op(search_ms, n), 4)}
            rows.append(row)
            print(f"  {mode:>8}  n={n:>7}  m={ht.num_buckets:>7}  "
                  f"alpha={row['final_alpha']:>7}  insert {row['insert_us']:.3f} us"
                  f"  search {row['search_us']:.3f} us")
    return rows


def measure_bucket_spread(bucket_count, rng):
    """Bucket sizes at alpha = 1 for three hash function / key combos."""
    random_keys = rng.sample(range(10**12), bucket_count)
    strided_keys = [64 * i for i in range(bucket_count)]  # evenly spaced keys, like aligned memory addresses
    cases = {
        "universal, random keys": (HashTable, random_keys),
        "universal, keys = 64i": (HashTable, strided_keys),
        "k mod m, keys = 64i": (ModHashTable, strided_keys),
    }
    results = {}
    for label, (cls, keys) in cases.items():
        ht = cls(capacity=bucket_count, resize=False, rng=random.Random(2))
        for key in keys:
            ht.insert(key, key)
        lengths = ht.bucket_sizes()
        succ, _ = avg_keys_checked(ht)
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
    ax1.set_title(f"Fixed ht, m = {load_rows[0]['m']}")
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
    ax.set_title("Dynamic resizing vs. a fixed ht of 1024 slots")
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


# ===== main =====

def write_csv(name, rows):
    with open(os.path.join(RESULTS_DIR, name), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small sizes only")
    args = parser.parse_args()

    random.seed(532)  # so the random pivots are the same every run
    rng = random.Random(532)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.quick:
        sort_sizes, sort_repeats = [250, 500, 1000, 2000], 3
        alphas, load_buckets = [0.5, 1, 2, 4, 8], 500
        scale_sizes, hash_repeats = [1000, 4000, 16000], 3
        spread_buckets = 2000
    else:
        sort_sizes, sort_repeats = [1000, 2000, 4000, 8000, 16000, 32000, 64000], 5
        alphas, load_buckets = [0.25, 0.5, 1, 2, 4, 8, 16], 2000
        scale_sizes, hash_repeats = [2**k for k in range(10, 20)], 5
        spread_buckets = 10000

    print(f"Python {sys.version.split()[0]}\n")
    print("Part 1: Quicksort")
    sort_rows = time_sorts(sort_sizes, sort_repeats, rng)
    print("\nPart 2a: hash ht, growing load factor (no resizing)")
    load_rows = measure_load_factor(load_buckets, alphas, hash_repeats, rng)
    print("\nPart 2b: hash ht, growing n")
    scale_rows = measure_growth(scale_sizes, 1024, 2**17, hash_repeats, rng)
    print("\nPart 2c: chain lengths at alpha = 1")
    chains = measure_bucket_spread(spread_buckets, rng)

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
