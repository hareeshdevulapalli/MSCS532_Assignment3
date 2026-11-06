# MSCS532 Assignment 3: Randomized Quicksort and Hashing with Chaining

This project implements Randomized Quicksort and a hash table with chaining in
Python, analyzes their expected running times, and checks the analysis with
experiments. The full write-up is in `MSCS532_Assignment3_Report.docx`.

| File | Contents |
|---|---|
| `quicksort.py` | Randomized Quicksort (uniform random pivot) and Deterministic Quicksort (first-element pivot), both with a three-way partition; a two-way Lomuto variant for comparison |
| `hash_table.py` | `HashTable`: chaining, a Carter–Wegman universal hash function, insert / search / delete, dynamic resizing |
| `test_assignment3.py` | Unit tests for both parts |
| `benchmark.py` | Timing and counting experiments; writes CSV files and graphs to `results/` |
| `results/` | CSV data, graphs, and `benchmark_output.txt` from the last full run |
| `MSCS532_Assignment3_Report.docx` | Report: implementation, analysis, empirical results, discussion |

## Requirements

Python 3.8 or later. The algorithms and tests use only the standard library.
`benchmark.py` needs `matplotlib` for the graphs (it still writes the CSV files
without it):

```bash
pip install matplotlib
```

## Running the code

```bash
python3 quicksort.py                  # small demo on each kind of input
python3 hash_table.py                 # insert / search / update / delete demo
python3 -m unittest -v test_assignment3
python3 benchmark.py --quick          # small sizes, a few seconds
python3 benchmark.py                  # full run, about 3 minutes
```

Using the code from another script:

```python
from quicksort import randomized_quicksort, deterministic_quicksort
from hash_table import HashTable

data = [5, 3, 8, 3, 1]
randomized_quicksort(data)            # sorts in place and returns the list

table = HashTable()
table.insert("apple", 3)
table.search("apple")                 # 3
table.delete("apple")                 # returns 3; KeyError if absent
```

## Part 1: Randomized Quicksort

**Implementation.** The pivot is chosen with `random.randint(first, last)`.
The partition splits each subarray into `< pivot | == pivot | > pivot`, so
repeated keys are handled in one pass. The sort recurses into the smaller part
and loops on the larger one, which keeps the stack depth at O(lg n) even in the
worst case. Empty lists, single items, sorted lists, and all-equal lists are
all covered by the tests.

**Analysis.** With indicator variables X_ij = 1 if the i-th and j-th smallest
keys are ever compared, Pr{X_ij = 1} = 2/(j − i + 1), because they are compared
only if one of them is the first pivot chosen from the keys between them. So

```
E[comparisons] = Σ_{i<j} 2/(j − i + 1) = 2(n+1)H_n − 4n ≈ 1.39 n lg n = O(n lg n)
```

for every input. A first-element pivot on sorted input gives
T(n) = T(n−1) + Θ(n) = Θ(n²).

**Results** (Python 3.13, Apple Silicon, median of 5 runs, n = 8,000, times in ms):

| Algorithm | Random | Sorted | Reverse | Repeated (10 keys) |
|---|---|---|---|---|
| Randomized Quicksort | 7.85 | 7.28 | 7.38 | 1.30 |
| Deterministic Quicksort (first pivot) | 6.70 | 715.0 | 1,967.2 | 1.33 |
| Randomized, two-way Lomuto | 5.54 | 5.16 | 5.09 | 105.6 |

![Quicksort running time](results/quicksort_time.png)

- Deterministic Quicksort is Θ(n²) on sorted and reverse input: exactly
  n(n−1)/2 comparisons, and time ×4 per doubling of n.
- Randomized Quicksort runs in the same time on every order. Its comparison
  count was within 1% of 2(n+1)H_n − 4n at n = 64,000.
- On random input the deterministic version is slightly faster, because
  calling `random.randint` per partition costs time in Python.
- Reverse input is ~3× slower than sorted input for the deterministic version
  even with identical comparison counts: smaller-than-pivot items need a
  3-write rotation, larger ones are not moved.
- The textbook two-way partition goes quadratic on repeated keys even with
  random pivots (the O(n lg n) proof assumes distinct keys). The three-way
  partition makes repeated keys the fastest case.

## Part 2: Hash table with chaining

**Implementation.** Each slot holds a list of `[key, value]` pairs. Slots are
chosen by h(k) = ((a·k + b) mod p) mod m with p = 2⁶¹ − 1 and random a, b,
redrawn on every resize. The table doubles when α = n/m would exceed 1 and
halves when α drops below 1/4.

**Analysis** (simple uniform hashing, α = n/m):

| Operation | Expected time |
|---|---|
| Unsuccessful search | Θ(1 + α) — scans a whole chain of expected length α |
| Successful search | 1 + α/2 − α/(2n) keys examined = Θ(1 + α) |
| Insert | Θ(1 + α) here (checks for an existing key), plus O(1) amortized for resizing |
| Delete | Θ(1 + α) to find the key, O(1) to remove it |

Keeping α ≤ 1 by doubling makes every operation O(1) expected; the total
rehashing work over n inserts is under 2n.

**Results.**

- Keys examined per search matched theory almost exactly: 1.11 vs 1.12 at
  α = 0.25, 8.99 vs 9.00 at α = 16.
- Time grows with α, but with a large constant term (hash computation and
  call overhead in Python): unsuccessful search went from 0.192 µs to 0.510 µs
  as α went from 0.25 to 16.
- With resizing, α stayed ≤ 1 and search stayed between 0.23 and 0.55 µs as n
  grew from 1,024 to 524,288. A fixed 1,024-slot table reached 1.54 µs at
  n = 131,072 (α = 128). The remaining growth with resizing is from CPU cache
  misses, not the algorithm.
- With keys that are multiples of 64 and m = 10,000, `k mod m` used only 625
  slots (chains of 16); the universal hash spread the same keys with a
  longest chain of 2.

![Hash table load factor](results/hash_load_factor.png)
![Resizing vs. fixed table](results/hash_scaling.png)
![Chain lengths](results/hash_chains.png)
