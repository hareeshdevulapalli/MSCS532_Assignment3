"""Randomized and deterministic Quicksort.

Both sorts share the same three-way partition (Dijkstra's "Dutch national
flag" scheme), so the only difference between them is how the pivot is
chosen:

* randomized_quicksort picks the pivot uniformly at random from the
  subarray being partitioned (RANDOMIZED-QUICKSORT, Cormen et al., 4th ed.,
  Section 7.3).
* deterministic_quicksort always uses the first element of the subarray.

Three-way partitioning splits a subarray into  < pivot | == pivot | > pivot
and only recurses on the two outer parts. Every copy of the pivot value is
put in its final place in one pass, so arrays with many repeated keys do not
degrade to quadratic time.

For comparison, randomized_quicksort(arr, three_way=False) uses the
textbook two-way Lomuto PARTITION from Chapter 7 instead. It is correct but
slows to Theta(n^2) when most keys are equal; the benchmark uses it to show
why the three-way scheme matters.

All sorts work in place, return the list they were given, and keep the call
stack at O(lg n) depth by recursing into the smaller part and looping on the
larger one.
"""

import random


def partition_three_way(arr, first, last):
    """Partition arr[first..last] around the pivot arr[first].

    Returns (lt, gt) such that afterwards
        arr[first..lt-1] <  pivot,
        arr[lt..gt]      == pivot,
        arr[gt+1..last]  >  pivot.
    Each of the last - first non-pivot items is compared with the pivot once
    (one or two Python comparisons), so the pass takes Theta(n) time.

    The scan moves only forward, like Lomuto's PARTITION, so items larger
    than the pivot are never moved and keep their order. (Dijkstra's version
    swaps them in from the far end, which shuffles sorted input and would
    hide the worst case of a first-element pivot.)
    """
    pivot = arr[first]
    lt, eq_end = first, first
    # Invariant: arr[first..lt-1] < pivot, arr[lt..eq_end] == pivot,
    # arr[eq_end+1..idx-1] > pivot, arr[idx..last] not yet examined.
    for idx in range(first + 1, last + 1):
        item = arr[idx]
        if item < pivot:
            # Rotate: first ">" item -> idx, first "==" item -> end of the
            # "==" block, item -> front of the "==" block.
            arr[idx] = arr[eq_end + 1]
            arr[eq_end + 1] = arr[lt]
            arr[lt] = item
            lt += 1
            eq_end += 1
        elif item == pivot:
            eq_end += 1
            arr[idx], arr[eq_end] = arr[eq_end], item
    return lt, eq_end


def partition_lomuto(arr, first, last):
    """Textbook Lomuto PARTITION of arr[first..last] around arr[last].

    Returns the pivot's final index q. Items equal to the pivot all go to
    the left part, which is why many repeated keys make it lopsided.
    """
    pivot = arr[last]
    small_end = first - 1
    for idx in range(first, last):
        if arr[idx] <= pivot:
            small_end += 1
            arr[small_end], arr[idx] = arr[idx], arr[small_end]
    arr[small_end + 1], arr[last] = arr[last], arr[small_end + 1]
    return small_end + 1


def _sort(arr, first, last, randomized, three_way, stats):
    while first < last:
        if stats is not None:
            # Number of items compared with the pivot in this partition.
            stats["comparisons"] += last - first
        if randomized:
            pick = random.randint(first, last)  # uniform over the subarray
            target = first if three_way else last
            arr[pick], arr[target] = arr[target], arr[pick]

        if three_way:
            left_end, right_start = partition_three_way(arr, first, last)
            left_end -= 1
            right_start += 1
        else:
            q = partition_lomuto(arr, first, last)
            left_end, right_start = q - 1, q + 1

        # Recurse into the smaller part, loop on the larger one. This keeps
        # the stack depth at O(lg n) even when the partitions are lopsided,
        # without changing the number of comparisons.
        if left_end - first < last - right_start:
            _sort(arr, first, left_end, randomized, three_way, stats)
            first = right_start
        else:
            _sort(arr, right_start, last, randomized, three_way, stats)
            last = left_end


def randomized_quicksort(arr, three_way=True, stats=None):
    """Sort ``arr`` in place using uniformly random pivots and return it.

    Expected running time is O(n lg n) for every input. Pass a dict such as
    {"comparisons": 0} as ``stats`` to count pivot comparisons.
    """
    _sort(arr, 0, len(arr) - 1, True, three_way, stats)
    return arr


def deterministic_quicksort(arr, stats=None):
    """Sort ``arr`` in place using the first element as the pivot.

    Theta(n lg n) on typical random input, but Theta(n^2) on sorted and
    reverse-sorted input, where every pivot is the smallest or largest key.
    """
    _sort(arr, 0, len(arr) - 1, False, True, stats)
    return arr


if __name__ == "__main__":
    samples = {
        "random": [38, 27, 43, 3, 9, 82, 10, 55],
        "sorted": [1, 2, 3, 4, 5, 6, 7, 8],
        "reverse": [8, 7, 6, 5, 4, 3, 2, 1],
        "repeated": [5, 1, 5, 3, 5, 1, 3, 5],
        "empty": [],
    }
    for name, data in samples.items():
        print(f"{name:>8}: {data} -> {randomized_quicksort(list(data))}")
