"""Randomized and deterministic Quicksort for MSCS 532 Assignment 3.

Both sorts use the same three-way partition, so the only thing that is
different between them is how the pivot gets picked:

    randomized_quicksort    - pivot is a random index in the current range
                              (RANDOMIZED-QUICKSORT, CLRS 4th ed. sec 7.3)
    deterministic_quicksort - pivot is always the first item in the range

The three-way partition splits the range into  < pivot | == pivot | > pivot
and we only keep sorting the two outer parts. That way lots of duplicate
values don't slow things down.

I also kept the normal two-way Lomuto partition from Chapter 7 around
(randomized_quicksort(nums, three_way=False)). It sorts correctly but gets
really slow (n^2) when most of the values are the same, which I use in the
benchmark to show why three-way is better.

All the sorts work in place and return the same list.
"""

import random


def three_way_split(nums, lo, hi):
    """Split nums[lo..hi] around the pivot nums[lo].

    Returns (eq_start, eq_stop) where after the call:
        nums[lo .. eq_start-1]   are smaller than the pivot
        nums[eq_start .. eq_stop] are equal to the pivot
        nums[eq_stop+1 .. hi]    are bigger than the pivot

    Every item gets compared to the pivot once, so it's O(n).
    I only scan left to right (like Lomuto) so items bigger than the pivot
    never move. Dijkstra's version swaps from the right end instead, which
    scrambles a sorted list and hides the worst case of the first-item
    pivot - found that out the hard way in testing.
    """
    pivot = nums[lo]
    eq_start = lo
    eq_stop = lo
    # nums[lo..eq_start-1]       < pivot
    # nums[eq_start..eq_stop]   == pivot
    # nums[eq_stop+1..pos-1]     > pivot
    # nums[pos..hi]              not looked at yet
    for pos in range(lo + 1, hi + 1):
        val = nums[pos]
        if val < pivot:
            # 3-way rotate so val ends up just before the equal block
            nums[pos] = nums[eq_stop + 1]
            nums[eq_stop + 1] = nums[eq_start]
            nums[eq_start] = val
            eq_start += 1
            eq_stop += 1
        elif val == pivot:
            eq_stop += 1
            nums[pos], nums[eq_stop] = nums[eq_stop], val
        # val > pivot: nothing to do, it's already on the right side
    return eq_start, eq_stop


def lomuto_split(nums, lo, hi):
    """Regular Lomuto PARTITION from CLRS, pivot is nums[hi].

    Returns the index where the pivot ends up. Anything equal to the pivot
    goes to the left side, so with lots of duplicates the split is very
    uneven.
    """
    pivot = nums[hi]
    boundary = lo - 1
    for pos in range(lo, hi):
        if nums[pos] <= pivot:
            boundary += 1
            nums[boundary], nums[pos] = nums[pos], nums[boundary]
    nums[boundary + 1], nums[hi] = nums[hi], nums[boundary + 1]
    return boundary + 1


def _quicksort_range(nums, lo, hi, use_random, three_way, counter):
    while lo < hi:
        if counter is not None:
            # count how many items get compared with the pivot this round
            counter["comparisons"] += hi - lo

        if use_random:
            r = random.randint(lo, hi)
            # three-way split wants the pivot at the front, Lomuto at the end
            spot = lo if three_way else hi
            nums[r], nums[spot] = nums[spot], nums[r]

        if three_way:
            eq_start, eq_stop = three_way_split(nums, lo, hi)
            left_hi = eq_start - 1
            right_lo = eq_stop + 1
        else:
            p = lomuto_split(nums, lo, hi)
            left_hi = p - 1
            right_lo = p + 1

        # Recurse on the smaller half and loop on the bigger one. Otherwise a
        # sorted list of a few thousand items blows past Python's recursion
        # limit with the first-item pivot. Same number of comparisons either way.
        if left_hi - lo < hi - right_lo:
            _quicksort_range(nums, lo, left_hi, use_random, three_way, counter)
            lo = right_lo
        else:
            _quicksort_range(nums, right_lo, hi, use_random, three_way, counter)
            hi = left_hi


def randomized_quicksort(nums, three_way=True, stats=None):
    """Sort nums in place using a random pivot each time, and return it.

    Expected time is O(n lg n) no matter what the input looks like.
    To count comparisons pass something like stats={"comparisons": 0}.
    """
    _quicksort_range(nums, 0, len(nums) - 1, True, three_way, stats)
    return nums


def deterministic_quicksort(nums, stats=None):
    """Sort nums in place using the first item as the pivot, and return it.

    Fine on random data (n lg n) but n^2 on sorted or reverse sorted data,
    since the pivot is then always the smallest or biggest value.
    """
    _quicksort_range(nums, 0, len(nums) - 1, False, True, stats)
    return nums


if __name__ == "__main__":
    examples = {
        "random": [38, 27, 43, 3, 9, 82, 10, 55],
        "sorted": [1, 2, 3, 4, 5, 6, 7, 8],
        "reverse": [8, 7, 6, 5, 4, 3, 2, 1],
        "repeated": [5, 1, 5, 3, 5, 1, 3, 5],
        "empty": [],
    }
    for label, values in examples.items():
        print(f"{label:>8}: {values} -> {randomized_quicksort(list(values))}")
