"""Unit tests for quicksort.py and hash_table.py.

Run with:  python3 -m unittest -v test_assignment3
"""

import random
import unittest

from hash_table import HashTable
from quicksort import deterministic_quicksort, randomized_quicksort


def lomuto_quicksort(nums):
    return randomized_quicksort(nums, three_way=False)


ALL_SORTS = [randomized_quicksort, deterministic_quicksort, lomuto_quicksort]


class TestQuicksort(unittest.TestCase):
    def assert_sorts(self, values):
        for sort_fn in ALL_SORTS:
            with self.subTest(sort=sort_fn.__name__, values=values[:10]):
                lst = list(values)
                out = sort_fn(lst)
                self.assertIs(out, lst)          # should sort in place
                self.assertEqual(lst, sorted(values))

    def test_empty_and_one_item(self):
        self.assert_sorts([])
        self.assert_sorts([7])

    def test_two_items(self):
        self.assert_sorts([2, 1])
        self.assert_sorts([1, 2])
        self.assert_sorts([1, 1])

    def test_already_sorted_and_reversed(self):
        self.assert_sorts(list(range(1000)))
        self.assert_sorts(list(range(1000, 0, -1)))

    def test_duplicates(self):
        self.assert_sorts([4] * 500)
        self.assert_sorts([3, 1, 3, 3, 2, 1, 3, 2, 2, 1])
        gen = random.Random(1)
        self.assert_sorts([gen.randint(0, 4) for _ in range(2000)])

    def test_negatives_floats_and_strings(self):
        self.assert_sorts([3, -2, 0.5, -7.25, 10, 0, -2])
        self.assert_sorts(["pear", "apple", "fig", "apple", "kiwi"])

    def test_random_lists(self):
        gen = random.Random(532)
        for size in (3, 10, 100, 1001, 5000):
            self.assert_sorts([gen.randint(-10**6, 10**6) for _ in range(size)])

    def test_big_sorted_list_no_recursion_error(self):
        # first-item pivot is n^2 here, but since we only recurse on the
        # smaller side the stack stays shallow and this shouldn't crash
        values = list(range(3000))
        self.assertEqual(deterministic_quicksort(list(values)), values)

    def test_comparison_counter(self):
        n = 400
        counts = {"comparisons": 0}
        deterministic_quicksort(list(range(n)), stats=counts)
        self.assertEqual(counts["comparisons"], n * (n - 1) // 2)  # worst case

        counts = {"comparisons": 0}
        randomized_quicksort([5] * n, stats=counts)
        self.assertEqual(counts["comparisons"], n - 1)  # done in one pass


class TestHashTable(unittest.TestCase):
    def test_basic_insert_search_delete(self):
        ht = HashTable()
        ht.insert("a", 1)
        ht.insert("b", 2)
        self.assertEqual(ht.search("a"), 1)
        self.assertEqual(ht.search("b"), 2)
        self.assertIsNone(ht.search("c"))
        self.assertEqual(ht.search("c", "missing"), "missing")
        self.assertEqual(ht.delete("a"), 1)
        self.assertIsNone(ht.search("a"))
        self.assertEqual(len(ht), 1)

    def test_insert_same_key_updates(self):
        ht = HashTable()
        ht.insert(42, "old")
        ht.insert(42, "new")
        self.assertEqual(ht.search(42), "new")
        self.assertEqual(len(ht), 1)

    def test_delete_missing_key(self):
        ht = HashTable()
        with self.assertRaises(KeyError):
            ht.delete("nope")
        ht.insert(1, 1)
        ht.delete(1)
        with self.assertRaises(KeyError):
            ht.delete(1)

    def test_odd_keys_and_none_values(self):
        ht = HashTable()
        ht[None] = "none key"
        ht[(1, 2)] = None
        ht[-5] = "negative"
        ht[3.5] = "float"
        self.assertEqual(ht[None], "none key")
        self.assertIn((1, 2), ht)
        self.assertIsNone(ht[(1, 2)])
        self.assertEqual(ht[-5], "negative")
        self.assertEqual(ht[3.5], "float")
        with self.assertRaises(KeyError):
            _ = ht["absent"]

    def test_same_as_dict_after_random_ops(self):
        gen = random.Random(7)
        ht, expected = HashTable(rng=random.Random(8)), {}
        for _ in range(20000):
            key = gen.randint(-500, 500)
            roll = gen.random()
            if roll < 0.5:
                ht[key] = expected[key] = gen.random()
            elif roll < 0.8:
                self.assertEqual(ht.search(key), expected.get(key))
            elif key in expected:
                self.assertEqual(ht.delete(key), expected.pop(key))
            else:
                with self.assertRaises(KeyError):
                    ht.delete(key)
            self.assertEqual(len(ht), len(expected))
        self.assertEqual(dict(ht.items()), expected)

    def test_resize_keeps_load_factor_in_range(self):
        ht = HashTable()
        for i in range(10000):
            ht.insert(i, i)
            self.assertLessEqual(ht.load_factor, HashTable.MAX_LOAD)
        self.assertEqual(sum(ht.bucket_sizes()), 10000)
        for i in range(9990):
            ht.delete(i)
        self.assertGreaterEqual(ht.load_factor, HashTable.MIN_LOAD)
        self.assertEqual(sorted(k for k, _ in ht.items()), list(range(9990, 10000)))

    def test_no_resize_when_turned_off(self):
        ht = HashTable(capacity=16, resize=False)
        for i in range(160):
            ht.insert(i, i)
        self.assertEqual(ht.num_buckets, 16)
        self.assertAlmostEqual(ht.load_factor, 10.0)
        self.assertTrue(all(ht.search(i) == i for i in range(160)))

    def test_universal_hash_with_bad_keys(self):
        # with h(k) = k mod m these keys would all go to bucket 0,
        # the universal hash should still spread them out
        m = 1024
        ht = HashTable(capacity=m, resize=False, rng=random.Random(3))
        for i in range(m):
            ht.insert(i * m, i)
        self.assertLess(max(ht.bucket_sizes()), 10)


if __name__ == "__main__":
    unittest.main()
