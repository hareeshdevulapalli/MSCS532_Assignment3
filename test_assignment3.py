"""Unit tests for quicksort.py and hash_table.py."""

import random
import unittest

from hash_table import HashTable
from quicksort import deterministic_quicksort, randomized_quicksort


def randomized_lomuto(arr):
    return randomized_quicksort(arr, three_way=False)


SORTS = [randomized_quicksort, deterministic_quicksort, randomized_lomuto]


class QuicksortTests(unittest.TestCase):
    def check(self, data):
        for sort in SORTS:
            with self.subTest(sort=sort.__name__, data=data[:10]):
                copy = list(data)
                result = sort(copy)
                self.assertIs(result, copy)  # sorted in place
                self.assertEqual(copy, sorted(data))

    def test_empty_and_single(self):
        self.check([])
        self.check([7])

    def test_two_items(self):
        self.check([2, 1])
        self.check([1, 2])
        self.check([1, 1])

    def test_sorted_and_reverse(self):
        self.check(list(range(1000)))
        self.check(list(range(1000, 0, -1)))

    def test_repeated_elements(self):
        self.check([4] * 500)
        self.check([3, 1, 3, 3, 2, 1, 3, 2, 2, 1])
        rng = random.Random(1)
        self.check([rng.randint(0, 4) for _ in range(2000)])

    def test_negatives_floats_strings(self):
        self.check([3, -2, 0.5, -7.25, 10, 0, -2])
        self.check(["pear", "apple", "fig", "apple", "kiwi"])

    def test_random(self):
        rng = random.Random(532)
        for size in (3, 10, 100, 1001, 5000):
            self.check([rng.randint(-10**6, 10**6) for _ in range(size)])

    def test_large_sorted_input_does_not_overflow_stack(self):
        # Deterministic Quicksort is Theta(n^2) here, but the smaller-side
        # recursion keeps the stack shallow, so this must not raise
        # RecursionError.
        data = list(range(3000))
        self.assertEqual(deterministic_quicksort(list(data)), data)

    def test_comparison_counts(self):
        n = 400
        stats = {"comparisons": 0}
        deterministic_quicksort(list(range(n)), stats=stats)
        self.assertEqual(stats["comparisons"], n * (n - 1) // 2)  # worst case

        stats = {"comparisons": 0}
        randomized_quicksort([5] * n, stats=stats)
        self.assertEqual(stats["comparisons"], n - 1)  # one three-way pass


class HashTableTests(unittest.TestCase):
    def test_insert_search_delete(self):
        table = HashTable()
        table.insert("a", 1)
        table.insert("b", 2)
        self.assertEqual(table.search("a"), 1)
        self.assertEqual(table.search("b"), 2)
        self.assertIsNone(table.search("c"))
        self.assertEqual(table.search("c", "missing"), "missing")
        self.assertEqual(table.delete("a"), 1)
        self.assertIsNone(table.search("a"))
        self.assertEqual(len(table), 1)

    def test_insert_existing_key_updates_value(self):
        table = HashTable()
        table.insert(42, "old")
        table.insert(42, "new")
        self.assertEqual(table.search(42), "new")
        self.assertEqual(len(table), 1)

    def test_delete_missing_key_raises(self):
        table = HashTable()
        with self.assertRaises(KeyError):
            table.delete("nope")
        table.insert(1, 1)
        table.delete(1)
        with self.assertRaises(KeyError):
            table.delete(1)

    def test_none_value_and_mixed_keys(self):
        table = HashTable()
        table[None] = "none key"
        table[(1, 2)] = None
        table[-5] = "negative"
        table[3.5] = "float"
        self.assertEqual(table[None], "none key")
        self.assertIn((1, 2), table)
        self.assertIsNone(table[(1, 2)])
        self.assertEqual(table[-5], "negative")
        self.assertEqual(table[3.5], "float")
        with self.assertRaises(KeyError):
            _ = table["absent"]

    def test_matches_dict_under_random_operations(self):
        rng = random.Random(7)
        table, reference = HashTable(rng=random.Random(8)), {}
        for _ in range(20000):
            key = rng.randint(-500, 500)
            op = rng.random()
            if op < 0.5:
                table[key] = reference[key] = rng.random()
            elif op < 0.8:
                self.assertEqual(table.search(key), reference.get(key))
            elif key in reference:
                self.assertEqual(table.delete(key), reference.pop(key))
            else:
                with self.assertRaises(KeyError):
                    table.delete(key)
            self.assertEqual(len(table), len(reference))
        self.assertEqual(dict(table.items()), reference)

    def test_resizing_keeps_load_factor_bounded(self):
        table = HashTable()
        for i in range(10000):
            table.insert(i, i)
            self.assertLessEqual(table.load_factor, HashTable.MAX_LOAD)
        self.assertEqual(sum(table.chain_lengths()), 10000)
        for i in range(9990):
            table.delete(i)
        self.assertGreaterEqual(table.load_factor, HashTable.MIN_LOAD)
        self.assertEqual(sorted(k for k, _ in table.items()), list(range(9990, 10000)))

    def test_fixed_size_table_does_not_resize(self):
        table = HashTable(capacity=16, resize=False)
        for i in range(160):
            table.insert(i, i)
        self.assertEqual(table.num_slots, 16)
        self.assertAlmostEqual(table.load_factor, 10.0)
        self.assertTrue(all(table.search(i) == i for i in range(160)))

    def test_universal_hash_spreads_adversarial_keys(self):
        # Keys that are all multiples of m would all land in slot 0 under
        # h(k) = k mod m. The universal function should still spread them.
        m = 1024
        table = HashTable(capacity=m, resize=False, rng=random.Random(3))
        for i in range(m):
            table.insert(i * m, i)
        self.assertLess(max(table.chain_lengths()), 10)


if __name__ == "__main__":
    unittest.main()
