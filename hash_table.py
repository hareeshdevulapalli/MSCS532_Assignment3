"""Hash table with chaining for MSCS 532 Assignment 3.

Every bucket is a plain Python list of [key, value] entries. To pick the
bucket for a key I use the Carter-Wegman universal hash family (CLRS 4th ed.
sec 11.3):

    h(k) = ((a*k + b) mod p) mod m

p = 2^61 - 1 is a prime bigger than any key I'll hash, a and b are random
(1 <= a < p, 0 <= b < p) and m is the number of buckets. For any two
different keys the chance they land in the same bucket is at most 1/m, and
that holds for any set of keys, not just random ones.

The table resizes itself to keep the load factor (n/m) between MIN_LOAD and
MAX_LOAD: it doubles when it gets too full and halves when it gets too empty.
Each resize picks new a and b and re-inserts everything. That's O(n) for the
resize, but it happens rarely enough that insert/delete are still O(1)
amortized.
"""

import random

MERSENNE_P = (1 << 61) - 1


class HashTable:
    """Key -> value map using separate chaining."""

    MAX_LOAD = 1.0    # double the buckets once n/m goes over this
    MIN_LOAD = 0.25   # halve them once n/m drops under this
    MIN_BUCKETS = 8

    def __init__(self, capacity=MIN_BUCKETS, resize=True, rng=None):
        """Make an empty table with `capacity` buckets.

        With resize=False the bucket count never changes. I only use that in
        the benchmark, to watch what happens as the load factor grows.
        """
        self._rand = rng or random.Random()
        self._auto_resize = resize
        self._count = 0
        self._make_buckets(max(1, capacity))

    # --- hashing helpers ---

    def _make_buckets(self, how_many):
        self._buckets = [[] for _ in range(how_many)]
        self._mult = self._rand.randrange(1, MERSENNE_P)
        self._shift = self._rand.randrange(0, MERSENNE_P)

    @staticmethod
    def _key_to_int(key):
        # ints go in as-is. Anything else (strings, tuples...) goes through
        # Python's hash() first, then the universal hash spreads it out.
        if isinstance(key, int):
            return key % MERSENNE_P
        return hash(key) % MERSENNE_P

    def _bucket_of(self, key):
        k = self._key_to_int(key)
        return ((self._mult * k + self._shift) % MERSENNE_P) % len(self._buckets)

    def _rehash(self, new_size):
        old = self._buckets
        self._make_buckets(new_size)
        for bucket in old:
            for key, value in bucket:
                self._buckets[self._bucket_of(key)].append([key, value])

    # --- the three main operations ---

    def insert(self, key, value):
        """Store key -> value. If the key is already there, update its value.

        Expected O(1 + alpha) because we walk the bucket to check for the key,
        plus O(1) amortized for resizing.
        """
        bucket = self._buckets[self._bucket_of(key)]
        for entry in bucket:
            if entry[0] == key:
                entry[1] = value
                return
        bucket.append([key, value])
        self._count += 1
        if self._auto_resize and self._count > self.MAX_LOAD * len(self._buckets):
            self._rehash(2 * len(self._buckets))

    def search(self, key, default=None):
        """Return the value for key, or `default` if the key isn't stored.

        Expected O(1 + alpha) with simple uniform hashing.
        """
        for k, v in self._buckets[self._bucket_of(key)]:
            if k == key:
                return v
        return default

    def delete(self, key):
        """Remove key and return its value. Raises KeyError if it's missing.

        Expected O(1 + alpha), mostly from finding the key.
        """
        bucket = self._buckets[self._bucket_of(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                # order inside a bucket doesn't matter, so just move the
                # last entry into this spot instead of shifting the list
                bucket[i] = bucket[-1]
                bucket.pop()
                self._count -= 1
                too_empty = self._count < self.MIN_LOAD * len(self._buckets)
                if (self._auto_resize and too_empty
                        and len(self._buckets) > self.MIN_BUCKETS):
                    self._rehash(max(self.MIN_BUCKETS, len(self._buckets) // 2))
                return v
        raise KeyError(key)

    # --- extras so it works like a dict ---

    def __len__(self):
        return self._count

    def __contains__(self, key):
        missing = object()
        return self.search(key, missing) is not missing

    def __getitem__(self, key):
        missing = object()
        found = self.search(key, missing)
        if found is missing:
            raise KeyError(key)
        return found

    def __setitem__(self, key, value):
        self.insert(key, value)

    def __delitem__(self, key):
        self.delete(key)

    def items(self):
        for bucket in self._buckets:
            for key, value in bucket:
                yield key, value

    @property
    def num_buckets(self):
        return len(self._buckets)

    @property
    def load_factor(self):
        return self._count / len(self._buckets)

    def bucket_sizes(self):
        return [len(b) for b in self._buckets]

    def __repr__(self):
        return (f"HashTable(n={self._count}, m={self.num_buckets}, "
                f"alpha={self.load_factor:.2f})")


if __name__ == "__main__":
    fruit = HashTable()
    for name, qty in [("apple", 3), ("banana", 7), ("cherry", 1), ("date", 4)]:
        fruit.insert(name, qty)
    print(fruit)
    print("search('banana') ->", fruit.search("banana"))
    fruit.insert("banana", 8)
    print("after update, search('banana') ->", fruit.search("banana"))
    print("delete('apple') ->", fruit.delete("apple"))
    print("search('apple') ->", fruit.search("apple"))
    for num in range(100):
        fruit.insert(num, num * num)
    print(fruit, "longest bucket:", max(fruit.bucket_sizes()))
