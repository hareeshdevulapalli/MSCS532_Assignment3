"""Hash table with chaining and a universal hash function.

Each slot of the table holds a chain: a Python list of [key, value] pairs.
Keys are mapped to slots with a function drawn at random from the
Carter-Wegman universal family (Cormen et al., 4th ed., Section 11.3.4)

    h_ab(k) = ((a * k + b) mod p) mod m,

where p = 2^61 - 1 is a prime larger than any key we hash, 1 <= a < p and
0 <= b < p are chosen at random, and m is the number of slots. For any two
distinct keys the probability (over the choice of a and b) that they collide
is at most 1/m, whatever the keys are.

The table keeps its load factor alpha = n / m between MIN_LOAD and MAX_LOAD
by doubling or halving m. A resize draws a fresh hash function and rehashes
every item, which costs Theta(n), but resizes happen rarely enough that
insert and delete still take O(1) amortized expected time.
"""

import random

_PRIME = (1 << 61) - 1  # Mersenne prime, larger than any 61-bit key


class HashTable:
    """A dictionary-like map built on separate chaining."""

    MAX_LOAD = 1.0   # grow when alpha would go above this
    MIN_LOAD = 0.25  # shrink when alpha drops below this
    MIN_SLOTS = 8

    def __init__(self, capacity=MIN_SLOTS, resize=True, rng=None):
        """Create an empty table with ``capacity`` slots.

        resize=False fixes the number of slots, which lets the benchmark
        measure how a growing load factor slows the operations down.
        """
        self._rng = rng or random.Random()
        self._resize_enabled = resize
        self._size = 0
        self._new_slots(max(1, capacity))

    # ----- hashing -----------------------------------------------------

    def _new_slots(self, num_slots):
        self._slots = [[] for _ in range(num_slots)]
        self._a = self._rng.randrange(1, _PRIME)
        self._b = self._rng.randrange(0, _PRIME)

    @staticmethod
    def _to_int(key):
        # Integers are hashed directly. Other hashable keys (strings, tuples,
        # ...) are first turned into an integer with Python's built-in hash;
        # the universal function then spreads those integers over the slots.
        if isinstance(key, int):
            return key % _PRIME
        return hash(key) % _PRIME

    def _index(self, key):
        return ((self._a * self._to_int(key) + self._b) % _PRIME) % len(self._slots)

    def _resize(self, num_slots):
        old_slots = self._slots
        self._new_slots(num_slots)
        for chain in old_slots:
            for key, value in chain:
                self._slots[self._index(key)].append([key, value])

    # ----- the three dictionary operations ------------------------------

    def insert(self, key, value):
        """Add key -> value, or replace the value if key is already present.

        Expected time O(1 + alpha) to scan the chain, O(1) amortized for the
        occasional resize.
        """
        chain = self._slots[self._index(key)]
        for pair in chain:
            if pair[0] == key:
                pair[1] = value
                return
        chain.append([key, value])
        self._size += 1
        if self._resize_enabled and self._size > self.MAX_LOAD * len(self._slots):
            self._resize(2 * len(self._slots))

    def search(self, key, default=None):
        """Return the value stored under key, or ``default`` if it is absent.

        Expected time Theta(1 + alpha) under simple uniform hashing.
        """
        for stored_key, value in self._slots[self._index(key)]:
            if stored_key == key:
                return value
        return default

    def delete(self, key):
        """Remove key and return its value. Raise KeyError if it is absent.

        Expected time Theta(1 + alpha): finding the pair dominates, removing
        it from a Python list is O(chain length) as well.
        """
        chain = self._slots[self._index(key)]
        for pos, (stored_key, value) in enumerate(chain):
            if stored_key == key:
                # Order inside a chain does not matter, so move the last pair
                # into the hole instead of shifting everything after it.
                chain[pos] = chain[-1]
                chain.pop()
                self._size -= 1
                if (self._resize_enabled
                        and len(self._slots) > self.MIN_SLOTS
                        and self._size < self.MIN_LOAD * len(self._slots)):
                    self._resize(max(self.MIN_SLOTS, len(self._slots) // 2))
                return value
        raise KeyError(key)

    # ----- convenience ---------------------------------------------------

    def __len__(self):
        return self._size

    def __contains__(self, key):
        sentinel = object()
        return self.search(key, sentinel) is not sentinel

    def __getitem__(self, key):
        sentinel = object()
        value = self.search(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        self.insert(key, value)

    def __delitem__(self, key):
        self.delete(key)

    def items(self):
        for chain in self._slots:
            for key, value in chain:
                yield key, value

    @property
    def num_slots(self):
        return len(self._slots)

    @property
    def load_factor(self):
        return self._size / len(self._slots)

    def chain_lengths(self):
        return [len(chain) for chain in self._slots]

    def __repr__(self):
        return (f"HashTable(n={self._size}, m={self.num_slots}, "
                f"alpha={self.load_factor:.2f})")


if __name__ == "__main__":
    table = HashTable()
    for word, count in [("apple", 3), ("banana", 7), ("cherry", 1), ("date", 4)]:
        table.insert(word, count)
    print(table)
    print("search('banana') ->", table.search("banana"))
    table.insert("banana", 8)
    print("after update, search('banana') ->", table.search("banana"))
    print("delete('apple') ->", table.delete("apple"))
    print("search('apple') ->", table.search("apple"))
    for i in range(100):
        table.insert(i, i * i)
    print(table, "max chain:", max(table.chain_lengths()))
