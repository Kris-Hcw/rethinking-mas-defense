import unittest

import numpy as np

from src.topology import build_adjacency


class TopologyTests(unittest.TestCase):
    def test_chain_has_no_return_edge(self):
        adjacency = build_adjacency("chain", 5)
        expected = np.zeros((5, 5), dtype=int)
        for index in range(4):
            expected[index, index + 1] = 1
        np.testing.assert_array_equal(adjacency, expected)
        self.assertEqual(adjacency[4, 0], 0)


if __name__ == "__main__":
    unittest.main()
