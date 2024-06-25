import unittest
import dwave_networkx as dnx
from src.utils import random_walk


class TestPlantedSolutions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p2 = dnx.pegasus_graph(2)

    def test_random_walk(self):
        walk = random_walk(self.p2, [1, 2, 3])
        self.assertEqual(walk[0], walk[-1])
        self.assertTrue(len(walk) >= 3)


if __name__ == '__main__':
    unittest.main()
