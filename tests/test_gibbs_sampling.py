import unittest
import numpy as np
from src.utils import gibbs_sampling_ising
from itertools import product
from dimod import BinaryQuadraticModel
from math import exp


rng = np.random.default_rng()


class GibbsSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spins = 5
        cls.distribution = {}
        cls.beta = 1
        cls.h = {i: rng.uniform(-1, 1) for i in range(spins)}
        cls.J = {(i, i+1): rng.uniform(-1, 1) for i in range(spins-1)}
        # bqm = BinaryQuadraticModel("SPIN")
        # bqm = bqm.from_ising(cls.h, cls.J)
        # for config in product([-1, 1], repeat=spins):
        #     state = {idx: spin for idx, spin in enumerate(config)}
        #     energy = bqm.energy(state)
        #     cls.distribution[state] = exp(-cls.beta * energy)
        # z = sum(list(cls.distribution.values()))
        # for state, energy in cls.distribution.items():
        #     cls.distribution[state] = energy/z

    def test_gibbs_sampling(self):
        sample = gibbs_sampling_ising(self.h, self.J, self.beta, 100)


if __name__ == '__main__':
    unittest.main()
