import unittest
import numpy as np
from src.utils import vectorize, energy
from dimod import BinaryQuadraticModel

rng = np.random.default_rng()


class EnergyEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.size = 300
        cls.h = {i: rng.uniform(-1, 1) for i in range(cls.size)}
        cls.J = {(i, i+1): rng.uniform(-1, 1) for i in range(cls.size-1)}

    def test_vectorisation(self):
        h_vect, J_vect = vectorize(self.h, self.J)
        state = {i: rng.choice([-1, 1]) for i in range(self.size)}
        state_vect = np.array([v for v in state.values()])
        energy_naive = 0
        for s, h in self.h.items():
            energy_naive += state[s] * h
        for (e1, e2), j in self.J.items():
            energy_naive += state[e1] * state[e2] * j

        energy_vect = energy(state_vect, h_vect, J_vect)
        self.assertAlmostEqual(energy_naive, energy_vect)

    def test_bqm(self):
        state = {i: rng.choice([-1, 1]) for i in range(self.size)}
        energy_naive = 0
        for s, h in self.h.items():
            energy_naive += state[s] * h
        for (e1, e2), j in self.J.items():
            energy_naive += state[e1] * state[e2] * j
        bqm = BinaryQuadraticModel("SPIN")
        bqm = bqm.from_ising(self.h, self.J)
        self.assertAlmostEqual(bqm.energy(state), energy_naive)


if __name__ == '__main__':
    unittest.main()
