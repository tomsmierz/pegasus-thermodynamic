import numpy as np
import pandas as pd
from dimod import BinaryQuadraticModel
from collections import OrderedDict

rng = np.random.default_rng()


# def pseudo_likelihood(beta_eff, samples):
#     J = - 1.0
#     L = 0.0
#     N = samples.shape[1]
#     D = samples.shape[0]
#     for i in range(D-1):
#         for j in range(N-1):
#             if j == 0:
#                 L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j+1])*beta_eff))
#             elif j == N-1:
#                 L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j-1])*beta_eff))
#             else:
#                 L += -np.log(1+np.exp(-2*(J*samples[i,j]*samples[i,j+1]+J*samples[i,j]*samples[i,j-1])*beta_eff))
#     return -L/(N*D)

def neighbour(i: int, N: int) -> list:
    if i == 0:
        return [1]
    elif i == N - 1:
        return [N - 2]
    else:
        return [i-1, i+1]

def extend(J: dict) -> dict:
    J_new = {}
    for a,b in J.keys():
        J_new[(a, b)] = J[(a, b)]
        J_new[(b, a)] = J[(a, b)]
    return J_new


def pseudo_likelihood(beta_eff: float, h: dict, J: dict, samples: np.ndarray):
    N = samples.shape[1]
    D = samples.shape[0]
    L = 0.0

    for d in range(D):
        for i in range(N):
            L += np.log(1 + np.exp(-2 * beta_eff * samples[d, i] *
                                   (h[i] + sum([J[(i, j)] * samples[d, j] for j in neighbour(i, N)]))))
    return L/(N * D)


def gibbs_sampling_ising(h: dict, J: dict, beta: float, num_steps: int):
    s = OrderedDict({i: rng.choice([-1, 1]) for i in h.keys()})

    bqm = BinaryQuadraticModel("SPIN")
    bqm = bqm.from_ising(h, J)
    nodes = list(bqm.variables)

    for _ in range(num_steps):
        pos = rng.choice(nodes)  # we chose an index

        s_plus = {k: v for k, v in s.items()}
        s_plus[pos] = 1
        s_minus = {k: v for k, v in s.items()}
        s_minus[pos] = -1

        deltaE = bqm.energy(s_plus) - bqm.energy(s_minus)
        prob = 1/(1+np.exp(beta*deltaE))  # P(s_i = 1| s_-i)
        s[pos] = rng.choice([-1, 1], p=[1-prob, prob])

    return s


def vectorize(h: dict, J: dict):
    # We assume that h an J are sorted
    h_vect = np.array(list(h.values()))
    n = len(h_vect)
    J_vect = np.zeros((n, n))
    for key, value in J.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect


def energy(s: np.ndarray, h: np.ndarray, J: np.ndarray):
    return np.dot(np.dot(s, J), s) + np.dot(s, h)

