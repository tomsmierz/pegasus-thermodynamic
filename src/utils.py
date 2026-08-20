import json
from copy import copy

import minorminer

import networkx as nx
import numpy as np
import pandas as pd
import h5py

from dimod import BinaryQuadraticModel
from collections import OrderedDict, defaultdict
from tqdm import tqdm

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


def pseudo_likelihood(
    beta_eff: float,
    h: dict,
    J: dict,
    samples: np.ndarray,
    h_vect: np.ndarray | None = None,
    J_vect: np.ndarray | None = None,
):
    beta_eff_scalar = float(np.asarray(beta_eff).reshape(-1)[0])
    if h_vect is None or J_vect is None:
        J_ext = extend(J)
        h_vect, J_vect = vectorize(h, J_ext)
    return pseudo_likelihood_2d_vectorised(beta_eff_scalar, h_vect, J_vect, samples)


def pseudo_likelihood_2d(beta_eff: float, h: dict, J: dict, samples: np.ndarray):
    N = samples.shape[1]
    D = samples.shape[0]
    L = 0.0

    for d in range(D):
        for (i, j) in J.keys():
            L += np.log(1 + np.exp(-2 * beta_eff * samples[d, i] *
                                   (h[i] + sum([J[(i, j)] * samples[d, j] ]))))
    return L/(N * D)


def pseudo_likelihood_2d_vectorised(beta_eff: float, h: np.ndarray, J: np.ndarray, samples: np.ndarray):
    N = samples.shape[1]
    D = samples.shape[0]

    S = samples.T  # we need states to be in columns vector
    h = h.reshape((len(h), 1))  # column vector

    A = -2 * beta_eff * S * (J @ S + h)
    B = np.log(1 + np.exp(A))
    L = np.sum(B)
    return L / (N * D)


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


def gibbs_sampling_ising_vectorized_2d(h: dict, J: dict, beta: float, num_steps: int):
    h_vect, J_vect, _, _, _ = vectorize_2d(h, J)
    n = len(h)
    spins = rng.choice([-1, 1], size=n)

    for _ in tqdm(range(num_steps), desc="gibbs sampling"):
        idx = rng.integers(0, n)
        s_plus = copy(spins)
        s_minus = copy(spins)
        s_plus[idx] = +1
        s_minus[idx] = -1
        deltaE = energy(s_plus, h_vect, J_vect) - energy(s_minus, h_vect, J_vect)
        prob = 1 / (1 + np.exp(beta * deltaE))  # P(s_i = 1| s_-i)
        spins[idx] = rng.choice([-1, 1], p=[1 - prob, prob])
    return spins


def find_neighbours(h: dict, J: dict):
    neighbourhood = {node: [] for node in h.keys()}
    for (u, v), value in J.items():
        neighbourhood[u].append(v)
        neighbourhood[v].append(u)
    return neighbourhood


def stable_sigmoid(x):
    """Numerically stable sigmoid function."""
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def gibbs_sampling_efficient(h: dict, J: dict, beta: float, num_steps: int):

    spins = OrderedDict({i: rng.choice([-1, 1]) for i in h.keys()})
    neighbourhood = find_neighbours(h, J)
    # for _ in tqdm(range(num_steps), desc="gibbs sampling"):
    for _ in range(num_steps):
        idx = rng.choice(list(h.keys()), size=1)
        idx = idx.item()
        neighbours = neighbourhood[idx]
        sum = 0
        for j in neighbours:
            J_ij = J[(idx, j)] if (idx, j) in J.keys() else J[(j, idx)]
            sum += J_ij * spins[j]
        # Difference s+ and s-
        deltaE = 2 * (sum + h[idx])
        # prob = 1 / (1 + np.exp(beta * deltaE))  # P(s_i = 1| s_-i)
        prob = stable_sigmoid(beta * deltaE)
        spins[idx] = rng.choice([-1, 1], p=[1 - prob, prob])
    return spins


def vectorize(h: dict, J: dict):
    # We assume that h an J are sorted
    h_vect = np.array(list(h.values()))
    n = len(h_vect)
    J_vect = np.zeros((n, n))
    for key, value in J.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect


def vectorize_2d(h: dict, J: dict):
    key_map = {key: i for i, key in enumerate(h.keys())} 
    h_new = {i: h[key] for i, key in enumerate(h.keys())}
    J_new = {(i, j): J[(key_i, key_j)] for i, key_i in enumerate(h.keys()) for j, key_j in enumerate(h.keys()) if (key_i, key_j) in J}
    n = len(h_new)
    h_vect = np.array(list(h_new.values()))
    J_vect = np.zeros((n, n))
    for key, value in J_new.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect, h_new, J_new, key_map


def energy(s: np.ndarray, h: np.ndarray, J: np.ndarray):
    return np.dot(np.dot(s, J), s) + np.dot(s, h)


def read_3_body_instance(path: str):
    with open(path, "r") as f:
        file = json.load(f)
    J = {}
    h = {}
    model_temp = file["model"]
    solution = file["solution"]

    for k, v in model_temp.items():
        (k1, k2) = eval(k)
        if k1 == k2:
            h[k1] = v
        else:
            J[(k1, k2)] = v
    return h, J, solution


def find_embedding(source: nx.Graph, target: nx.Graph):
    embedding = minorminer.find_embedding(source, target, chainlength_patience=10)
    contains_chains = any(len(chain) > 1 for chain in embedding.values())
    while contains_chains:
        embedding = minorminer.find_embedding(source, target, chainlength_patience=100)
        contains_chains = any(len(chain) > 1 for chain in embedding.values())

    for value in embedding.values():
        assert len(value) == 1
    return embedding


def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                try:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '└── ' + key + ' (scalar)')
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                try:
                    print(pre + '├── ' + key + ' (%d)' % len(val))
                except TypeError:
                    print(pre + '├── ' + key + ' (scalar)')