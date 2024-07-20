import os
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from dimod import BinaryQuadraticModel
from collections import OrderedDict, namedtuple


rng = np.random.default_rng()
Instance = namedtuple("Instance", ["h", "J", "name"])
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    h = dict(sorted(h.items()))
    J = dict(sorted(J.items()))
    n = max(list(h.keys()))
    h_list = [h[i] if i in h.keys() else 0 for i in range(n+1)]
    h_vect = np.array(h_list)
    n = len(h_vect)
    J_vect = np.zeros((n, n))
    for key, value in J.items():
        J_vect[key[0]][key[1]] = value
    return h_vect, J_vect


def energy(s: np.ndarray, h: np.ndarray, J: np.ndarray):
    if len(s) != len(h):
        s2 = [s[i] if h[i] != 0 else 0 for i in range(len(h))]
        s2 = np.array(s2)
        return np.dot(np.dot(s2, J), s2) + np.dot(s2, h)
    else:
        return np.dot(np.dot(s, J), s) + np.dot(s, h)


def random_walk(G: nx.Graph):

    start_node = rng.choice(list(G.nodes))
    visited_nodes = set()
    current_node = start_node

    walk_path = [current_node]
    visited_nodes.add(current_node)

    while True:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break

        next_node = rng.choice(neighbors)
        if next_node in visited_nodes:
            walk_path.append(next_node)
            if next_node != start_node:
                index = walk_path.index(next_node)
                walk_path = walk_path[index:]
            break
        else:
            walk_path.append(next_node)
            visited_nodes.add(next_node)
            current_node = next_node

    return walk_path


def create_planted_solution_instance(min_loop_size, num_loops, instance_graph):
    """
    right now, the ground states it produces are without frustrations, so there is Z2 symetry in the
    low energy spectrum. In general, the ground states produced by this method are not unique.
    """
    loops = []
    print("generating random walks")
    while len(loops) < num_loops:
        loop = random_walk(instance_graph)
        if len(loop) >= min_loop_size+1:
            loops.append(loop)
    unique_nodes = set(sum(loops, []))
    assert len(unique_nodes) <= len(list(instance_graph.nodes))
    planted_solution = {node: rng.choice([-1, 1]) for node in unique_nodes}
    J = {}
    for loop in loops:
        J_loop = {}
        for i in range(len(loop) - 1):
            J_loop[(loop[i], loop[i+1])] = -1 * planted_solution[loop[i]] * planted_solution[loop[i+1]]
        # chosen_edge = tuple(rng.choice(list(J_loop.keys())))
        # J_loop[chosen_edge] *= -1
        for key, value in J_loop.items():
            if key not in J.keys():
                J[key] = value
    energy = sum([value * planted_solution[n1] * planted_solution[n2] for (n1, n2), value in J.items()])
    h = {node: 0 for node in unique_nodes}
    return h, J, energy, planted_solution


def create_and_save_instance(graph: nx.Graph, instance_type: str, name: str, save_path=os.path.join(ROOT, "data")):
    if instance_type not in ["uniform", "constant", "CBFM"]:
        raise ValueError("instance_type shoud be \"uniform\", \"constant\" or \"CBFM\"")

    if instance_type == "constant":
        h = {node: 0 for node in graph.nodes}
        J = {edge: -1 for edge in graph.edges}
    elif instance_type == "uniform":
        h = {node: 0 for node in graph.nodes}
        J = {edge: rng.uniform(-1, 1) for edge in graph.edges}
    elif instance_type == "CBFM":
        h = {node: rng.choice([-1, 0], p=[0.85, 0.15]) for node in graph.nodes}
        J = {edge: rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55]) for edge in graph.edges}
    else:
        raise ValueError("Something is really wrong")

    inst = Instance(h=h, J=J, name=instance_type)
    with open(os.path.join(save_path, name), "rb") as f:
        pickle.dump(inst, f)