import os
import pickle
import dwave.inspector

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from src.utils import (pseudo_likelihood, gibbs_sampling_ising, energy, vectorize, create_and_save_instance,
                       create_planted_solution_instance, Instance)
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer import find_embedding
from copy import deepcopy
from tqdm import tqdm

rng = np.random.default_rng()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()
output_path = os.path.join(ROOT, "data", "raw_data", "quadrants2")
if not os.path.exists(output_path):
    os.makedirs(output_path)

try:
    from src.config import TOKEN
except ImportError:
    print(f"To run {__file__}, you must have \"config.py\" file with your dwave's ocean token")
    with open(os.path.join(CWD, "config.py"), "w") as f:
        f.write("TOKEN = \"your_ocean_token\"")


def test_embedding(chain: nx.Graph, target: nx.Graph):
    embedding = find_embedding(chain, target)
    sampler = FixedEmbeddingComposite(qpu_sampler, embedding)
    sampleset = sampler.sample_ising(h, J, num_reads=100, annealing_time=10)
    dwave.inspector.show(sampleset)


if __name__ == '__main__':

    # Setup
    qpu_sampler = DWaveSampler(solver='Advantage_system6.3', token=TOKEN)
    target = qpu_sampler.to_networkx_graph()
    pegasus_nice_numbering = {node: dnx.pegasus_coordinates(16).linear_to_nice(node) for node in target.nodes}

    # Experiment setup
    CHAIN_LENGTH = 300
    NUM_SAMPLES = 100
    GIBBS_NUM_STEPS = 10 ** 4
    ANNEAL_LENGTH = 200
    NUM_READS = 100
    PAUSES = [0, 20, 60, 100]


    # Right now generates random instances.
    # instance_path = os.path.join(ROOT, "data", "instance.pkl")
    # if os.path.exists(instance_path):
    #     with open(instance_path, "rb") as f:
    #         print(f"loading existing instance")
    #         h, J = pickle.load(f)
    # else:
    #     print(f"generating new instance")
    #     h = {node: 0 for node in range(CHAIN_LENGTH)}
    #     J = {(node, node + 1): rng.uniform(-1, 1) for node in range(CHAIN_LENGTH - 1)}
    #     with open(instance_path, "wb") as f:
    #         l = [h, J]
    #         pickle.dump(l, f)
    #
    # h_vect, J_vect = vectorize(h, J)
    # chain = nx.Graph(J.keys())

    # First Quadrant
    Q1 = deepcopy(target)
    Q1.graph["name"] = "Q1"
    for node in target.nodes:
        t, y, x, u, k = pegasus_nice_numbering[node]
        if y > 6 or x > 6:
            Q1.remove_node(node)

    # Second Quadrant
    Q2 = deepcopy(target)
    Q2.graph["name"] = "Q2"
    for node in target.nodes:
        t, y, x, u, k = pegasus_nice_numbering[node]
        if y > 6 or x < 8:
            Q2.remove_node(node)

    # Third Quadrant
    Q3 = deepcopy(target)
    Q3.graph["name"] = "Q3"
    for node in target.nodes:
        t, y, x, u, k = pegasus_nice_numbering[node]
        if y < 8 or x > 6:
            Q3.remove_node(node)

    # Fourth Quadrant
    Q4 = deepcopy(target)
    Q4.graph["name"] = "Q4"
    for node in target.nodes:
        t, y, x, u, k = pegasus_nice_numbering[node]
        if y < 8 or x < 8:
            Q4.remove_node(node)


    # what instance do you wand
    # h, J = create_planted_solution_instance()
    # save_path = ""
    # with open(save_path, "wb") as f:
    #   inst = Instance(h=h, J=J, name="planted"solution)
    #   pickle.dump(inst, f)
    #
    instance_name = ""
    create_and_save_instance(Q1, "constant", instance_name)

    with open(os.path.join(ROOT, "data", instance_name), "rb") as f:
        inst = pickle.load(f)

    h = inst.h
    J = inst.J

    h_vect, J_vect = vectorize(h, J)

    for quadrant in [Q1, Q2, Q3, Q4]:
        name = quadrant.graph["name"]
        for pause_duration in PAUSES:
            for anneal_param in np.linspace(0, 1, num=25):

                E_fin = []
                configurations = []
                Q = []
                raw_data = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
                for i in tqdm(range(NUM_SAMPLES),
                              desc=f"samples for {name} pause duration {pause_duration:.2f} s {anneal_param:.2f}"):
                    initial_state = dict(gibbs_sampling_ising(h, J, 1, GIBBS_NUM_STEPS))
                    init_state = np.array(list(initial_state.values()))

                    E_init = energy(init_state, h_vect, J_vect) / CHAIN_LENGTH  # per spin

                    anneal_schedule = [[0, 1], [ANNEAL_LENGTH * 1 / 2 - pause_duration / 2, anneal_param],
                                       [ANNEAL_LENGTH * 1 / 2 + pause_duration / 2, anneal_param],
                                       [ANNEAL_LENGTH, 1]] if pause_duration != 0 else \
                                       [[0, 1], [ANNEAL_LENGTH / 2, anneal_param], [ANNEAL_LENGTH, 1]]
                    # try:
                    #     embedding = find_embedding(chain, quadrant, tries=1000)
                    # except ValueError:
                    #     embedding = find_embedding(chain, quadrant, tries=100000)
                    # sampler = FixedEmbeddingComposite(qpu_sampler, embedding)

                    sampleset = qpu_sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                     anneal_schedule=anneal_schedule,
                                                     num_reads=NUM_READS, auto_scale=False, reinitialize_state=True)

                    df = sampleset.to_pandas_dataframe(sample_column=True)
                    df["init_state"] = [initial_state for _ in range(len(df))]
                    raw_data = pd.concat([raw_data, df], ignore_index=True)

                    raw_data.to_csv(os.path.join(output_path,
                                                 f"raw_data_pegasus_{name}_{pause_duration}_{anneal_param:.2f}.csv"))


