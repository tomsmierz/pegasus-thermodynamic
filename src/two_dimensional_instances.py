import os
import pickle

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from src.utils import pseudo_likelihood, gibbs_sampling_ising, energy, vectorize
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer import find_embedding
from copy import deepcopy
from tqdm import tqdm
from collections import namedtuple

rng = np.random.default_rng()
Instance = namedtuple("Instance", ["h", "J", "name"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()

try:
    from src.config import TOKEN
except ImportError:
    print(f"To run {__file__}, you must have \"config.py\" file with your dwave's ocean token")
    with open(os.path.join(CWD, "config.py"), "w") as f:
        f.write("TOKEN = \"your_ocean_token\"")


if __name__ == '__main__':

    # Setup
    qpu_sampler = DWaveSampler(solver='Advantage_system4.1', token=TOKEN)

    # Experiment setup
    num_samples = 100
    gibbs_num_steps = 10 ** 4
    anneal_length = 2000
    num_reads = 100

    graph = qpu_sampler.to_networkx_graph()

    h_const = {node: 0 for node in graph.nodes}
    J_const = {edge: -1 for edge in graph.edges}
    const = Instance(h=h_const, J=J_const, name="constant")
    with open(os.path.join(ROOT, "data", "p16_4.1_const.pkl"), "wb") as f:
        pickle.dump(const, f)

    h_uniform = {node: 0 for node in graph.nodes}
    J_uniform = {edge: rng.uniform(-1, 1) for edge in graph.edges}
    uniform = Instance(h=h_uniform, J=J_uniform, name="Uniform")
    with open(os.path.join(ROOT, "data", "p16_4.1_uniform.pkl"), "wb") as f:
        pickle.dump(uniform, f)

    h_cbfm = {node: rng.choice([-1, 0], p=[0.85, 0.15]) for node in graph.nodes}
    J_cbfm = {edge: rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55]) for edge in graph.edges}
    cbfm = Instance(h=h_cbfm, J=J_cbfm, name="CBFM")
    with open(os.path.join(ROOT, "data", "p16_4.1_cbfm.pkl"), "wb") as f:
        pickle.dump(cbfm, f)

    # with open(os.path.join(ROOT, "data", "p16_4.1_const.pkl"), "rb") as f:
    #     const = pickle.load(f)
    #
    # with open(os.path.join(ROOT, "data", "p16_4.1_uniform.pkl"), "rb") as f:
    #     uniform = pickle.load(f)
    #
    # with open(os.path.join(ROOT, "data", "p16_4.1_cbfm.pkl"), "rb") as f:
    #     cbfm = pickle.load(f)

    for inst in [const, uniform, cbfm]:
        h = inst.h
        J = inst.J
        name = inst.name
        for pause_duration in [0, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1500, 1900, 1990, 1998]:
            for anneal_param in np.linspace(0.2, 0.4, num=11):

                raw_data = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
                for i in tqdm(range(num_samples),
                              desc=f"samples for instance {name} pause duration "
                                   f"{pause_duration:.2f} s {anneal_param:.2f}"):
                    initial_state = dict(gibbs_sampling_ising(h, J, 1, gibbs_num_steps))
                    init_state = np.array(list(initial_state.values())) # per spin

                    anneal_schedule = [[0, 1], [anneal_length * 1 / 2 - pause_duration / 2, anneal_param],
                                       [anneal_length * 1 / 2 + pause_duration / 2, anneal_param],
                                       [anneal_length, 1]] if pause_duration != 0 else \
                                       [[0, 1], [anneal_length / 2, anneal_param], [anneal_length, 1]]

                    sampleset = qpu_sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                         anneal_schedule=anneal_schedule, num_reads=num_reads,
                                                         auto_scale=False, reinitialize_state=True)


                    df = sampleset.to_pandas_dataframe(sample_column=True)
                    df["init_state"] = [initial_state for _ in range(len(df))]
                    raw_data = pd.concat([raw_data, df], ignore_index=True)
                    raw_data.to_csv(os.path.join(ROOT, "data", "raw_data", "2d"
                                                 f"raw_data_2d_{name}_{pause_duration}_{anneal_param:.2f}.csv"))
