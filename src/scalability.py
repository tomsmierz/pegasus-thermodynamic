import os
import pickle
import dwave.inspector

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from src.utils import pseudo_likelihood, gibbs_sampling_ising, energy, vectorize
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
from minorminer import find_embedding
from copy import deepcopy
from tqdm import tqdm

rng = np.random.default_rng()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()

try:
    from src.config import TOKEN
except ImportError:
    print(f"To run {__file__}, you must have \"config.py\" file with your dwave's ocean token")
    with open(os.path.join(CWD, "config.py"), "w") as f:
        f.write("TOKEN = \"your_ocean_token\"")



if __name__ == '__main__':

    qpu_sampler = DWaveSampler(solver='Advantage_system6.3')
    target = qpu_sampler.to_networkx_graph()
    num_samples = 100
    gibbs_num_steps = 10 ** 4
    anneal_length = 200
    num_reads = 100
    anneal_param = 0.3

    mean_E = {}
    var_E = {}
    mean_Q = {}
    var_Q = {}
    beta_eff = {}

    for chain_length in [500, 1000, 2000, 3000, 4000, 5000]:
        if chain_length in [500, 1000]:
            print(f"loading instance of lengh {chain_length}")
            with open(os.path.join(ROOT, "data", f"instance_{chain_length}.pkl"), "rb") as f:
                h, J = pickle.load(f)
        else:
            print(f"generating new instance of lenght {chain_length}")
            h = {node: 0 for node in range(chain_length)}
            J = {(node, node + 1): rng.uniform(-1, 1) for node in range(chain_length-1)}
            with open(os.path.join(ROOT, "data", f"instance_{chain_length}.pkl"), "wb") as f:
                l = [h, J]
                pickle.dump(l, f)

        h_vect, J_vect = vectorize(h, J)
        chain = nx.Graph(J.keys())

        for pause_duration in [20, 60, 100]:

            if chain_length == 500 and pause_duration == 20:
                continue
            E_fin = []
            configurations = []
            Q = []
            raw_data = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
            for i in tqdm(range(num_samples),
                          desc=f"samples for {chain_length} pause duration {pause_duration:.2f} s"):
                initial_state = dict(gibbs_sampling_ising(h, J, 1, gibbs_num_steps))
                init_state = np.array(list(initial_state.values()))

                E_init = energy(init_state, h_vect, J_vect) / chain_length  # per spin

                anneal_schedule = [[0, 1], [anneal_length * 1 / 2 - pause_duration / 2, anneal_param],
                                   [anneal_length * 1 / 2 + pause_duration / 2, anneal_param],
                                   [anneal_length, 1]] if pause_duration != 0 else \
                    [[0, 1], [anneal_length / 2, anneal_param], [anneal_length, 1]]

                sampler = EmbeddingComposite(qpu_sampler)

                sampleset = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                 anneal_schedule=anneal_schedule,
                                                 num_reads=num_reads, auto_scale=False, reinitialize_state=True)

                # for s in sampleset.samples():
                #     final_state = np.array(list(s.values()))
                #     E_fin.append(energy(final_state, h_vect, J_vect) / chain_length)
                #     configurations.append(final_state)
                #     Q.append(energy(final_state, h_vect, J_vect) / chain_length - E_init)

                df = sampleset.to_pandas_dataframe(sample_column=True)
                df["init_state"] = [initial_state for _ in range(len(df))]
                raw_data = pd.concat([raw_data, df], ignore_index=True)
                raw_data.to_csv(os.path.join(ROOT, "data", "raw_data", "scalability",
                                             f"raw_data_pegasus_{chain_length}_{pause_duration}.csv"))