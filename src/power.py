import os
import pickle
import dwave.inspector

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from utils import pseudo_likelihood, gibbs_sampling_ising, energy, vectorize
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, EmbeddingComposite
from minorminer import find_embedding
from copy import deepcopy
from tqdm import tqdm

rng = np.random.default_rng()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()
output_path = os.path.join(ROOT, "data", "raw_data", "power")
if not os.path.exists(output_path):
    os.makedirs(output_path)


# try:
#     from config import TOKEN
# except ImportError:
#     print(f"To run {__file__}, you must have \"config.py\" file with your dwave's ocean token")
#     with open(os.path.join(CWD, "config.py"), "w") as f:
#         f.write("TOKEN = \"your_ocean_token\"")


if __name__ == '__main__':

    # This part may need changes depending on how you comunicate with your machine
    qpu_sampler = DWaveSampler(solver='Advantage_system5.4',token='julr-a86ece088ec3ae431ae7ee0541c03112c43d7af4',region="eu-central-1")  # specify device used
    target = qpu_sampler.to_networkx_graph()

    BETA_1 = 1
    NUM_SAMPLES = 1000
    GIBBS_NUM_STEPS = 10 ** 4
    anneal_time = [50, 100, 200, 500, 1000, 2000] #crashed for 500 at 89 samples
    NUM_READS = 10
    initial_value = 1.0
    ANNEAL_PARAM = 2.96227 / 8.586335
    SCALING = [300] #crashed for 1000 at 382 samples
    pause_duration = 0

    for chain_length in SCALING:
        filepath = os.path.join(ROOT, "data", f"instance_{chain_length}.pkl")
        if os.path.exists(filepath):
            print(f"loading instance of length {chain_length}")
            with open(filepath, "rb") as f:
                h, J = pickle.load(f)
        else:
            print(f"generating new instance of length {chain_length}")
            h = {node: 0 for node in range(chain_length)}
            J = {(node, node + 1): rng.uniform(-1, 1) for node in range(chain_length-1)}
            with open(filepath, "wb") as f:
                l = [h, J]
                pickle.dump(l, f)

        h_vect, J_vect = vectorize(h, J)
        chain = nx.Graph(J.keys())

        #for pause_duration in PAUSES:
        for ANNEAL_TIME in anneal_time:

            E_fin = []
            configurations = []
            Q = []
            raw_data = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
            for i in tqdm(range(NUM_SAMPLES),
                          desc=f"samples for {chain_length} anneal time {ANNEAL_TIME:.2f} s"):
                initial_state = dict(gibbs_sampling_ising(h, J, BETA_1, GIBBS_NUM_STEPS))
                init_state = np.array(list(initial_state.values()))

                E_init = energy(init_state, h_vect, J_vect) / chain_length  # per spin
                """
                anneal_schedule = [[0, 1], [ANNEAL_TIME * 1 / 2 - pause_duration / 2, ANNEAL_PARAM],
                                   [ANNEAL_TIME * 1 / 2 + pause_duration / 2, ANNEAL_PARAM],
                                   [ANNEAL_TIME, 1]] if pause_duration != 0 else \
                    [[0, 1], [ANNEAL_TIME / 2, ANNEAL_PARAM], [ANNEAL_TIME, 1]]
                """
                anneal_schedule = [[0.0, initial_value], [ANNEAL_TIME * 1 / 2 - pause_duration / 2, ANNEAL_PARAM],
                                   [ANNEAL_TIME * 1 / 2 + pause_duration / 2, ANNEAL_PARAM],
                                   [ANNEAL_TIME, initial_value]] if pause_duration != 0 else \
                    [[0.0, initial_value], [ANNEAL_TIME / 2, ANNEAL_PARAM], [ANNEAL_TIME, initial_value]]

                sampler = EmbeddingComposite(qpu_sampler)

                sampleset = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                 anneal_schedule=anneal_schedule,
                                                 num_reads=NUM_READS, auto_scale=False, reinitialize_state=True)

                df = sampleset.to_pandas_dataframe(sample_column=True)
                df["init_state"] = [initial_state for _ in range(len(df))]
                raw_data = pd.concat([raw_data, df], ignore_index=True)

                raw_data.to_csv(os.path.join(output_path, f"raw_data_pegasus_{chain_length}_{ANNEAL_TIME}.csv"))
