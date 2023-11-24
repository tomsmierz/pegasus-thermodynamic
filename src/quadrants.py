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

    # Setup
    qpu_sampler = DWaveSampler(solver='Advantage_system6.3', token=TOKEN)
    target = qpu_sampler.to_networkx_graph()

    # First Quadrant
    Q1 = deepcopy(target)
    Q1.graph["name"] = "Q1"
    for node in target.nodes:
        t, y, x, u, k = node
        if y > 6 or x > 6:
            Q1.remove_node(node)

    # Second Quadrant
    Q2 = deepcopy(target)
    Q2.graph["name"] = "Q2"
    for node in target.nodes:
        t, y, x, u, k = node
        if y > 6 or x < 8:
            Q2.remove_node(node)

    # Third Quadrant
    Q3 = deepcopy(target)
    Q3.graph["name"] = "Q3"
    for node in target.nodes:
        t, y, x, u, k = node
        if y < 8 or x > 6:
            Q3.remove_node(node)

    # Fourth Quadrant
    Q4 = deepcopy(target)
    Q4.graph["name"] = "Q4"
    for node in target.nodes:
        t, y, x, u, k = node
        if y < 8 or x < 8:
            Q4.remove_node(node)

    # Experiment setup
    chain_length = 300
    num_samples = 100
    gibbs_num_steps = 10 ** 4
    with open(os.path.join(ROOT, "data", "instance.pkl")) as f:
        h, J = pickle.load(f)

    mean_E = {}
    var_E = {}
    mean_Q = {}
    var_Q = {}
    beta_eff = {}

    for quadrant in [Q1, Q2, Q3, Q4]:
        for pause_duration in [0, 20, 40, 60, 80, 100]:
            for anneal_param in np.linspace(0, 1, num=25):
                E_fin = []
                configurations = []
                Q = []
                raw_data = pd.DataFrame(columns=["sample", "energy", "num_occurrences", "init_state"])
                for i in tqdm(range(num_samples),
                              desc=f"samples for pause duration {pause_duration:.2f} s {anneal_param:.2f}"):
                    initial_state = dict(gibbs_sampling_ising(h, J, 1, gibbs_num_steps))
                    init_state = np.array(list(initial_state.values()))

                    E_init = energy(init_state, h_vect, J_vect) / chain_length  # per spin

                    anneal_schedule = [[0, 1], [ANNEAL_LENGTH * 1 / 2 - pause_duration / 2, anneal_param],
                                       [ANNEAL_LENGTH * 1 / 2 + pause_duration / 2, anneal_param],
                                       [ANNEAL_LENGTH, 1]] if pause_duration != 0 else [[0, 1],
                                                                                        [ANNEAL_LENGTH / 2,
                                                                                         anneal_param],
                                                                                        [ANNEAL_LENGTH, 1]]

                    sampleset = sampler.sample_ising(h=h, J=J, initial_state=initial_state,
                                                     anneal_schedule=anneal_schedule,
                                                     num_reads=10, auto_scale=False, reinitialize_state=True)

                    for s in sampleset.samples():
                        final_state = np.array(list(s.values()))
                        E_fin.append(energy(final_state, h_vect, J_vect) / chain_length)
                        configurations.append(final_state)
                        Q.append(energy(final_state, h_vect, J_vect) / chain_length - E_init)

                    df = sampleset.to_pandas_dataframe(sample_column=True)
                    df["init_state"] = [initial_state for _ in range(len(df))]
                    raw_data = pd.concat([raw_data, df], ignore_index=True)
                    raw_data.to_csv(os.path.join(cwd, "..\\results\\raw_data\\pegasus_thermo",
                                                 f"raw_data_pegasus_thermo_{pause_duration}_{anneal_param:.2f}.csv"),
                                    sep=";")

                optim = optimize.minimize(pseudo_likelihood, 1.0, args=(np.array(configurations),))
                beta_eff[(pause_duration, anneal_param)] = optim.x
                mean_E[(pause_duration, anneal_param)] = np.mean(np.array(E_fin))
                var_E[(pause_duration, anneal_param)] = np.var(np.array(E_fin))
                mean_Q[(pause_duration, anneal_param)] = np.mean(np.array(Q))
                var_Q[(pause_duration, anneal_param)] = np.var(np.array(Q))

                with open(os.path.join(cwd, "..\\results\\checkpoints", f"checkpoints_pegasus_thermo.pkl"), "wb") as f:
                    pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)

            with open(os.path.join(cwd, "..", "results", f"results_pegasus_thermo.pkl"), "wb") as f:
                pickle.dump([beta_eff, mean_E, var_E, mean_Q, var_Q], f)
