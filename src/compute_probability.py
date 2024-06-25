import os
import pickle
import sys

import numpy as np
import pandas as pd

from dwave.samplers import PlanarGraphSolver
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "raw_data", "scalability")
RESULTS = os.path.join(ROOT, "data", "results", "scalability")

SCALING = [100, 500, 1000, 2000]
PAUSES = [0, 20, 60, 100]
CALCULATE_GROUND_STATES = False


solver = PlanarGraphSolver()  # exact solver, so it always reaches ground state

# be careful with that, it may lead to stackoverflow, depending on machine,
# should be >= num spins
sys.setrecursionlimit(1500)


if __name__ == '__main__':

    energies_data = {}
    energies = []
    sizes = []
    types = []
    if CALCULATE_GROUND_STATES:
        # scaling
        for chain_length in SCALING:
            filepath = os.path.join(ROOT, "data", f"instance_{chain_length}.pkl")
            if os.path.exists(filepath):
                print(f"loading instance of length {chain_length}")
                with open(filepath, "rb") as f:
                    _, J = pickle.load(f)
            h = {}
            if chain_length < 500:  # this line is only for testing, can be removed
                sampleset = solver.sample_ising(h, J)
                sample = sampleset.to_pandas_dataframe()
                energy = sample["energy"].iloc[0]
                energies.append(energy)
                sizes.append(chain_length)
                types.append("SCALING")

        # quadrants
        with open(os.path.join(ROOT, "data", f"instance.pkl"), "rb") as f:
            _, J = pickle.load(f)
        h = {}
        sampleset = solver.sample_ising(h, J)
        sample = sampleset.to_pandas_dataframe()
        energy = sample["energy"].iloc[0]
        energies.append(energy)
        sizes.append(300)
        types.append("QUADRANTS")
        energies_data["size"] = sizes
        energies_data["energy"] = energies
        energies_data["experiment_type"] = types

        energies_df = pd.DataFrame(data=energies_data)
        energies_df.to_csv(os.path.join(ROOT, "data", "ground_state_energies.csv"))

    ground_states = pd.read_csv(os.path.join(ROOT, "data", "ground_state_energies.csv"), index_col=0)
    probabilities = {}
    for filename in os.listdir(DATA):
        file_path = os.path.join(DATA, filename)
        if os.path.isfile(file_path):
            name = filename[0:-4]
            parameters = name.split("_")
            chain_length = parameters[3]
            pause_time = parameters[4]

            df = pd.read_csv(file_path, index_col=0)

            p = 0
            c = 0
            ground_state_energy = ground_states.loc[ground_states["size"] == chain_length, "energy"].values[0]
            for init_state in tqdm(df.init_state.unique().tolist(), desc=f"calculate probability for {filename}"):
                temp_df = df[df["init_state"] == init_state]
                c += 1
                if ground_state_energy in temp_df["energy"].values:
                    p += 1
            prob = p/c
            probabilities[pause_time] = prob
            with open(os.path.join(RESULTS, f"comp_eff_{chain_length}.pkl"), "wb") as f:
                pickle.dump(probabilities, f)