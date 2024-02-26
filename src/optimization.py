import os
import pickle

import numpy as np
import pandas as pd

from scipy import optimize
from src.utils import pseudo_likelihood, extend, vectorize, energy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()
DATA = os.path.join(ROOT, "data", "raw_data", "scalability")
RESULTS = os.path.join(ROOT, "data", "results", "scalability")

betas = {}
energies = {}
Q = {}

# with open(os.path.join(ROOT, "data", "instance.pkl"), "rb") as f:
#     h, J = pickle.load(f)
#     h_vect, J_vect = vectorize(h, J)
#     J = extend(J)
#     chain_length = len(h)


def main():
    for filename in os.listdir(DATA):
        file_path = os.path.join(DATA, filename)
        if os.path.isfile(file_path):
            print(f"optimizing {filename}")
            name = filename[0:-4]
            parameters = name.split("_")
            chain_length = parameters[3]
            pause_time = parameters[4]
            if int(chain_length) != 500:
                continue

            with open(os.path.join(ROOT, "data", f"instance_{chain_length}.pkl"), "rb") as f:
                h, J = pickle.load(f)
                h_vect, J_vect = vectorize(h, J)
                J = extend(J)

            df = pd.read_csv(file_path, index_col=0)

            configurations = []
            E_final = []
            Q_vect = []
            for row in df.itertuples():
                state = eval(row.sample)
                state = list(state.values())
                configurations.append(state)
                energy_final = row.energy
                E_final.append(energy_final / int(chain_length))  # per spin
                init_state = eval(row.init_state)
                init_state = np.array(list(init_state.values()))
                energy_init = energy(init_state, h_vect, J_vect)
                Q_vect.append((energy_final - energy_init) / int(chain_length))  # per spin

            optim = optimize.minimize(pseudo_likelihood, np.array([1.0]), args=(h, J, np.array(configurations)))
            with open(os.path.join(RESULTS, "betas2_500.pkl"), "wb") as f:
                betas[(chain_length, pause_time)] = optim.x.item()
                pickle.dump(betas, f)
            print("result: beta = ", optim.x.item())
            with open(os.path.join(RESULTS, "energies_500.pkl"), "wb") as f2:
                E_mean, E_var = np.mean(np.array(E_final)), np.var(np.array(E_final))
                energies[(chain_length, pause_time)] = (E_mean, E_var)
                pickle.dump(energies, f2)
            print("result: energies = ", (E_mean, E_var))
            with open(os.path.join(RESULTS, "Q_500.pkl"), "wb") as f3:
                Q_mean, Q_var = np.mean(np.array(Q_vect)), np.var(np.array(Q_vect))
                Q[(chain_length, pause_time)] = (Q_mean, Q_var)
                pickle.dump(Q, f3)
            print("result: Q = ", (Q_mean, Q_var))


if __name__ == '__main__':
    main()
