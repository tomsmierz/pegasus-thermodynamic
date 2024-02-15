import os
import pickle

import numpy as np
import pandas as pd

from scipy import optimize
from src.utils import pseudo_likelihood, extend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()
DATA = os.path.join(ROOT, "data", "raw_data", "thermodynamics")
RESULTS = os.path.join(ROOT, "data", "results", "thermodynamics")

betas = {}
delta_E = {}
Q = {}

with open(os.path.join(ROOT, "data", "instance.pkl"), "rb") as f:
    h, J = pickle.load(f)
    J = extend(J)


def calculate_betas():
    for filename in os.listdir(DATA):

        file_path = os.path.join(DATA, filename)
        if os.path.isfile(file_path):

            df = pd.read_csv(file_path, sep=";", index_col=0)

            configurations = []
            for row in df.itertuples():
                state = eval(row.sample)
                state = list(state.values())
                configurations.append(state)
            print(f"optimizing {filename}")
            optim = optimize.minimize(pseudo_likelihood, np.array([1.0]), args=(h, J, np.array(configurations)))
            with open(os.path.join(RESULTS, "betas2.pkl"), "wb") as f:
                name = filename[0:-4]
                parameters = name.split("_")
                pause_time = parameters[4]
                anneal_param = parameters[5]
                betas[(pause_time, anneal_param)] = optim.x.item()
                pickle.dump(betas, f)
            print("result: beta = ", optim.x.item())
