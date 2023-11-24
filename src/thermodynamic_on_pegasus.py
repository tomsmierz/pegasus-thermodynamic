import os

from src.utils import pseudo_likelihood, gibbs_sampling_ising, energy, vectorize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()

try:
    from src.config import TOKEN
except ImportError:
    print(f"To run {__file__}, you must have \"config.py\" file with your dwave's ocean token")
    with open(os.path.join(CWD, "config.py"), "w") as f:
        f.write("TOKEN = \"your_ocean_token\"")








