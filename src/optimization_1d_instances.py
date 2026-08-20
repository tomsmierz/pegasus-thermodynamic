import argparse
import os
import pickle
import re

import numpy as np
from scipy import optimize

from raw_data_io import load_raw_data_file
from utils import extend, pseudo_likelihood, vectorize

PHYSICAL_UNITS = False

B0 = 1.0  # actually B0 = 8.58.. but this is included into annealing schedule
h = 6.62607015e-34  # Planck constant, in J/Hz
kb = 1.380649e-23  # Boltzmann constant, in J/K
energy_units = (B0 / 2) * 10**9 * h
beta_units = kb / energy_units
if PHYSICAL_UNITS is False:
    energy_units = 1
    beta_units = 1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILENAME_RE = re.compile(
    r"raw_data_chain_L_(\d+)_at_([\d.]+)_ap_([\d.]+)_beta_([\d.]+)\.(?:csv|npz)$"
)


def _parse_float_list(raw: str | None) -> set[float] | None:
    if raw is None:
        return None
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        return None
    return {float(x) for x in values}


def _matches_filters(
    anneal_time: str,
    anneal_param: str,
    beta: str,
    anneal_time_filter: str | None,
    ap_filter: set[float] | None,
    beta_filter: set[float] | None,
) -> bool:
    if anneal_time_filter is not None and float(anneal_time) != float(anneal_time_filter):
        return False
    if ap_filter is not None and float(anneal_param) not in ap_filter:
        return False
    if beta_filter is not None and float(beta) not in beta_filter:
        return False
    return True


def parse_args() -> argparse.Namespace:
    default_hname = "unif_small_h"
    default_results = os.path.join(
        ROOT,
        "data",
        "results",
        f"phase_diagrams_1d_pegasus_{default_hname}",
        # f"scaling_1d_pegasus_{default_hname}",
    )
    parser = argparse.ArgumentParser(
        description="Optimize 1D chain presampled raw data with optional parameter filters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chain-lengths", default="300", help="Comma-separated chain lengths")
    parser.add_argument("--hname", default=default_hname, help="Instance field-name suffix")
    parser.add_argument(
        "--anneal-time",
        default=None,
        help="Optional anneal_time filter (for example: 10)",
    )
    parser.add_argument(
        "--ap-list",
        default=None,
        help="Optional comma-separated anneal_param values (for example: 0.200,0.500,0.800)",
    )
    parser.add_argument(
        "--beta-list",
        default=None,
        help="Optional comma-separated beta values; defaults to all discovered values",
    )
    parser.add_argument(
        "--results-dir",
        default=default_results,
        help="Directory to write optimization result pickle files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and resave selected optimization results even when they already exist.",
    )
    return parser.parse_args()


def _load_pickle_if_exists(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    args = parse_args()
    chain_lengths = [int(x.strip()) for x in args.chain_lengths.split(",") if x.strip()]
    ap_filter = _parse_float_list(args.ap_list)
    beta_filter = _parse_float_list(args.beta_list)

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    total_files = 0
    matched_name = 0
    selected_by_filter = 0
    processed = 0
    skipped_bad_name = 0
    skipped_empty = 0
    skipped_instance = 0
    skipped_existing = 0

    for L in chain_lengths:
        chain_length = str(L)
        betas_path = os.path.join(results_dir, f"betas2_{chain_length}.pkl")
        energies_path = os.path.join(results_dir, f"energies_{chain_length}.pkl")
        q_path = os.path.join(results_dir, f"Q_{chain_length}.pkl")
        q_dist_path = os.path.join(results_dir, f"Q_dist_{chain_length}.pkl")

        betas: dict[tuple[str, str, str, str], float] = _load_pickle_if_exists(betas_path)
        energies: dict[tuple[str, str, str, str], tuple[float, float]] = _load_pickle_if_exists(
            energies_path
        )
        Q: dict[tuple[str, str, str, str], tuple[float, float]] = _load_pickle_if_exists(q_path)
        Q_dist: dict[tuple[str, str, str, str], np.ndarray] = _load_pickle_if_exists(q_dist_path)

        data_dir = os.path.join(
            ROOT, "data", "raw_data", f"phase_diagram_pegasus_1D_chain_L_{L}_{args.hname}"
        )
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue

        for filename in sorted(os.listdir(data_dir)):
            file_path = os.path.join(data_dir, filename)
            if not os.path.isfile(file_path) or not (
                filename.endswith(".csv") or filename.endswith(".npz")
            ):
                continue
            total_files += 1
            print(f"optimizing {filename}")

            m = FILENAME_RE.match(filename)
            if not m:
                skipped_bad_name += 1
                print(f"Skipping unexpected filename: {filename}")
                continue
            matched_name += 1
            chain_length = m.group(1)
            anneal_time = m.group(2)
            anneal_param = m.group(3)
            beta = m.group(4)

            if not _matches_filters(
                anneal_time=anneal_time,
                anneal_param=anneal_param,
                beta=beta,
                anneal_time_filter=args.anneal_time,
                ap_filter=ap_filter,
                beta_filter=beta_filter,
            ):
                continue
            selected_by_filter += 1
            key = (chain_length, anneal_time, anneal_param, beta)
            if (
                not args.overwrite
                and key in betas
                and key in energies
                and key in Q
                and key in Q_dist
            ):
                skipped_existing += 1
                print(f"Skipping existing result for {filename}")
                continue

            instance_path = os.path.join(
                ROOT,
                "data",
                "instances",
                "1d_pegasus_chains",
                f"pegasus_1D_chain_L_{chain_length}_{args.hname}.pkl",
            )
            if not os.path.exists(instance_path):
                skipped_instance += 1
                print(f"Instance not found: {instance_path}")
                continue

            with open(instance_path, "rb") as f:
                h_ising, J_ising = pickle.load(f)
                h_vect, J_vect = vectorize(h_ising, J_ising)
                J_ext = extend(J_ising)
                h_vect_ext, J_vect_ext = vectorize(h_ising, J_ext)
                if not np.allclose(J_vect_ext, J_vect_ext.T):
                    raise ValueError(
                        "Expected symmetric J matrix for pseudo-likelihood, got non-symmetric J_ext vectorization."
                    )

            data = load_raw_data_file(file_path)
            sample_vectors = np.asarray(data["sample"], dtype=np.float64)
            init_state_vectors = np.asarray(data["init_state"], dtype=np.float64)
            energy_vectors = np.asarray(data["energy"], dtype=np.float64).reshape(-1)

            if sample_vectors.shape[0] == 0:
                skipped_empty += 1
                print(f"No valid samples found in {filename}")
                continue

            sample_count = sample_vectors.shape[0]
            chain_length_int = int(chain_length)

            init_states = np.array(sample_vectors, copy=True)
            if init_state_vectors.ndim == 2 and init_state_vectors.shape[0] > 0:
                valid_init_count = min(sample_count, init_state_vectors.shape[0])
                init_states[:valid_init_count] = init_state_vectors[:valid_init_count]

            energy_init_vectors = (
                np.einsum("bi,ij,bj->b", init_states, J_vect, init_states) + init_states @ h_vect
            )
            E_final = energy_vectors / chain_length_int
            Q_vect = (energy_vectors - energy_init_vectors) / chain_length_int

            optim = optimize.minimize(
                pseudo_likelihood,
                np.array([1.0]),
                args=(h_ising, J_ext, sample_vectors, h_vect_ext, J_vect_ext),
            )

            with open(betas_path, "wb") as f:
                betas[key] = optim.x.item() * beta_units
                pickle.dump(betas, f)
            print("result: beta = ", optim.x.item())

            with open(energies_path, "wb") as f2:
                E_mean, E_var = np.mean(E_final), np.var(E_final)
                energies[key] = (E_mean * energy_units, E_var * energy_units**2)
                pickle.dump(energies, f2)
            print("result: energies = ", (E_mean, E_var))

            with open(q_path, "wb") as f3:
                Q_mean, Q_var = np.mean(Q_vect), np.var(Q_vect)
                Q[key] = (Q_mean * energy_units, Q_var * energy_units**2)
                pickle.dump(Q, f3)
            print("result: Q = ", (Q_mean, Q_var))

            with open(q_dist_path, "wb") as f4:
                Q_dist[key] = Q_vect
                pickle.dump(Q_dist, f4)

            processed += 1

    print("\nOptimization summary")
    print(f"  data files seen:          {total_files}")
    print(f"  filenames matched regex:  {matched_name}")
    print(f"  selected by filters:      {selected_by_filter}")
    print(f"  processed:                {processed}")
    print(f"  skipped existing results: {skipped_existing}")
    print(f"  skipped bad filename:     {skipped_bad_name}")
    print(f"  skipped missing instance: {skipped_instance}")
    print(f"  skipped empty sample set: {skipped_empty}")


if __name__ == "__main__":
    main()
