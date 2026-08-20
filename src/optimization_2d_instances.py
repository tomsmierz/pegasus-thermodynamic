import argparse
import os
import pickle
import re

import numpy as np
from scipy import optimize

from raw_data_io import load_raw_data_file
from utils import pseudo_likelihood_2d_vectorised, vectorize_2d

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
    r"raw_data_([PZ])6_([A-Z]+)_(\d+)_beta_([\d.]+)_at_([\d.]+)_ap_([\d.]+)\.(?:csv|npz)$"
)


def _parse_float_list(raw: str | None) -> set[float] | None:
    if raw is None:
        return None
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        return None
    return {float(x) for x in values}


def _parse_int_list(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def _parse_str_list(raw: str) -> list[str]:
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


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


def _data_dir_name(topology_prefix: str, data_family: str, inst_type: str) -> str:
    prefix = "phase_diagram" if data_family == "standard" else "more_sweeps_phase_diagram"
    system = "advantage6.4" if topology_prefix == "P" else "advantage2"
    return f"{prefix}_{topology_prefix}6_{inst_type}_{system}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize 2D presampled raw data with optional parameter filters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--instance-types",
        default="CBFM,RAU",
        help="Comma-separated 2D instance types",
    )
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="Optional comma-separated instance ids (for example: 1,2,3)",
    )
    parser.add_argument(
        "--anneal-time",
        default=None,
        help="Optional anneal_time filter (for example: 100)",
    )
    parser.add_argument(
        "--ap-list",
        default=None,
        help="Optional comma-separated anneal_param values (for example: 0.700,0.790)",
    )
    parser.add_argument(
        "--beta-list",
        default=None,
        help="Optional comma-separated beta values; defaults to all discovered values",
    )
    parser.add_argument(
        "--data-family",
        default="standard",
        choices=["standard", "more_sweeps"],
        help="Raw data folder family to read",
    )
    parser.add_argument(
        "--topology",
        default="zephyr",
        choices=["pegasus", "zephyr"],
        help="2D topology family to optimize",
    )
    parser.add_argument(
        "--results-base-dir",
        default=os.path.join(ROOT, "data", "results"),
        help="Base directory under which per-type result folders are written",
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


def _load_2d_instance(instance_path: str) -> tuple[dict, dict]:
    with open(instance_path, "rb") as f:
        inst_data = pickle.load(f)

    if hasattr(inst_data, "h") and hasattr(inst_data, "J"):
        return inst_data.h, inst_data.J
    if isinstance(inst_data, (list, tuple)) and len(inst_data) == 2:
        return inst_data[0], inst_data[1]
    raise ValueError(f"Unknown instance format in {instance_path}")


def main() -> None:
    args = parse_args()
    inst_types = _parse_str_list(args.instance_types)
    instance_id_filter = _parse_int_list(args.instance_ids)
    ap_filter = _parse_float_list(args.ap_list)
    beta_filter = _parse_float_list(args.beta_list)
    topology_prefix = "P" if args.topology == "pegasus" else "Z"
    instance_subdir = "subpegasus_native" if args.topology == "pegasus" else "subzephyr_native"
    chain_length = "6"
    chain_length_int = 6

    total_files = 0
    matched_name = 0
    selected_by_filter = 0
    processed = 0
    skipped_bad_name = 0
    skipped_empty = 0
    skipped_instance = 0
    skipped_existing = 0

    for inst_type in inst_types:
        data_dir_name = _data_dir_name(topology_prefix, args.data_family, inst_type)
        data_dir = os.path.join(ROOT, "data", "raw_data", data_dir_name)
        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            continue

        results_dir = os.path.join(args.results_base_dir, data_dir_name)
        os.makedirs(results_dir, exist_ok=True)

        betas_path = os.path.join(results_dir, f"betas2_P{chain_length}.pkl")
        energies_path = os.path.join(results_dir, f"energies_P{chain_length}.pkl")
        q_path = os.path.join(results_dir, f"Q_P{chain_length}.pkl")
        q_dist_path = os.path.join(results_dir, f"Q_dist_P{chain_length}.pkl")

        betas: dict[tuple[str, str, str, str], float] = _load_pickle_if_exists(betas_path)
        energies: dict[tuple[str, str, str, str], tuple[float, float]] = _load_pickle_if_exists(
            energies_path
        )
        Q: dict[tuple[str, str, str, str], tuple[float, float]] = _load_pickle_if_exists(q_path)
        Q_dist: dict[tuple[str, str, str, str], np.ndarray] = _load_pickle_if_exists(q_dist_path)

        def sort_key(filename: str) -> tuple[float, float, float, float]:
            m = FILENAME_RE.match(filename)
            if not m:
                return (float("inf"), float("inf"), float("inf"), float("inf"))
            return (int(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)))

        for filename in sorted(os.listdir(data_dir), key=sort_key):
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

            file_topology_prefix = m.group(1)
            file_inst_type = m.group(2)
            instance_num = m.group(3)
            beta = m.group(4)
            anneal_time = m.group(5)
            anneal_param = m.group(6)

            if file_topology_prefix != topology_prefix or file_inst_type != inst_type:
                skipped_bad_name += 1
                print(f"Skipping mismatched instance type in filename: {filename}")
                continue

            if instance_id_filter is not None and int(instance_num) not in instance_id_filter:
                continue
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

            key = (instance_num, beta, anneal_time, anneal_param)
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
                instance_subdir,
                f"{topology_prefix}6_{inst_type}_{instance_num}.pkl",
            )
            if not os.path.exists(instance_path):
                skipped_instance += 1
                print(f"Instance not found: {instance_path}")
                continue

            h_ising, J_ising = _load_2d_instance(instance_path)
            h_vect, J_vect, _, _, _ = vectorize_2d(h_ising, J_ising)
            model_order = list(h_ising.keys())

            data = load_raw_data_file(file_path)
            coordinates = list(np.asarray(data["coordinates"]).tolist())
            sample_vectors = np.asarray(data["sample"], dtype=np.float64)
            init_state_vectors = np.asarray(data["init_state"], dtype=np.float64)
            energy_vectors = np.asarray(data["energy"], dtype=np.float64).reshape(-1)

            if sample_vectors.shape[0] == 0:
                skipped_empty += 1
                print(f"No valid samples found in {filename}")
                continue

            coord_to_col = {coordinates[i]: i for i in range(len(coordinates))}
            try:
                column_order = np.array([coord_to_col[key] for key in model_order], dtype=np.int64)
            except KeyError as err:
                raise ValueError(
                    f"Sample coordinates in {filename} do not match instance keys; missing key: {err}"
                ) from err

            aligned_samples = sample_vectors[:, column_order]
            sample_count = aligned_samples.shape[0]

            aligned_init_states = np.array(aligned_samples, copy=True)
            if init_state_vectors.ndim == 2 and init_state_vectors.shape[0] > 0:
                valid_init_count = min(sample_count, init_state_vectors.shape[0])
                aligned_init_states[:valid_init_count] = init_state_vectors[:valid_init_count, column_order]

            energy_init_vectors = (
                np.einsum("bi,ij,bj->b", aligned_init_states, J_vect, aligned_init_states)
                + aligned_init_states @ h_vect
            )
            E_final = energy_vectors / chain_length_int
            Q_vect = (energy_vectors - energy_init_vectors) / chain_length_int

            optim = optimize.minimize(
                pseudo_likelihood_2d_vectorised,
                np.array([1.0]),
                args=(h_vect, J_vect, aligned_samples),
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
