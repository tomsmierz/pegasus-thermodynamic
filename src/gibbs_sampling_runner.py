#!/usr/bin/env python3
"""gibbs_sampling_runner_fast.py

Optimized version of the Gibbs sampling runner.

Main speedups vs. the baseline runner:
  - spins stored as a NumPy int8 array rather than a Python dict
  - precomputes linear terms h and an adjacency list with integer indices
  - tracks energy and magnetization incrementally (O(1) per sweep to read)
  - avoids opening/closing the diagnostics CSV for every sample

The command-line interface and output format are kept intentionally similar.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
from tqdm import tqdm

try:
    from dimod import BinaryQuadraticModel
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This script requires 'dimod'. Install it in your environment (e.g. pip install dimod)."
    ) from e


# RNG (seeded in main)
rng = np.random.default_rng()


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("gibbs_sampling")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console_handler)

    log_file = os.path.join(output_dir, f"gibbs_sampling_{datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(funcName)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


def prob_spin_plus_one(beta_deltaE: float) -> float:
    """Return P(s=+1) = 1/(1+exp(beta*ΔE)) in a numerically stable way."""
    # For large +x we evaluate exp(-x) to avoid overflow.
    x = beta_deltaE
    if x >= 0.0:
        z = math.exp(-x)  # small
        return z / (1.0 + z)
    else:
        z = math.exp(x)  # small
        return 1.0 / (1.0 + z)


def next_pow_two(n: int) -> int:
    i = 1
    while i < n:
        i <<= 1
    return i


def compute_autocorrelation_fft(x: np.ndarray) -> Optional[np.ndarray]:
    n = len(x)
    var = float(np.var(x))
    if var == 0.0:
        return None
    n_fft = next_pow_two(2 * n)
    x0 = x - float(np.mean(x))
    f = np.fft.fft(x0, n=n_fft)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    return acf / acf[0]


def compute_integrated_autocorrelation_time(x: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Sokal-style τ_int = 1/2 + Σ_{t>=1} ρ(t), truncated at first non-positive ρ."""
    acf = compute_autocorrelation_fft(x)
    if acf is None:
        return float("inf")
    if max_lag is None:
        max_lag = len(acf)
    tau = 0.5
    for k in range(1, max_lag):
        if acf[k] <= 0.0:
            break
        tau += float(acf[k])
    return float(tau)


def tau_cap(beta: float) -> Optional[float]:
    if beta > 1.5:
        return 2000.0
    elif beta > 0.8:
        return 3000.0
    return None


def should_record_burn_in_sweep(sweep_number: int) -> bool:
    """Keep all sweeps through 1000, then every 100 sweeps after that."""
    return sweep_number <= 1000 or (sweep_number % 100 == 0)


def load_instance(instance_path: str, logger: logging.Logger) -> BinaryQuadraticModel:
    logger.info(f"Loading instance from: {instance_path}")

    if instance_path.endswith(".pkl"):
        with open(instance_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list) and len(data) == 2:
            h, J = data[0], data[1]
        elif isinstance(data, tuple) and len(data) == 2:
            h, J = data
        elif isinstance(data, dict):
            h = data.get("h", {})
            J = data.get("J", {})
        else:
            raise ValueError(f"Unexpected pickle format: {type(data)}")
    elif instance_path.endswith(".json"):
        import json

        with open(instance_path, "r") as f:
            data = json.load(f)
        h = {int(k): v for k, v in data.get("h", {}).items()}
        J = {tuple(map(int, k.strip("()").split(","))): v for k, v in data.get("J", {}).items()}
    else:
        raise ValueError(f"Unsupported file format: {instance_path}")

    bqm = BinaryQuadraticModel.from_ising(h, J)

    logger.info("Instance loaded successfully")
    logger.info(f"  Number of spins: {bqm.num_variables}")
    logger.info(f"  Number of couplings: {bqm.num_interactions}")
    return bqm


def compile_bqm(bqm: BinaryQuadraticModel):
    """Compile BQM into array-friendly structures for fast Gibbs updates."""
    nodes = list(bqm.variables)
    n = len(nodes)
    node_to_i = {v: i for i, v in enumerate(nodes)}
    h = np.array([float(bqm.get_linear(v)) for v in nodes], dtype=np.float64)

    # adjacency list: for each i, a list of (j_index, coupling)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i, v in enumerate(nodes):
        for u, bias in bqm.adj[v].items():
            adj[i].append((node_to_i[u], float(bias)))
    return nodes, node_to_i, h, adj


def load_last_sample_from_directory(
    samples_dir: str,
    nodes: List,
    bqm: BinaryQuadraticModel,
    logger: logging.Logger,
    energy_tolerance: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Load the latest Gibbs sample from a directory and validate its energy."""
    if not os.path.isdir(samples_dir):
        raise ValueError(f"Resume directory does not exist: {samples_dir}")

    candidate_files = [
        os.path.join(samples_dir, name)
        for name in os.listdir(samples_dir)
        if name.endswith(".npz")
    ]
    if not candidate_files:
        raise ValueError(f"No .npz files found in resume directory: {samples_dir}")

    node_to_target_index = {node: i for i, node in enumerate(nodes)}

    # Try newest files first and select the first valid Gibbs-sample container.
    for npz_path in sorted(candidate_files, key=os.path.getmtime, reverse=True):
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "samples" not in data:
                    continue
                samples = data["samples"]
                if samples.ndim != 2 or samples.shape[0] == 0:
                    continue

                latest_sample = np.asarray(samples[-1], dtype=np.int8)
                if latest_sample.ndim != 1:
                    continue

                if "node_order" in data:
                    source_nodes = [int(v) for v in np.asarray(data["node_order"]).tolist()]
                    if len(source_nodes) != len(nodes):
                        raise ValueError(
                            f"Node count mismatch in {npz_path}: "
                            f"file has {len(source_nodes)}, instance has {len(nodes)}"
                        )
                    if set(source_nodes) != set(nodes):
                        raise ValueError(
                            f"Node labels in {npz_path} do not match currently loaded instance"
                        )

                    source_index = {node: i for i, node in enumerate(source_nodes)}
                    reordered = np.empty((len(nodes),), dtype=np.int8)
                    for node, target_idx in node_to_target_index.items():
                        reordered[target_idx] = int(latest_sample[source_index[node]])
                    latest_sample = reordered
                elif latest_sample.shape[0] != len(nodes):
                    raise ValueError(
                        f"Sample length mismatch in {npz_path}: "
                        f"file has {latest_sample.shape[0]}, instance has {len(nodes)}"
                    )

                if not np.all(np.isin(latest_sample, np.array([-1, 1], dtype=np.int8))):
                    raise ValueError(f"Last sample in {npz_path} is not a valid Ising spin state")

                recomputed_energy = float(
                    bqm.energy({nodes[i]: int(latest_sample[i]) for i in range(len(nodes))})
                )

                if "sample_energies" in data and data["sample_energies"].size > 0:
                    stored_last_energy = float(np.asarray(data["sample_energies"])[-1])
                    diff = abs(stored_last_energy - recomputed_energy)
                    if diff > energy_tolerance:
                        raise ValueError(
                            f"Energy validation failed for {npz_path}: "
                            f"stored={stored_last_energy:.10f}, recomputed={recomputed_energy:.10f}, "
                            f"diff={diff:.3e}"
                        )

                logger.info(f"Resuming from file: {npz_path}")
                logger.info(
                    f"Loaded last sample with validated energy: {recomputed_energy:.4f}"
                )
                return latest_sample, recomputed_energy
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Skipping unreadable resume candidate {npz_path}: {e}")
            continue

    raise ValueError(
        f"No valid Gibbs sample file found in directory: {samples_dir}. "
        f"Expected an .npz with a non-empty 'samples' array."
    )


def get_existing_sample_count(samples_path: str) -> int:
    """Return the number of already-saved samples in an .npz file, or 0 if unavailable."""
    if not os.path.isfile(samples_path):
        return 0

    try:
        with np.load(samples_path, allow_pickle=True) as data:
            if "samples" not in data:
                return 0
            samples = np.asarray(data["samples"])
            if samples.ndim != 2:
                return 0
            return int(samples.shape[0])
    except Exception:
        return 0


def load_existing_output_samples(
    samples_path: str,
    nodes: List,
    bqm: BinaryQuadraticModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load existing output arrays and return last sample aligned to current node order.

    Returns:
      samples, energies, magnetizations, last_sample, last_energy
    """
    if not os.path.isfile(samples_path):
        raise ValueError(f"Existing output file not found: {samples_path}")

    with np.load(samples_path, allow_pickle=True) as data:
        if "samples" not in data:
            raise ValueError(f"Missing 'samples' in existing output: {samples_path}")

        samples = np.asarray(data["samples"], dtype=np.int8)
        if samples.ndim != 2 or samples.shape[0] == 0:
            raise ValueError(f"Existing 'samples' array is empty or invalid: {samples_path}")

        if "sample_energies" in data and data["sample_energies"].size == samples.shape[0]:
            sample_energies = np.asarray(data["sample_energies"], dtype=np.float64)
        else:
            sample_energies = np.empty((samples.shape[0],), dtype=np.float64)
            for i in range(samples.shape[0]):
                sample_energies[i] = float(
                    bqm.energy({nodes[j]: int(samples[i, j]) for j in range(len(nodes))})
                )

        if "sample_magnetizations" in data and data["sample_magnetizations"].size == samples.shape[0]:
            sample_mags = np.asarray(data["sample_magnetizations"], dtype=np.float64)
        else:
            sample_mags = samples.mean(axis=1, dtype=np.float64)

        if "node_order" in data:
            source_nodes = [int(v) for v in np.asarray(data["node_order"]).tolist()]
            if len(source_nodes) != len(nodes):
                raise ValueError(
                    f"Node count mismatch in existing output: file has {len(source_nodes)}, "
                    f"instance has {len(nodes)}"
                )
            if set(source_nodes) != set(nodes):
                raise ValueError("Node labels in existing output do not match loaded instance")

            source_index = {node: i for i, node in enumerate(source_nodes)}
            target_index = {node: i for i, node in enumerate(nodes)}
            aligned = np.empty_like(samples)
            for node, t_idx in target_index.items():
                aligned[:, t_idx] = samples[:, source_index[node]]
            samples = aligned
            sample_mags = samples.mean(axis=1, dtype=np.float64)

        if samples.shape[1] != len(nodes):
            raise ValueError(
                f"Spin dimension mismatch in existing output: file has {samples.shape[1]}, "
                f"instance has {len(nodes)}"
            )

        last_sample = np.asarray(samples[-1], dtype=np.int8)
        if not np.all(np.isin(last_sample, np.array([-1, 1], dtype=np.int8))):
            raise ValueError("Last sample in existing output is not a valid Ising spin state")

        last_energy = float(
            bqm.energy({nodes[i]: int(last_sample[i]) for i in range(len(nodes))})
        )

    return samples, sample_energies, sample_mags, last_sample, last_energy


def gibbs_sweep_fast(
    spins: np.ndarray,
    energy: float,
    m_sum: int,
    h: np.ndarray,
    adj: List[List[Tuple[int, float]]],
    beta: float,
) -> Tuple[float, int]:
    """One random-scan sweep: N single-site Gibbs updates.

    Returns updated (energy, m_sum). Spins are updated in-place.
    """
    n = spins.shape[0]

    for _ in range(n):
        i = int(rng.integers(0, n))

        # local field f_i = h_i + Σ_j J_ij s_j
        field = float(h[i])
        for j, bias in adj[i]:
            field += bias * float(spins[j])

        # ΔE = E(+1)-E(-1) = 2 f_i
        deltaE = 2.0 * field
        p_plus = prob_spin_plus_one(beta * deltaE)

        new = 1 if rng.random() < p_plus else -1
        old = int(spins[i])
        if new != old:
            spins[i] = new
            energy += (new - old) * field
            m_sum += (new - old)

    return energy, m_sum


def run_burn_in_phase_fast(
    spins: np.ndarray,
    energy: float,
    m_sum: int,
    h: np.ndarray,
    adj: List[List[Tuple[int, float]]],
    beta: float,
    num_burn_in_sweeps: Optional[int],
    output_dir: str,
    logger: logging.Logger,
    energy_check_window: int = 100,
    convergence_threshold: float = 0.01,
    post_burn_in_sweeps: int = 1000,
) -> Tuple[float, int, int]:
    logger.info("=" * 60)
    logger.info("BURN-IN PHASE")
    logger.info("=" * 60)

    energy_history: List[float] = []
    mag_history: List[float] = []
    sweep_indices: List[int] = []

    sweeps_done = 0
    converged = False

    if num_burn_in_sweeps is not None:
        target = int(num_burn_in_sweeps)
        logger.info(f"Running fixed burn-in: {target} sweeps")
        pbar = tqdm(range(target), desc="Burn-in", unit="sweep")
        for i in pbar:
            energy, m_sum = gibbs_sweep_fast(spins, energy, m_sum, h, adj, beta)
            m = m_sum / spins.shape[0]
            sweeps_done += 1
            if should_record_burn_in_sweep(sweeps_done):
                energy_history.append(float(energy))
                mag_history.append(float(m))
                sweep_indices.append(sweeps_done)
            if (i + 1) % 100 == 0:
                pbar.set_description(f"Burn-in | E: {energy:.4f} | M: {m:.4f}")
        converged = True
    else:
        logger.info(
            f"Running adaptive burn-in (window={energy_check_window}, thresh={convergence_threshold})"
        )
        min_sweeps = 2 * int(energy_check_window)
        max_sweeps = 200_000
        from collections import deque

        recent_E = deque(maxlen=energy_check_window)
        prev_E = deque(maxlen=energy_check_window)
        recent_M = deque(maxlen=energy_check_window)
        prev_M = deque(maxlen=energy_check_window)

        pbar = tqdm(total=max_sweeps, desc="Adaptive Burn-in", unit="sweep")
        while not converged and sweeps_done < max_sweeps:
            energy, m_sum = gibbs_sweep_fast(spins, energy, m_sum, h, adj, beta)
            m = m_sum / spins.shape[0]

            current_sweep = sweeps_done + 1
            if should_record_burn_in_sweep(current_sweep):
                energy_history.append(float(energy))
                mag_history.append(float(m))
                sweep_indices.append(current_sweep)

            recent_E.append(float(energy))
            recent_M.append(float(m))
            if sweeps_done >= energy_check_window:
                prev_E.append(float(recent_E[0]))
                prev_M.append(float(recent_M[0]))

            sweeps_done += 1
            pbar.update(1)
            if sweeps_done % 100 == 0:
                pbar.set_description(f"Adap Burn-in | E: {energy:.4f} | M: {m:.4f}")

            if sweeps_done >= min_sweeps and sweeps_done % 100 == 0:
                curr_E_mean, curr_E_std = float(np.mean(recent_E)), float(np.std(recent_E))
                prev_E_mean, prev_E_std = float(np.mean(prev_E)), float(np.std(prev_E))
                curr_M_mean, curr_M_std = float(np.mean(recent_M)), float(np.std(recent_M))
                prev_M_mean, prev_M_std = float(np.mean(prev_M)), float(np.std(prev_M))

                def rel_change(curr: float, prev: float) -> float:
                    if abs(prev) < 1e-9:
                        return abs(curr - prev)
                    return abs(curr - prev) / abs(prev)

                d_E_mean = rel_change(curr_E_mean, prev_E_mean)
                d_E_std = rel_change(curr_E_std, prev_E_std)
                d_M_mean = rel_change(curr_M_mean, prev_M_mean)
                d_M_std = rel_change(curr_M_std, prev_M_std)

                if (
                    d_E_mean < convergence_threshold
                    and d_E_std < convergence_threshold
                    and d_M_mean < convergence_threshold
                    and d_M_std < convergence_threshold
                ):
                    converged = True
                    logger.info(f"Converged at sweep {sweeps_done}")
                    logger.info(f"dE_mean: {d_E_mean:.2e}, dE_std: {d_E_std:.2e}")
                    logger.info(f"dM_mean: {d_M_mean:.2e}, dM_std: {d_M_std:.2e}")

        pbar.close()
        if not converged:
            logger.warning("Burn-in reached max_sweeps without full convergence.")

    # strict post-burn buffer
    if post_burn_in_sweeps > 0:
        logger.info(f"Running buffer: {post_burn_in_sweeps} sweeps")
        pbar = tqdm(range(post_burn_in_sweeps), desc="Buffer", unit="sweep")
        for i in pbar:
            energy, m_sum = gibbs_sweep_fast(spins, energy, m_sum, h, adj, beta)
            if (i + 1) % 100 == 0:
                m = m_sum / spins.shape[0]
                pbar.set_description(f"Buffer | M: {m:.4f}")

    # save burn-in diagnostics
    burn_in_array = np.column_stack((
        np.asarray(sweep_indices, dtype=np.int64),
        np.asarray(energy_history, dtype=np.float64),
        np.asarray(mag_history, dtype=np.float64),
    ))
    burn_in_path = os.path.join(output_dir, "burn_in_diagnostics.npz")
    np.savez_compressed(burn_in_path, burn_in=burn_in_array)
    logger.info(
        "Saved burn-in diagnostics to %s with %d points from %d sweeps",
        burn_in_path,
        burn_in_array.shape[0],
        sweeps_done,
    )

    return energy, m_sum, sweeps_done


def run_sampling_phase_fast(
    spins: np.ndarray,
    energy: float,
    m_sum: int,
    h: np.ndarray,
    adj: List[List[Tuple[int, float]]],
    beta: float,
    num_sampling_sweeps: Optional[int],
    num_samples: int,
    output_dir: str,
    logger: logging.Logger,
    safety_factor: int = 4,
    sample_index_offset: int = 0,
    diag_append: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("=" * 60)
    logger.info("SAMPLING PHASE (FAST | ENERGY-CONTROLLED)")
    logger.info("=" * 60)

    n = spins.shape[0]
    samples = np.empty((num_samples, n), dtype=np.int8)
    sample_energies = np.empty((num_samples,), dtype=np.float64)
    sample_mags = np.empty((num_samples,), dtype=np.float64)

    from collections import deque

    acf_window_size = 500
    energy_window = deque(maxlen=acf_window_size)
    mag_window = deque(maxlen=acf_window_size)

    use_fixed = num_sampling_sweeps is not None
    if use_fixed:
        required_interval = int(num_sampling_sweeps)
        logger.info(f"Using fixed sampling interval: {required_interval}")
    else:
        required_interval = acf_window_size + 1
        logger.info("Using adaptive sampling (Energy-based)")
        logger.info(f"ACF Window Size: {acf_window_size}")

    diag_csv_path = os.path.join(output_dir, "sampling_diagnostics.csv")
    write_header = True
    diag_mode = "a" if diag_append else "w"
    if diag_append and os.path.isfile(diag_csv_path) and os.path.getsize(diag_csv_path) > 0:
        write_header = False

    diag_f = open(diag_csv_path, diag_mode, buffering=1)  # line-buffered
    if write_header:
        diag_f.write(
            "sample_idx,sweep,energy,magnetization,tau_raw_energy,tau_eff_energy,tau_raw_mag,"
            "sampling_interval,beta,sweeps_waited\n"
        )

    sweeps_since_last_sample = 0
    total_sweeps = 0
    collected = 0
    window_filled = False

    current_tau_energy = 10.0
    current_tau_eff = 10.0
    tau_raw_mag = 0.0

    pbar = tqdm(total=num_samples, desc="Sampling", unit="sample")
    try:
        while collected < num_samples:
            energy, m_sum = gibbs_sweep_fast(spins, energy, m_sum, h, adj, beta)
            total_sweeps += 1
            sweeps_since_last_sample += 1

            m = m_sum / n

            # Only maintain ACF windows if we are in adaptive mode
            if not use_fixed:
                energy_window.append(float(energy))
                mag_window.append(float(m))
                if not window_filled and len(energy_window) == acf_window_size:
                    window_filled = True
                    logger.info("ACF Window filled. Starting adaptive estimation.")

                if window_filled and total_sweeps % 50 == 0:
                    e_arr = np.fromiter(energy_window, dtype=np.float64)
                    tau_raw_energy = compute_integrated_autocorrelation_time(e_arr)
                    frozen = not np.isfinite(tau_raw_energy)

                    cap = tau_cap(beta)
                    if frozen:
                        current_tau_eff = cap if cap is not None else float("inf")
                    else:
                        current_tau_eff = min(tau_raw_energy, cap) if cap is not None else tau_raw_energy

                    m_arr = np.fromiter(mag_window, dtype=np.float64)
                    tau_raw_mag = compute_integrated_autocorrelation_time(m_arr)
                    current_tau_energy = tau_raw_energy

                    if frozen:
                        required_interval = 10
                        pbar.set_description("Sampling | frozen phase (τ undefined)")
                    else:
                        required_interval = int(math.ceil(safety_factor * current_tau_eff))
                        required_interval = max(required_interval, 100)
                        pbar.set_description(
                            f"Sampling | tau_eff: {current_tau_eff:.1f} | int: {required_interval}"
                        )

            # decide to sample
            if use_fixed:
                ready = sweeps_since_last_sample >= required_interval
            else:
                ready = window_filled and sweeps_since_last_sample >= required_interval

            if ready:
                samples[collected, :] = spins
                sample_energies[collected] = energy
                sample_mags[collected] = m

                diag_f.write(
                    f"{sample_index_offset + collected},{total_sweeps},{energy:.4f},{m:.4f},"
                    f"{current_tau_energy:.4f},{current_tau_eff:.4f},{tau_raw_mag:.4f},"
                    f"{required_interval},{beta},{sweeps_since_last_sample}\n"
                )

                collected += 1
                sweeps_since_last_sample = 0
                pbar.update(1)

    finally:
        pbar.close()
        diag_f.close()

    return samples, sample_energies, sample_mags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gibbs Sampling with Adaptive Burn-in and Sampling (FAST)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)

    parser.add_argument("--num_burn_in_sweeps", type=int, default=None)
    parser.add_argument("--burn_in_window", type=int, default=100)
    parser.add_argument("--burn_in_threshold", type=float, default=0.01)
    parser.add_argument("--post_burn_in_sweeps", type=int, default=10000)

    parser.add_argument("--num_sampling_sweeps", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1000)

    parser.add_argument("--output_dir", type=str, default="./gibbs_results")
    parser.add_argument("--output_prefix", type=str, default="gibbs")
    parser.add_argument(
        "--resume_samples_dir",
        type=str,
        default=None,
        help="Directory with existing Gibbs .npz samples. Uses the last sample as initial state.",
    )

    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    target_beta = float(args.beta)

    instance_basename = os.path.splitext(os.path.basename(args.instance_path))[0]
    beta_subdir = f"BETA_{target_beta:.3f}"
    output_dir = os.path.join(args.output_dir, instance_basename, beta_subdir)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir, args.log_level)

    samples_path = os.path.join(output_dir, f"{args.output_prefix}_samples.npz")
    existing_sample_count = get_existing_sample_count(samples_path)
    if existing_sample_count >= args.num_samples:
        logger.info("=" * 60)
        logger.info("SKIPPING RUN")
        logger.info("=" * 60)
        logger.info(
            "Requested %d samples, found %d existing in %s. Nothing to do.",
            args.num_samples,
            existing_sample_count,
            samples_path,
        )
        return

    global rng
    rng = np.random.default_rng(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    logger.info("=" * 60)
    logger.info("GIBBS SAMPLING RUNNER (FAST)")
    logger.info("=" * 60)
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")

    bqm = load_instance(args.instance_path, logger)
    nodes, node_to_i, h, adj = compile_bqm(bqm)
    n = len(nodes)

    existing_samples = None
    existing_sample_energies = None
    existing_sample_mags = None
    resuming_from_existing_output = existing_sample_count > 0

    if resuming_from_existing_output:
        logger.info("Resuming from existing output samples file...")
        (
            existing_samples,
            existing_sample_energies,
            existing_sample_mags,
            spins,
            energy,
        ) = load_existing_output_samples(
            samples_path=samples_path,
            nodes=nodes,
            bqm=bqm,
        )
        logger.info(
            "Loaded %d existing samples from %s",
            existing_samples.shape[0],
            samples_path,
        )
    elif args.resume_samples_dir:
        if os.path.exists(args.resume_samples_dir):
            logger.info("Initializing from existing Gibbs sample...")
            spins, energy = load_last_sample_from_directory(
                samples_dir=args.resume_samples_dir,
                nodes=nodes,
                bqm=bqm,
                logger=logger,
            )
        else:
            logger.warning(
                "Resume samples directory does not exist: %s. "
                "Initializing random spin configuration instead.",
                args.resume_samples_dir,
            )
            logger.info("Initializing random spin configuration...")
            spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
            energy = float(bqm.energy({nodes[i]: int(spins[i]) for i in range(n)}))
    else:
        logger.info("Initializing random spin configuration...")
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
        energy = float(bqm.energy({nodes[i]: int(spins[i]) for i in range(n)}))

    m_sum = int(spins.sum())
    logger.info(f"Initial energy: {energy:.4f}")
    logger.info(f"Initial magnetization: {m_sum / n:.4f}")

    if resuming_from_existing_output:
        logger.info(
            "Skipping burn-in because existing output already has samples."
        )
        burn_in_sweeps_done = 0
        samples_to_collect = args.num_samples - existing_sample_count
        logger.info(
            "Collecting %d additional samples to reach requested total of %d.",
            samples_to_collect,
            args.num_samples,
        )
    else:
        energy, m_sum, burn_in_sweeps_done = run_burn_in_phase_fast(
            spins=spins,
            energy=energy,
            m_sum=m_sum,
            h=h,
            adj=adj,
            beta=target_beta,
            num_burn_in_sweeps=args.num_burn_in_sweeps,
            output_dir=output_dir,
            logger=logger,
            energy_check_window=args.burn_in_window,
            convergence_threshold=args.burn_in_threshold,
            post_burn_in_sweeps=args.post_burn_in_sweeps,
        )
        samples_to_collect = args.num_samples

    samples, sample_energies, sample_mags = run_sampling_phase_fast(
        spins=spins,
        energy=energy,
        m_sum=m_sum,
        h=h,
        adj=adj,
        beta=target_beta,
        num_sampling_sweeps=args.num_sampling_sweeps,
        num_samples=samples_to_collect,
        output_dir=output_dir,
        logger=logger,
        sample_index_offset=existing_sample_count,
        diag_append=resuming_from_existing_output,
    )

    if resuming_from_existing_output:
        assert existing_samples is not None
        assert existing_sample_energies is not None
        assert existing_sample_mags is not None
        samples = np.concatenate([existing_samples, samples], axis=0)
        sample_energies = np.concatenate([existing_sample_energies, sample_energies], axis=0)
        sample_mags = np.concatenate([existing_sample_mags, sample_mags], axis=0)

    logger.info("=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    np.savez_compressed(
        samples_path,
        samples=samples,
        beta=target_beta,
        node_order=np.array(nodes, dtype=object),
        h_values=h,
        instance_path=args.instance_path,
        sample_energies=sample_energies,
        sample_magnetizations=sample_mags,
    )
    logger.info(f"Samples saved to: {samples_path}")

    metadata = {
        "instance_path": args.instance_path,
        "beta": target_beta,
        "num_burn_in_sweeps": args.num_burn_in_sweeps if args.num_burn_in_sweeps else burn_in_sweeps_done,
        "num_sampling_sweeps": args.num_sampling_sweeps,
        "num_samples": args.num_samples,
        "num_spins": n,
        "seed": args.seed,
    }
    metadata_path = os.path.join(output_dir, f"{args.output_prefix}_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata saved to: {metadata_path}")

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples collected: {samples.shape[0]}")
    logger.info(f"Sample shape: {samples.shape}")
    logger.info(f"Sample energy: {sample_energies.mean():.4f} ± {sample_energies.std():.4f}")
    logger.info(f"Sample magnetization: {sample_mags.mean():.4f} ± {sample_mags.std():.4f}")


if __name__ == "__main__":
    main()
