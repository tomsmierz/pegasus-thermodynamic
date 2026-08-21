import os
import pickle
import gc
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from utils import energy, vectorize
import glob
import re
from minorminer import find_embedding
from raw_data_io import get_raw_data_row_count, write_raw_data_npz_from_dataframe

# Helper to load precomputed Gibbs samples

def load_gibbs_samples_1d(gibbs_results_dir, chain_length, hname, beta):
    """
    Load precomputed Gibbs samples for 1D chain.
    """
    # Directory: gibbs_results_dir/pegasus_1D_chain_L_{chain_length}_{hname}/BETA_{beta:.3f}/gibbs_samples.npz
    beta_subdir = f"BETA_{beta:.3f}"
    samples_path = os.path.join(
        gibbs_results_dir,
        f"pegasus_1D_chain_L_{chain_length}_{hname}",
        beta_subdir,
        "gibbs_samples.npz",
    )
    diagnostics_path = os.path.join(
        gibbs_results_dir,
        f"pegasus_1D_chain_L_{chain_length}_{hname}",
        beta_subdir,
        "sampling_diagnostics.csv",
    )
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Gibbs samples not found at: {samples_path}")
    data = np.load(samples_path, allow_pickle=True)
    samples = data["samples"]
    node_order = data["node_order"]
    diagnostics_df = None
    if os.path.exists(diagnostics_path):
        diagnostics_df = pd.read_csv(diagnostics_path)
    return samples, node_order, diagnostics_df

def convert_sample_to_initial_state(sample, node_order):
    return {int(node): int(spin) for node, spin in zip(node_order, sample)}

def compute_energy_with_bqm(initial_state, h, J):
    import dimod
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    return bqm.energy(initial_state)

def to_builtin(obj):
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {to_builtin(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    return obj

def build_reverse_anneal_schedule(anneal_time, anneal_param, initial_value):
    return [
        [0.0, initial_value],
        [anneal_time / 2, anneal_param],
        [anneal_time, initial_value],
    ]

def get_native_embedding_if_available(h, J, qpu_sampler):
    target_nodes = set(qpu_sampler.nodelist)
    target_edges = {frozenset(edge) for edge in qpu_sampler.edgelist}
    variables = set(h)
    variables.update(node for edge in J for node in edge)
    problem_edges = {frozenset(edge) for edge in J}

    if variables <= target_nodes and problem_edges <= target_edges:
        return {node: [node] for node in variables}
    return None

def get_embedding_qubit_count(h, J, qpu_sampler):
    embedding = get_native_embedding_if_available(h, J, qpu_sampler)
    embedding_kind = "native"
    if embedding is None:
        embedding_kind = "minorminer"
        variables = sorted(set(h) | {node for edge in J for node in edge})
        embedding = find_embedding(J.keys(), qpu_sampler.edgelist)
        missing_variables = [node for node in variables if node not in embedding]
        if missing_variables:
            raise RuntimeError(
                f"Embedding failed for {len(missing_variables)} variables; "
                f"first missing variable: {missing_variables[0]}"
            )

    num_qubits = sum(len(chain) for chain in embedding.values())
    return num_qubits, embedding_kind

def build_timing_initial_state(h):
    return {int(node): 1 for node in h}

def format_time_us(microseconds):
    return (
        f"{microseconds:,.2f} us "
        f"({microseconds / 1_000:,.2f} ms, "
        f"{microseconds / 1_000_000:,.3f} s)"
    )

if __name__ == "__main__":
    qpu = "pegasus"
    ESTIMATE_DWAVE_TIME_ONLY = True
    qpu_sampler = DWaveSampler(
        solver="Advantage_system6.4",
        token=os.environ["DWAVE_API_TOKEN"],
    )
    NUM_SAMPLES = 200
    anneal_time = [100]
    NUM_READS = 50
    initial_value = 1.0
    anneal_param = np.arange(0.1, 1.0, 0.1)
    CHAIN_LENGHTS = [1000, 2000]
    hname = ["unif_small_h"]
    BETA1_MIN = 0.0
    BETA1_MAX = 6.0
    # BETA1_VALS = set(np.arange(5.05, 6.05, 0.1).round(3))
    BETA1_VALS = [1.0]
    

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    GIBBS_RESULTS_DIR = os.path.join(ROOT, "data", "raw_data", "gibbs_samples_1D")
    sampler = None
    if not ESTIMATE_DWAVE_TIME_ONLY:
        sampler = EmbeddingComposite(qpu_sampler)

    total_estimated_qpu_access_time = 0.0
    total_planned_qmis = 0
    total_planned_reads = 0
    estimate_rows = []

    for chain_length in CHAIN_LENGHTS:
        for name in hname:
            output_path = os.path.join(
                ROOT, "data", "raw_data", f"phase_diagram_pegasus_1D_chain_L_{chain_length}_{name}"
            )
            if not ESTIMATE_DWAVE_TIME_ONLY and not os.path.exists(output_path):
                os.makedirs(output_path)
            filepath = os.path.join(
                ROOT, "data", "instances", f"1d_{qpu}_chains", f"{qpu}_1D_chain_L_{chain_length}_{name}.pkl"
            )
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Instance file not found: {filepath}")
            print(f"loading instance of length {chain_length}, {name}")
            with open(filepath, "rb") as f:
                h, J = pickle.load(f)
            h_vect, J_vect = (None, None)
            if not ESTIMATE_DWAVE_TIME_ONLY:
                h_vect, J_vect = vectorize(h, J)
            embedded_qubits = None
            embedding_kind = None
            if ESTIMATE_DWAVE_TIME_ONLY:
                embedded_qubits, embedding_kind = get_embedding_qubit_count(h, J, qpu_sampler)
                timing_initial_state = build_timing_initial_state(h)
                print(
                    f"Estimated embedding for L={chain_length}, {name}: "
                    f"{embedded_qubits} qubits ({embedding_kind})"
                )
            # Discover available BETA_1 values from gibbs_samples_1D directory
            gibbs_dir = os.path.join(GIBBS_RESULTS_DIR, f"pegasus_1D_chain_L_{chain_length}_{name}")
            if not os.path.exists(gibbs_dir):
                print(f"No gibbs samples directory found for {gibbs_dir}")
                continue
            beta_dirs = [d for d in os.listdir(gibbs_dir) if os.path.isdir(os.path.join(gibbs_dir, d)) and d.startswith("BETA_")]
            beta_values = []
            for d in beta_dirs:
                match = re.match(r"BETA_([\d.]+)", d)
                if match:
                    beta = float(match.group(1))
                    if BETA1_MIN <= beta <= BETA1_MAX and round(beta, 3) in BETA1_VALS:
                        beta_values.append(round(beta, 3))
            beta_values = sorted(beta_values)
            if not beta_values:
                print(f"No BETA_1 values found in {gibbs_dir} within range {BETA1_MIN} to {BETA1_MAX}")
                continue
            for ANNEAL_TIME in anneal_time:
                for ANNEAL_PARAM in anneal_param:
                    for BETA_1 in beta_values:
                        try:
                            gibbs_samples, node_order, diagnostics_df = load_gibbs_samples_1d(
                                GIBBS_RESULTS_DIR, chain_length, name, BETA_1
                            )
                        except FileNotFoundError as e:
                            print(f"Skipping: {e}")
                            continue
                        num_available_samples = len(gibbs_samples)
                        num_samples_to_use = min(NUM_SAMPLES, num_available_samples)
                        results_base = (
                            f"raw_data_chain_L_{chain_length}_at_{ANNEAL_TIME:d}_ap_{ANNEAL_PARAM:.3f}_beta_{BETA_1:.3f}"
                        )
                        results_path = os.path.join(output_path, f"{results_base}.npz")
                        legacy_csv_path = os.path.join(output_path, f"{results_base}.csv")
                        existing_paths = [
                            p for p in (results_path, legacy_csv_path) if os.path.exists(p)
                        ]
                        if existing_paths:
                            row_count = max(get_raw_data_row_count(p) for p in existing_paths)
                            if row_count > num_samples_to_use * NUM_READS * 0.9:
                                print(f"results already exist for cl={chain_length}, at={ANNEAL_TIME}, ap={ANNEAL_PARAM}, b1={BETA_1} with sufficient data")
                                continue
                        anneal_schedule = build_reverse_anneal_schedule(
                            ANNEAL_TIME, ANNEAL_PARAM, initial_value
                        )
                        if ESTIMATE_DWAVE_TIME_ONLY:
                            estimated_qpu_access_time = qpu_sampler.solver.estimate_qpu_access_time(
                                embedded_qubits,
                                num_reads=NUM_READS,
                                anneal_schedule=anneal_schedule,
                                initial_state=timing_initial_state,
                                reinitialize_state=True,
                            )
                            estimated_total_time = estimated_qpu_access_time * num_samples_to_use
                            total_estimated_qpu_access_time += estimated_total_time
                            total_planned_qmis += num_samples_to_use
                            total_planned_reads += num_samples_to_use * NUM_READS
                            estimate_rows.append(
                                {
                                    "chain_length": chain_length,
                                    "hname": name,
                                    "beta": BETA_1,
                                    "anneal_time": ANNEAL_TIME,
                                    "anneal_param": ANNEAL_PARAM,
                                    "num_samples": num_samples_to_use,
                                    "num_reads": NUM_READS,
                                    "embedded_qubits": embedded_qubits,
                                    "embedding_kind": embedding_kind,
                                    "qpu_access_time_per_qmi_us": estimated_qpu_access_time,
                                    "qpu_access_time_total_us": estimated_total_time,
                                }
                            )
                            print(
                                "estimate "
                                f"L={chain_length}, h={name}, beta={BETA_1:.3f}, "
                                f"at={ANNEAL_TIME}, ap={ANNEAL_PARAM:.3f}, "
                                f"samples={num_samples_to_use}, reads={NUM_READS}, "
                                f"qubits={embedded_qubits} ({embedding_kind}), "
                                f"per QMI={format_time_us(estimated_qpu_access_time)}, "
                                f"total={format_time_us(estimated_total_time)}"
                            )
                            del gibbs_samples, node_order
                            gc.collect()
                            continue
                        print(f"samples for chain len {chain_length}, anneal time {ANNEAL_TIME:.2f} micro s and anneal param {ANNEAL_PARAM:.2f}, beta_1 {BETA_1:.2f}")
                        raw_data = pd.DataFrame(
                            columns=["sample", "energy", "num_occurrences", "init_state"]
                        )
                        for i in tqdm(range(num_samples_to_use), desc="samples: "):
                            gibbs_sample = gibbs_samples[i]
                            initial_state = convert_sample_to_initial_state(gibbs_sample, node_order)
                            energy_bqm = compute_energy_with_bqm(initial_state, h, J)
                            if diagnostics_df is not None and i < len(diagnostics_df):
                                logged_energy = diagnostics_df.iloc[i]["energy"]
                                energy_diff = abs(energy_bqm - logged_energy)
                                if energy_diff > 5e-5:
                                    raise ValueError(f"Sample {i}: BQM energy = {energy_bqm:.6f}, Logged energy = {logged_energy:.6f}, Difference = {energy_diff:.2e} exceeds threshold.")
                            init_state = np.array(list(initial_state.values()))
                            E_init = energy(init_state, h_vect, J_vect) / chain_length
                            try:
                                sampleset = sampler.sample_ising(
                                    h=h,
                                    J=J,
                                    initial_state=initial_state,
                                    anneal_schedule=anneal_schedule,
                                    num_reads=NUM_READS,
                                    auto_scale=False,
                                    reinitialize_state=True,
                                )
                                df = sampleset.to_pandas_dataframe(sample_column=True)
                                df["init_state"] = [initial_state for _ in range(len(df))]  # type: ignore[assignment]
                                for col in ["sample", "energy", "num_occurrences", "init_state"]:
                                    df[col] = df[col].apply(to_builtin)
                                raw_data = pd.concat([raw_data, df], ignore_index=True)
                                write_raw_data_npz_from_dataframe(raw_data, results_path)
                                del df, sampleset
                                gc.collect()
                            except Exception as e:
                                print(f"An error occurred: {e}")
                                continue
                        del gibbs_samples, node_order
                        gc.collect()

    if ESTIMATE_DWAVE_TIME_ONLY:
        print("\nD-Wave QPU access-time estimate summary")
        print("=" * 80)
        print(f"Estimated parameter combinations: {len(estimate_rows)}")
        print(f"Total planned QMI submissions: {total_planned_qmis:,}")
        print(f"Total planned reads: {total_planned_reads:,}")
        print(
            "Total estimated qpu_access_time: "
            f"{format_time_us(total_estimated_qpu_access_time)} "
            f"({total_estimated_qpu_access_time / 60_000_000:,.3f} min)"
        )
        print("No QPU jobs were submitted and no raw data files were written.")
