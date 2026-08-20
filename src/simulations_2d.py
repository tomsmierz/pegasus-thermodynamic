import os
import pickle
import gc  # for garbage collection to prevent memory leaks

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

from utils import energy, vectorize_2d
from dwave.system.samplers import DWaveSampler
from minorminer import find_embedding
from tqdm import tqdm
from collections import namedtuple
import dimod
import glob  # for file pattern matching
from raw_data_io import get_raw_data_row_count, write_raw_data_npz_from_dataframe

import warnings
# suppress FutureWarning from pandas
warnings.simplefilter("ignore", FutureWarning)

rng = np.random.default_rng()
Instance = namedtuple("Instance", ["h", "J", "name"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWD = os.getcwd()


def load_gibbs_samples(gibbs_results_dir, instance_basename, beta):
    """
    Load pre-generated Gibbs samples from gibbs_sampling_runner output.
    
    Parameters:
    -----------
    gibbs_results_dir : str
        Root directory containing Gibbs sampling results
    instance_basename : str
        Instance name (e.g., 'P6_CON_1')
    beta : float
        Inverse temperature
    
    Returns:
    --------
    samples : np.ndarray
        Array of shape (num_samples, num_spins) containing Gibbs samples
    node_order : list
        Order of nodes corresponding to samples columns
    diagnostics_df : pd.DataFrame
        Diagnostics data including logged energies for each sample
    """
    # Build path to samples: gibbs_results_dir/instance_basename/BETA_X.XXX/gibbs_samples.npz
    beta_subdir = f"BETA_{beta:.3f}"
    samples_path = os.path.join(gibbs_results_dir, instance_basename, beta_subdir, 'gibbs_samples.npz')
    diagnostics_path = os.path.join(gibbs_results_dir, instance_basename, beta_subdir, 'sampling_diagnostics.csv')
    
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Gibbs samples not found at: {samples_path}")
    
    # Load samples
    data = np.load(samples_path, allow_pickle=True)
    samples = data['samples']
    node_order = data['node_order']
    
    # Load diagnostics (contains logged energies)
    diagnostics_df = None
    if os.path.exists(diagnostics_path):
        diagnostics_df = pd.read_csv(diagnostics_path)
        print(f"Loaded {len(samples)} Gibbs samples and diagnostics from {samples_path}")
    else:
        print(f"Loaded {len(samples)} Gibbs samples from {samples_path} (no diagnostics file found)")
    
    return samples, node_order, diagnostics_df


def convert_sample_to_initial_state(sample, node_order):
    """
    Convert a Gibbs sample array to an initial_state dictionary.
    
    Parameters:
    -----------
    sample : np.ndarray
        1D array of spin values
    node_order : np.ndarray or list
        Order of nodes corresponding to sample indices
    
    Returns:
    --------
    initial_state : dict
        Dictionary mapping node indices to spin values
    """
    return {int(node): int(spin) for node, spin in zip(node_order, sample)}


def compute_energy_with_bqm(initial_state, h, J):
    """
    Compute energy of a state using dimod BinaryQuadraticModel.
    
    Parameters:
    -----------
    initial_state : dict
        Dictionary mapping node indices to spin values
    h : dict
        Linear biases (fields)
    J : dict
        Quadratic biases (couplings)
    
    Returns:
    --------
    energy : float
        Energy of the state
    """
    bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
    return bqm.energy(initial_state)


def to_builtin(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {to_builtin(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    return obj


if __name__ == "__main__":

    # This part may need changes depending on how you comunicate with your machine
    qpu_sampler = DWaveSampler(
        solver="Advantage_system6.4",
        # solver="Advantage2_system1.6",
        token=os.environ["DWAVE_API_TOKEN"],
    )  # specify device used
    target = qpu_sampler.to_networkx_graph()

    # Set beta schedule per instance type (like anneal_params)
    BETA_1_VALUES = {
        "CON": [round(s, 3) for s in list(np.arange(0.002, 0.02, 0.002))],
        "RAU": [round(s, 3) for s in list(np.arange(0.02, 1.02, 0.02))],
        "CBFM": [round(s, 3) for s in list(np.arange(0.02, 1.02, 0.02))],
    }
    
    # Configuration
    NUM_SAMPLES = 50  # Number of Gibbs samples to use from pre-generated set
    anneal_time = [100] #, 200, 500, 1000]
    NUM_READS = 50
    initial_value = 1.0
    anneal_params = {
        "CON": [round(s, 3) for s in np.arange(0.7, 0.9, 0.01)],
        # "RAU": [round(s, 3) for s in np.arange(0.1, 0.9, 0.05)],
        # "CBFM": [round(s, 3) for s in np.arange(0.7, 0.9, 0.01)],
    }
    
    SCALING = [6]
    # itype = ["CON", "RAU"]
    itype = ["CON"]
    instance_graph = "pegasus"
    
    # Path to pre-generated Gibbs samples
    GIBBS_RESULTS_DIR = os.path.join(ROOT, "data", "raw_data", "gibbs_samples")
    
    print("Starting processing of instances with pre-generated Gibbs samples...")
    print("anneal_time:", anneal_time)
    print(f"Gibbs samples directory: {GIBBS_RESULTS_DIR}")
    
    for chain_length in SCALING:
        for ITYPE in itype:
            print(f"beta_1_values for {ITYPE}:", BETA_1_VALUES.get(ITYPE))
            for BETA_1 in BETA_1_VALUES[ITYPE]:
                # Find all instances of the given type and chain length in pegasus_native
                pattern = os.path.join(
                    ROOT, "data", "instances", f"sub{instance_graph}_native", f"P{chain_length}_{ITYPE}_*.pkl"
                )
                instance_files = sorted(glob.glob(pattern))
                instance_files = instance_files[:3]
                
                if not instance_files:
                    print(f"No instances found for P{chain_length} type {ITYPE} in sub{instance_graph}_native directory")
                    continue
                
                print(f"Found {len(instance_files)} instances for P{chain_length} type {ITYPE}")
                
                # Process each instance file
                for inst_idx, instance_file in enumerate(instance_files):
                    instance_basename = os.path.basename(instance_file).replace('.pkl', '')
                    
                    output_path = os.path.join(
                        ROOT, "data", "raw_data", f"more_sweeps_phase_diagram_P{chain_length}_{ITYPE}_advantage6.4"
                    )
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
                    print(f"Processing instance {inst_idx + 1}/{len(instance_files)}: {instance_basename} with BETA_1={BETA_1}")
                    
                    # Load instance from pegasus_native
                    with open(instance_file, "rb") as f:
                        inst_data = pickle.load(f)
                    
                    # Convert from list format [h, J] to Instance namedtuple
                    if isinstance(inst_data, list) and len(inst_data) == 2:
                        h = inst_data[0]
                        J = inst_data[1]
                        inst = Instance(h=h, J=J, name=f"{instance_basename}_beta_{BETA_1:.1f}")
                    else:
                        print(f"Unexpected instance format in {instance_file}")
                        continue
                    
                    h = inst.h
                    J = inst.J
                    h_vect, J_vect, h_new, J_new, _ = vectorize_2d(h, J)

                    # Load pre-generated Gibbs samples for this instance and beta
                    try:
                        gibbs_samples, node_order, diagnostics_df = load_gibbs_samples(
                            GIBBS_RESULTS_DIR, instance_basename, BETA_1
                        )
                    except FileNotFoundError as e:
                        print(f"Skipping instance {instance_basename}: {e}")
                        continue
                    
                    # Determine how many samples to use
                    num_available_samples = len(gibbs_samples)
                    num_samples_to_use = min(NUM_SAMPLES, num_available_samples)
                    
                    if num_available_samples < NUM_SAMPLES:
                        print(f"Warning: Only {num_available_samples} samples available, using all of them")
                    
                    # Iterate over annealing parameters
                    for ANNEAL_TIME in anneal_time:
                        for ANNEAL_PARAM in anneal_params[ITYPE]:
                            
                            results_path = os.path.join(
                                output_path,
                                f"raw_data_{instance_basename}_beta_{BETA_1:.3f}_at_{ANNEAL_TIME}_ap_{ANNEAL_PARAM:.3f}.npz",
                            )
                            
                            # Check if results already exist
                            if os.path.exists(results_path):
                                row_count = get_raw_data_row_count(results_path)
                                if row_count > num_samples_to_use * NUM_READS * 0.9:
                                    print(
                                        f"Results already exist for {instance_basename}, beta={BETA_1}, at={ANNEAL_TIME}, ap={ANNEAL_PARAM} with sufficient data"
                                    )
                                    continue
                            
                            raw_data = pd.DataFrame(
                                columns=["sample", "energy", "num_occurrences", "init_state"]
                            )
                            
                            # Process each Gibbs sample
                            for i in tqdm(
                                range(num_samples_to_use),
                                desc=f"samples for {instance_basename} beta {BETA_1:.1f} anneal time {ANNEAL_TIME:.2f} micro s and anneal param {ANNEAL_PARAM:.2f}",
                            ):
                                # Get the i-th pre-generated Gibbs sample
                                gibbs_sample = gibbs_samples[i]
                                
                                # Convert to initial_state dictionary
                                initial_state = convert_sample_to_initial_state(gibbs_sample, node_order)
                                
                                # Compute energy using dimod BQM
                                energy_bqm = compute_energy_with_bqm(initial_state, h, J)
                                
                                # Get logged energy from diagnostics if available
                                if diagnostics_df is not None and i < len(diagnostics_df):
                                    logged_energy = diagnostics_df.iloc[i]['energy']
                                    energy_diff = abs(energy_bqm - logged_energy)
                                    
                                    # Print comparison for first few samples or if there's a discrepancy
                                    if energy_diff > 5e-5:
                                        print(f"\nSample {i}: BQM energy = {energy_bqm:.6f}, "
                                              f"Logged energy = {logged_energy:.6f}, "
                                              f"Difference = {energy_diff:.2e}")
                                
                                # Compute energy of initial state using original method
                                init_state = np.array(list(initial_state.values()))
                                E_init = energy(init_state, h_vect, J_vect) / len(h)

                                # Set up annealing schedule
                                anneal_schedule = [
                                    [0.0, initial_value],
                                    [ANNEAL_TIME / 2, ANNEAL_PARAM],
                                    [ANNEAL_TIME, initial_value],
                                ]
                                
                                try:
                                    sampler = qpu_sampler
                                    sampleset = sampler.sample_ising(
                                        h=h,
                                        J=J,
                                        initial_state=initial_state,
                                        anneal_schedule=anneal_schedule,
                                        num_reads=NUM_READS,
                                        auto_scale=True,
                                        reinitialize_state=True,
                                    )
            
                                    df = sampleset.to_pandas_dataframe(sample_column=True)
                                    df["init_state"] = [initial_state for _ in range(len(df))]

                                    # Ensure stored values are plain Python types (no numpy scalars)
                                    for col in ["sample", "energy", "num_occurrences", "init_state"]:
                                        df[col] = df[col].apply(to_builtin)
                                    raw_data = pd.concat([raw_data, df], ignore_index=True)

                                    write_raw_data_npz_from_dataframe(raw_data, results_path)
                                    
                                    # Memory management: explicitly delete large objects
                                    del df, sampleset
                                    gc.collect()
                                    
                                except Exception as e:
                                    print(f"Error during QPU sampling: {e}")
                                    continue
                    
                    # Clean up memory after processing each instance
                    del h_vect, J_vect, h_new, J_new, h, J, inst, gibbs_samples, node_order
                    gc.collect()
                    print(f"Completed processing {instance_basename} with BETA_1={BETA_1}")
                    
    print("All processing completed!")
