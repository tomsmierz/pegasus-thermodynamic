import dwave_networkx as dnx  # type: ignore
import networkx as nx
import argparse
import itertools
import numpy as np
import os
import pickle

from typing import Tuple, Union, Optional, List, Callable, Dict, Any
from dwave.system import DWaveSampler
from tqdm import tqdm
from math import inf
from dataclasses import dataclass

rng = np.random.default_rng()
path = os.getcwd()


def reverse_zephyr_sublattice_mapping(mapping: Callable, source: nx.Graph) -> Callable:
    node_dict = {node: mapping(node) for node in source.nodes}
    reversed_dict = {value: key for key, value in node_dict.items()}

    def func(node: int) -> Tuple:
        return reversed_dict[node]

    return func


def find_map(source: nx.Graph, sampler: DWaveSampler) -> Tuple:
    target = sampler.to_networkx_graph()
    perfect = False
    mappings = [mapp for mapp in dnx.zephyr_sublattice_mappings(source, target)]
    mapping = None
    best_missing_nodes = None
    best_missing_edges = None

    min_num_of_missing_edges = inf
    min_num_of_missing_nodes = inf
    best_imperfect_mapping = None

    for i in tqdm(range(len(mappings)), desc="Searching for a perfect mapping"):
        node_dict = {node: mappings[i](node) for node in source.nodes()}
        edge_dict = {(v, w): (node_dict[v], node_dict[w]) for v, w in source.edges()}

        if all(node in target.nodes() for node in node_dict.values()) and all(
            edge in target.edges() for edge in edge_dict.values()
        ):
            mapping = mappings[i]
            print("\nPerfect map found")
            perfect = True
            break
        else:
            mapped_source_nodes_set = set(node_dict.values())
            mapped_source_edges_set = {frozenset(edge) for edge in edge_dict.values()}

            real_nodes_set = set(sampler.nodelist)
            real_edges_set = {frozenset(edge) for edge in sampler.edgelist}

            missing_nodes = list(mapped_source_nodes_set - real_nodes_set)
            missing_edges = list(mapped_source_edges_set - real_edges_set)
            num_of_missing_nodes = len(missing_nodes)
            num_of_missing_edges = len(missing_edges)

            if (
                num_of_missing_nodes <= min_num_of_missing_nodes
                and num_of_missing_edges <= min_num_of_missing_edges
            ):
                min_num_of_missing_nodes = num_of_missing_nodes
                min_num_of_missing_edges = num_of_missing_edges
                best_missing_nodes = missing_nodes
                best_missing_edges = missing_edges
                best_imperfect_mapping = mappings[i]

    if mapping is None:
        mapping = best_imperfect_mapping
        print(
            f"\nNo perfect map found. Returning imperfect map with {min_num_of_missing_nodes} missing nodes"
            f" and {min_num_of_missing_edges} missing edges"
        )

    if mapping is None:
        raise RuntimeError(
            "No map found. Possible problem with the source or the target graph"
        )

    return mapping, perfect, best_missing_nodes, best_missing_edges


def find_all_mappings(source, sampler):
    target = sampler.to_networkx_graph()
    perfect = False
    mappings = [mapp for mapp in dnx.zephyr_sublattice_mappings(source, target)]
    mapping = None
    best_missing_nodes = None
    best_missing_edges = None

    min_num_of_missing_edges = inf
    min_num_of_missing_nodes = inf
    best_imperfect_mapping = None

    perfect_mappings = []
    imperfect_mappings = []

    for i in tqdm(range(len(mappings)), desc="Searching for all mappings"):
        node_dict = {node: mappings[i](node) for node in source.nodes()}
        edge_dict = {(v, w): (node_dict[v], node_dict[w]) for v, w in source.edges()}

        if all(node in target.nodes() for node in node_dict.values()) and all(
                edge in target.edges() for edge in edge_dict.values()
        ):
            perfect_mappings.append(mappings[i])

        else:
            mapped_source_nodes_set = set(node_dict.values())
            mapped_source_edges_set = {frozenset(edge) for edge in edge_dict.values()}

            real_nodes_set = set(sampler.nodelist)
            real_edges_set = {frozenset(edge) for edge in sampler.edgelist}

            missing_nodes = list(mapped_source_nodes_set - real_nodes_set)
            missing_edges = list(mapped_source_edges_set - real_edges_set)
            num_of_missing_nodes = len(missing_nodes)
            num_of_missing_edges = len(missing_edges)

            imperfect_mappings.append((mappings[i], num_of_missing_nodes, num_of_missing_edges))

    return perfect_mappings, sort_imperfect_mappings_by_missing_nodes(imperfect_mappings)


def sort_imperfect_mappings_by_missing_nodes(
    imperfect_mappings: List[Tuple[Callable, int, int]]
) -> List[Tuple[Callable, int, int]]:
    """
    Sort imperfect mappings by the number of missing nodes, using the number of missing edges
    as a deterministic tie-breaker.

    Args:
        imperfect_mappings: A list of tuples containing the mapping callable, the number of
            missing nodes, and the number of missing edges.

    Returns:
        A new list of imperfect mappings sorted by increasing number of missing nodes and then
        by missing edges.
    """

    return sorted(imperfect_mappings, key=lambda item: (item[1], item[2]))


def validate_instance_on_qpu(
    bias: Dict[Any, float], 
    couplings: Dict[Tuple[Any, Any], float], 
    sampler: DWaveSampler,
    num_reads: int = 100
) -> Dict[str, Any]:
    """
    Validate a generated instance by running it on the specified QPU.
    
    Args:
        bias: Dictionary of node biases
        couplings: Dictionary of edge couplings
        sampler: DWave sampler to use for validation
        num_reads: Number of reads to perform for validation
        
    Returns:
        Dictionary containing validation results including energy statistics,
        timing info, and success indicators
    """
    try:
        # Filter bias and couplings to only include nodes/edges that exist on the QPU
        qpu_nodes = set(sampler.nodelist)
        qpu_edges = set()
        for edge in sampler.edgelist:
            qpu_edges.add((edge[0], edge[1]))
            qpu_edges.add((edge[1], edge[0]))  # Add both directions
        
        # Filter bias to only include existing nodes
        filtered_bias = {node: value for node, value in bias.items() if node in qpu_nodes}
        
        # Filter couplings to only include existing edges
        filtered_couplings = {}
        for edge, value in couplings.items():
            edge_tuple = (edge[0], edge[1]) if isinstance(edge, (list, tuple)) else edge
            if edge_tuple in qpu_edges:
                filtered_couplings[edge_tuple] = value
        
        # Verify we have a non-empty problem
        if not filtered_bias and not filtered_couplings:
            raise ValueError("No valid nodes or edges found on QPU after filtering")
        
        print(f"Filtered problem: {len(filtered_bias)} nodes, {len(filtered_couplings)} edges")
        
        # Run the problem on the QPU
        response = sampler.sample_ising(
            filtered_bias, 
            filtered_couplings, 
            num_reads=num_reads
        )
        
        # Extract validation metrics
        energies = response.data_vectors['energy']
        timing = response.info.get('timing', {})
        
        validation_results = {
            'success': True,
            'num_samples': len(response),
            'min_energy': float(min(energies)),
            'max_energy': float(max(energies)),
            'mean_energy': float(np.mean(energies)),
            'std_energy': float(np.std(energies)),
            'timing': timing,
            'embedding_context': response.info.get('embedding_context', {}),
            'chain_break_fraction': response.info.get('chain_break_fraction', None),
            'filtered_nodes': len(filtered_bias),
            'filtered_edges': len(filtered_couplings),
            'original_nodes': len(bias),
            'original_edges': len(couplings)
        }
        
        print(f"✓ Instance validation successful - Energy range: [{validation_results['min_energy']:.3f}, {validation_results['max_energy']:.3f}]")
        
    except Exception as e:
        validation_results = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(f"✗ Instance validation failed: {e}")
    
    return validation_results


def generate_zephyr_instances(
    number: int,
    size: int,
    output_path: str,
    output_types: List[str] = ["DWave"],
    category: str = "RAU",
    device: Optional[str] = None,
    name: Optional[str] = None,
    all_mappings: Optional[bool] = False,
    validate_on_qpu: Optional[bool] = False,
    validation_reads: int = 1
) -> None:

    source = dnx.zephyr_graph(size, coordinates=True)
    username = name is not None

    if device is not None:

        if size > 12:
            raise AssertionError("Maximum size for working device is 12")

        sampler = DWaveSampler(solver=device, token=os.environ["DWAVE_API_TOKEN"])

        if not all_mappings:
            mapping, perfect_mapping, missing_nodes, missing_edges = find_map(
                source, sampler
            )
            if perfect_mapping:
                graph = source
            else:
                reverse_mapping = reverse_zephyr_sublattice_mapping(mapping, source)
                graph = source
                for node in missing_nodes:
                    graph.remove_node(reverse_mapping(node))

                for edge in missing_edges:
                    edge = tuple(edge)
                    edge = (reverse_mapping(edge[0]), reverse_mapping(edge[1]))
                    reversed_edge = (edge[1], edge[0])
                    if edge in graph.edges():
                        graph.remove_edge(edge[0], edge[1])

                    if reversed_edge in graph.edges():
                        graph.remove_edge(reversed_edge[0], reversed_edge[1])
        else:
            perfect_mappings, imperfect_mappings = find_all_mappings(source, sampler)
            imperfect_mappings = [mapping for mapping in imperfect_mappings if mapping[1] <= 30]
            print(f"Found {len(perfect_mappings)} perfect mappings")
            if not perfect_mappings:
                # raise ValueError(f"No perfect mapping found. Found {len(imperfect_mappings)} imperfect mappings")
                print(f"No perfect mapping found. Found {len(imperfect_mappings)} imperfect mappings")
                mapping_all = imperfect_mappings[rng.integers(0, len(imperfect_mappings))]
                mapping = mapping_all[0]
                print(f"Using imperfect mapping with {mapping_all[1]} missing nodes and "
                      f"{mapping_all[2]} missing edges")
                
                # For imperfect mappings, we need to remove missing nodes and edges from the graph
                # to ensure the instance is compatible with the QPU
                graph = source.copy()  # Make a copy to avoid modifying the original
                
                # Find missing nodes and edges for this specific mapping
                node_dict = {node: mapping(node) for node in source.nodes()}
                edge_dict = {(v, w): (node_dict[v], node_dict[w]) for v, w in source.edges()}
                
                mapped_source_nodes_set = set(node_dict.values())
                mapped_source_edges_set = {frozenset(edge) for edge in edge_dict.values()}
                
                real_nodes_set = set(sampler.nodelist)
                real_edges_set = {frozenset(edge) for edge in sampler.edgelist}
                
                missing_nodes = list(mapped_source_nodes_set - real_nodes_set)
                missing_edges = list(mapped_source_edges_set - real_edges_set)
                
                # Remove missing nodes and their edges from the graph
                reverse_mapping = reverse_zephyr_sublattice_mapping(mapping, source)
                for node in missing_nodes:
                    reverse_node = reverse_mapping(node)
                    if reverse_node in graph.nodes():
                        graph.remove_node(reverse_node)
                
                # Remove missing edges
                for edge in missing_edges:
                    edge = tuple(edge)
                    edge = (reverse_mapping(edge[0]), reverse_mapping(edge[1]))
                    reversed_edge = (edge[1], edge[0])
                    if edge in graph.edges():
                        graph.remove_edge(edge[0], edge[1])
                    if reversed_edge in graph.edges():
                        graph.remove_edge(reversed_edge[0], reversed_edge[1])
                        
            else:
              mapping = perfect_mappings[rng.integers(0, len(perfect_mappings))]
              graph = source

    else:
        graph = source

    for i in tqdm(
        range(number),
        desc=f"generating zephyr instances size = {size}, category={category}: ",
    ):
        if category == "AC3":
            bias = {node: rng.uniform(-1 / 9, 1 / 9) for node in graph.nodes()}
            couplings = {
                edge: rng.uniform(-1 / 3, 1 / 3)
                if edge[0][1:3] == edge[1][1:3]
                else rng.uniform(-1, 1)
                for edge in graph.edges()
            }
        elif category in {"CBFM", "CBFM-P"}:
            bias = {node: rng.choice([-1, 0], p=[0.85, 0.15]) for node in graph.nodes()}
            couplings = {
                edge: rng.choice([-1, 0, 1], p=[0.1, 0.35, 0.55])
                for edge in graph.edges()
            }
        elif category == "RAU":
            bias = {node: rng.uniform(-1, 1) for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        elif category == "RCO":
            bias = {node: 0 for node in graph.nodes()}
            couplings = {edge: rng.uniform(-1, 1) for edge in graph.edges()}
        elif category == "CON":
            bias = {node: 0 for node in graph.nodes()}
            couplings = {edge: -1 for edge in graph.edges()}
        else:
            raise ValueError(
                f'Category {category} is not a valid choice. It should be "RAU", "RCO", "CON", "AC3" or "CBFM-P"'
            )

        file_category = "CBFM" if category == "CBFM-P" else category
        instance_name = f"{name}{i + 1}" if username else f"Z{size}_{file_category}_{i + 1}"
        for output_type in output_types:
            if output_type == "DWave":
                if device is not None:
                    couplings_dv = {
                        (mapping(edge[0]), mapping(edge[1])): value
                        for edge, value in couplings.items()
                    }
                    bias_dv = {mapping(node): value for node, value in bias.items()}
                    data = [bias_dv, couplings_dv]
                else:
                    data = [bias, couplings]

                output_name = f"{instance_name}.pkl"
                with open(os.path.join(output_path, output_name), "wb") as f:
                    pickle.dump(data, f)
                
                # Validate instance on QPU if requested and device is available
                if validate_on_qpu and device is not None:
                    print(f"Validating instance {instance_name} on QPU {device}...")
                    # Use the properly mapped bias/couplings for device validation
                    validation_bias = bias_dv if device is not None else bias
                    validation_couplings = couplings_dv if device is not None else couplings
                    validation_results = validate_instance_on_qpu(
                        validation_bias, 
                        validation_couplings, 
                        sampler, 
                        validation_reads
                    )
                    
                    # Save validation results alongside the instance for later analysis
                    validation_name = f"{instance_name}_validation.pkl"
                    with open(os.path.join(output_path, validation_name), "wb") as f:
                        pickle.dump(validation_results, f)

            else:
                raise ValueError(
                    f'{output_type} is not valid output type. It should be "DWave", '
                    f'or "MatrixMarket"'
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-S",
        "--size",
        type=int,
        default=4,
        help="Size of the pegasus graph. Minimum 2. Default is 4 (P4).",
    )
    parser.add_argument(
        "-N",
        "--number",
        type=int,
        default=1,
        help="Number of instances to be generated. Default is 1.",
    )
    parser.add_argument(
        "-C",
        "--category",
        type=str,
        default="CON",
        choices=["CON", "RAU", "RCO", "AC3", "CBFM", "CBFM-P"],
        help="Category of generated instances. CON - constant coupling, RAU - random uniform, RCO - random couplings only, "
        "AC3 - anti-cluster",
    )
    parser.add_argument(
        "-P",
        "--path",
        type=str,
        default=path,
        help="path to folder where generated instances will be located. "
        "Default is working directory",
    )
    parser.add_argument(
        "-T",
        "--types",
        type=str,
        default=["DWave"],
        choices=["DWave", "MatrixMarket"],
        nargs="*",
    )
    parser.add_argument(
        "-D",
        "--device",
        default=None,
        help="Map instance info physical D-Wave's device. Input None for no Mapping",
    )
    parser.add_argument(
        "-V",
        "--validate",
        default=False,
        action="store_true",
        help="Validate generated instances on the specified QPU device",
    )
    parser.add_argument(
        "-R",
        "--validation-reads",
        type=int,
        default=1,
        help="Number of reads to use for QPU validation (default: 100)",
    )

    args = parser.parse_args()

    if args.size and args.size < 2:
        parser.error("Minimum size of pegasus instance is 2")
    
    if args.validate and args.device is None:
        parser.error("Validation requires a device to be specified with -D/--device")

    generate_zephyr_instances(
        args.number,
        args.size,
        args.path,
        args.types,
        args.category,
        device=args.device,
        all_mappings=True,
        validate_on_qpu=args.validate,
        validation_reads=args.validation_reads,
    )
