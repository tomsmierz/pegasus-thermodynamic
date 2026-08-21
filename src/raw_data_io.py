from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


def _sort_key(value: Any) -> Tuple[int, Any]:
    try:
        return (0, int(value))
    except Exception:
        pass
    try:
        return (1, float(value))
    except Exception:
        pass
    return (2, str(value))


def _parse_mapping(cell: Any) -> Mapping[Any, Any]:
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, str):
        return ast.literal_eval(cell)
    raise ValueError(f"Expected mapping-compatible cell, got {type(cell).__name__}")


def _normalize_num_occurrences_column(df: pd.DataFrame) -> str | None:
    if "num_occurrences" in df.columns:
        return "num_occurrences"
    if "num_occurences" in df.columns:
        return "num_occurences"
    return None


def _build_columnar_arrays_from_dataframe(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    if "sample" not in df.columns:
        raise ValueError("Missing required column: sample")

    sample_mappings = [_parse_mapping(cell) for cell in df["sample"].tolist()]

    if len(sample_mappings) == 0:
        return {
            "coordinates": np.array([], dtype=np.int64),
            "sample": np.empty((0, 0), dtype=np.int8),
            "init_state": np.empty((0, 0), dtype=np.int8),
            "num_occurrences": np.array([], dtype=np.int64),
        }

    coordinates = sorted(sample_mappings[0].keys(), key=_sort_key)
    coordinates_array = np.array(coordinates, dtype=object)

    sample_rows: List[List[int]] = []
    for mapping in sample_mappings:
        if set(mapping.keys()) != set(coordinates):
            raise ValueError("Inconsistent sample keys encountered across rows")
        sample_rows.append([int(mapping[k]) for k in coordinates])
    sample_array = np.asarray(sample_rows, dtype=np.int8)

    init_rows: List[List[int]] = []
    if "init_state" in df.columns:
        init_mappings = [_parse_mapping(cell) for cell in df["init_state"].tolist()]
        for mapping in init_mappings:
            if set(mapping.keys()) != set(coordinates):
                raise ValueError("Inconsistent init_state keys encountered across rows")
            init_rows.append([int(mapping[k]) for k in coordinates])
    else:
        init_rows = [row[:] for row in sample_rows]
    init_array = np.asarray(init_rows, dtype=np.int8)

    arrays: Dict[str, np.ndarray] = {
        "coordinates": coordinates_array,
        "sample": sample_array,
        "init_state": init_array,
    }

    num_occ_col = _normalize_num_occurrences_column(df)
    if num_occ_col is not None:
        arrays["num_occurrences"] = np.asarray(df[num_occ_col].to_numpy())
    else:
        arrays["num_occurrences"] = np.ones(sample_array.shape[0], dtype=np.int64)

    for col in df.columns:
        if col.startswith("Unnamed:"):
            continue
        if col in {"sample", "init_state", "num_occurrences", "num_occurences"}:
            continue
        arrays[col] = np.asarray(df[col].to_numpy())

    return arrays


def write_raw_data_npz_from_dataframe(df: pd.DataFrame, output_npz: str) -> None:
    arrays = _build_columnar_arrays_from_dataframe(df)
    tmp_output = output_npz + ".tmp.npz"
    np.savez_compressed(tmp_output, **arrays)  # type: ignore[arg-type]
    os.replace(tmp_output, output_npz)


def write_raw_data_npz_from_csv(csv_path: str, output_npz: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    arrays = _build_columnar_arrays_from_dataframe(df)
    tmp_output = output_npz + ".tmp.npz"
    np.savez_compressed(tmp_output, **arrays)  # type: ignore[arg-type]
    os.replace(tmp_output, output_npz)
    return arrays


def load_raw_data_file(path: str) -> Dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)

            coordinates = data["coordinates"] if "coordinates" in keys else data.get("node_order")
            if coordinates is None:
                raise ValueError(f"Missing coordinates/node_order in {path}")

            sample = data["sample"] if "sample" in keys else data.get("samples")
            if sample is None:
                raise ValueError(f"Missing sample/samples in {path}")

            init_state = data["init_state"] if "init_state" in keys else data.get("init_states")
            if init_state is None:
                init_state = np.asarray(sample, dtype=np.int8)

            energy = data["energy"] if "energy" in keys else data.get("sample_energies")
            if energy is None:
                raise ValueError(f"Missing energy/sample_energies in {path}")

            num_occurrences = data["num_occurrences"] if "num_occurrences" in keys else data.get("num_occurences")
            if num_occurrences is None:
                num_occurrences = np.ones(sample.shape[0], dtype=np.int64)

            result: Dict[str, np.ndarray] = {
                "coordinates": np.asarray(coordinates),
                "sample": np.asarray(sample),
                "init_state": np.asarray(init_state),
                "energy": np.asarray(energy),
                "num_occurrences": np.asarray(num_occurrences),
            }

            reserved = {
                "coordinates",
                "node_order",
                "sample",
                "samples",
                "init_state",
                "init_states",
                "energy",
                "sample_energies",
                "num_occurrences",
                "num_occurences",
            }
            for key in data.files:
                if key not in reserved:
                    result[key] = np.asarray(data[key])
            return result

    if ext == ".csv":
        df = pd.read_csv(path)
        arrays = _build_columnar_arrays_from_dataframe(df)
        if "energy" not in arrays:
            raise ValueError(f"Missing energy column in {path}")
        return arrays

    raise ValueError(f"Unsupported raw data extension: {path}")


def get_raw_data_row_count(path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = load_raw_data_file(path)
        return int(data["sample"].shape[0])
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as file:
            line_count = sum(1 for _ in file)
        return max(0, line_count - 1)
    return 0