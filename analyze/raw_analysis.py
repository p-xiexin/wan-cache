"""Standalone offline reader for collect_raw_trajectories.py (no Wan import)."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(value: str | Path) -> Path:
    """Expand ~ and environment variables; resolve relative to these scripts."""
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if not expanded.strip() or re.search(r"\$(?:\w+|\{[^}]+\})", expanded):
        raise ValueError(f"Empty path or unresolved environment variable: {value!r}")
    path = Path(expanded)
    return (path if path.is_absolute() else SCRIPT_DIR / path).resolve()


def find_trajectories(root: str | Path, limit: int = 0) -> list[Path]:
    root = resolve_path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw trajectory directory does not exist: {root}")
    if limit < 0:
        raise ValueError("MAX_TRAJECTORIES must be >= 0 (0 means all)")
    if (root / "metadata.json").is_file():
        candidates = [root]
    else:
        candidates = sorted(set(root.glob("trajectory_*")) | set(root.glob("trajectories/trajectory_*")))
    complete = []
    for path in candidates:
        if not path.is_dir():
            continue
        if not (path / "_SUCCESS").is_file():
            print(f"Skipping incomplete trajectory: {path}", flush=True)
            continue
        complete.append(path)
    if not complete:
        raise ValueError(f"No completed trajectories with _SUCCESS found under {root}")
    if len({p.name for p in complete}) != len(complete):
        raise ValueError("Duplicate trajectory names in flat/nested layouts; pass one layout's directory")
    return complete[:limit] if limit else complete


@dataclass
class Series:
    branch: str
    item: int
    values: np.ndarray  # [denoising step, sampled latent position, channel]
    steps: np.ndarray
    timesteps: np.ndarray
    token_indices: np.ndarray
    source_shape: tuple[int, ...]
    source_dtypes: tuple[str, str]


def load_trajectory(
    path: Path, target: str, max_tokens: int, seed: int,
) -> list[Series]:
    """Read one shard at a time and keep only fixed sampled latent positions.

    Input/output must be tuples/lists of [C,F,H,W] tensors, as saved by the
    collector. These positions are not transformer hidden tokens.
    """
    if target not in {"x", "v", "residual"}:
        raise ValueError("TARGET must be x, v, or residual")
    if max_tokens < 0:
        raise ValueError("MAX_TOKENS must be >= 0 (0 means all)")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    shards = metadata.get("shards")
    if not isinstance(shards, list) or not shards or not all(isinstance(s, str) for s in shards):
        raise ValueError(f"Missing/invalid metadata.shards: {path}")
    if len(set(shards)) != len(shards) or any(Path(s).name != s for s in shards):
        raise ValueError(f"Duplicate or non-local shard names: {path}")
    if set(shards) != {p.name for p in path.glob("shard_*.pt")}:
        raise ValueError(f"Shard files do not match metadata.shards: {path}")

    buffers, layouts, dtypes, indices = {}, {}, {}, {}
    expected_items = None
    for shard in shards:
        records = torch.load(path / shard, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list of raw records: {path / shard}")
        for record in records:
            branch = record["branch"]
            step, timestep = record["step_index"], float(record["timestep"])
            if branch not in {"conditional", "unconditional"}:
                raise ValueError(f"Unexpected branch {branch!r}: {path / shard}")
            if not isinstance(step, int) or step < 0 or not np.isfinite(timestep):
                raise ValueError(f"Invalid step/timestep: {path / shard}")
            xs, vs = record["model_input"], record["model_output"]
            if not isinstance(xs, (tuple, list)) or not isinstance(vs, (tuple, list)):
                raise ValueError("model_input/model_output must be lists or tuples")
            if not xs or len(xs) != len(vs):
                raise ValueError("Input/output tensor counts differ or are empty")
            if expected_items is None:
                expected_items = len(xs)
            if len(xs) != expected_items:
                raise ValueError("Tensor count changed within a trajectory")
            for item, (x, v) in enumerate(zip(xs, vs)):
                if not isinstance(x, torch.Tensor) or not isinstance(v, torch.Tensor):
                    raise ValueError("Expected torch tensors")
                shape = tuple(x.shape)
                if x.ndim != 4 or shape != tuple(v.shape) or min(shape) < 1:
                    raise ValueError(f"Expected matching [C,F,H,W], got {shape}, {tuple(v.shape)}")
                if item not in layouts:
                    layouts[item] = shape
                    n = int(np.prod(shape[1:]))
                    count = min(n, max_tokens) if max_tokens else n
                    rng = np.random.default_rng(seed + item)
                    indices[item] = np.sort(rng.choice(n, count, replace=False))
                if layouts[item] != shape:
                    raise ValueError("Tensor shape changed within a trajectory")
                key = (branch, item)
                dtype_pair = (str(x.dtype), str(v.dtype))
                if key in dtypes and dtypes[key] != dtype_pair:
                    raise ValueError("Tensor dtype changed within a trajectory")
                dtypes[key] = dtype_pair
                selected = torch.from_numpy(indices[item])
                # Convert before subtraction, so bfloat16 does not round the delta.
                def sample(tensor):
                    return tensor.reshape(shape[0], -1).index_select(1, selected).float().T.numpy()
                values = sample(x) if target == "x" else sample(v)
                if target == "residual":
                    values = values - sample(x)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite sampled values: {path}, {key}, step {step}")
                buffers.setdefault(key, []).append((step, timestep, values))
        del records

    expected_keys = {(b, i) for b in ("conditional", "unconditional") for i in range(expected_items or 0)}
    if not buffers or set(buffers) != expected_keys:
        raise ValueError(f"Missing CFG branch/tensor: {path}")
    result, reference_grid = [], None
    for (branch, item), records in sorted(buffers.items()):
        records.sort(key=lambda r: r[0])
        steps = np.array([r[0] for r in records])
        timesteps = np.array([r[1] for r in records], dtype=np.float64)
        if len(steps) < 3 or not np.array_equal(steps, np.arange(len(steps))):
            raise ValueError(f"Need >=3 consecutive steps starting at 0, without duplicates: {path}")
        dt = np.diff(timesteps)
        if not (np.all(dt < 0) or np.all(dt > 0)):
            raise ValueError(f"Model timesteps must be strictly monotonic: {path}")
        grid = (steps.tolist(), timesteps.tolist())
        if reference_grid is not None and grid != reference_grid:
            raise ValueError(f"CFG branches/tensors have different step grids: {path}")
        reference_grid = grid
        result.append(Series(branch, item, np.stack([r[2] for r in records]), steps,
                             timesteps, indices[item], layouts[item], dtypes[(branch, item)]))
    return result


def divide(numerator, denominator):
    """Undefined zero-energy ratios remain NaN rather than misleading zeroes."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 1e-20)


def cosine(a, b, axis):
    return divide(np.sum(a * b, axis=axis),
                  np.sqrt(np.sum(a * a, axis=axis) * np.sum(b * b, axis=axis)))


def write_json(path: Path, value) -> None:
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, np.generic):
            return clean(obj.item())
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                             for k, v in row.items()})


def provenance(path: Path, series: Series, target: str, seed: int) -> dict:
    return {
        "source": str(path), "target": target, "branch": series.branch, "item": series.item,
        "source_shape_CFHW": series.source_shape, "source_dtypes_x_v": series.source_dtypes,
        "sampled_positions": len(series.token_indices), "total_positions": int(np.prod(series.source_shape[1:])),
        "sampling_seed": seed, "steps": len(series.steps),
        "scope": "Sampled latent/output channels; not transformer hidden tokens; full-compute trajectory, not closed-loop generation.",
        "time_coordinate": "Recorded model timestep; not assumed to be sigma. Deltas are consecutive-step differences.",
        "undefined_ratios": "null in JSON, empty in CSV, NaN in NPZ",
    }
