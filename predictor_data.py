"""Prompt-disjoint raw trajectories and direct temporal pretraining samples."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset, get_worker_info

from eval.model.polynomial import ResidualHistory
from eval.predictor_schedule import reconstruct_wan_schedule, validate_schedule


def find_raw_trajectories(data_dir: str | Path):
    root = Path(data_dir).expanduser().resolve()
    if (root / "metadata.json").is_file():
        paths = [root]
    else:
        paths = sorted(set(root.glob("trajectory_*")) | set(root.glob("trajectories/trajectory_*")))
    paths = [p for p in paths if p.is_dir() and (p / "_SUCCESS").is_file()]
    if not paths:
        raise ValueError(f"no completed raw trajectories under {root}")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("duplicate trajectory names in flat and nested raw layouts")
    for directory in [root, root.parent, root.parent.parent]:
        config = directory / "dataset.yaml"
        if config.is_file():
            return paths, OmegaConf.load(config).generation
    raise ValueError("raw dataset.yaml with generation settings is required")


def split_by_prompt(paths, val_ratio: float, seed: int):
    groups = {}
    for path in paths:
        prompt = read_metadata(path)["prompt"].strip()
        groups.setdefault(prompt, []).append(path)
    if not 0 < val_ratio < 1 or len(groups) < 2:
        raise ValueError("use at least two distinct prompts and 0 < val_ratio < 1")
    prompts = sorted(groups)
    random.Random(seed).shuffle(prompts)
    count = min(len(prompts) - 1, max(1, round(len(prompts) * val_ratio)))
    return (
        [path for prompt in prompts[count:] for path in sorted(groups[prompt])],
        [path for prompt in prompts[:count] for path in sorted(groups[prompt])],
    )


def read_metadata(path: Path):
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def read_schedule(path: Path, generation, num_train_timesteps: int = 1000):
    saved = read_metadata(path).get("scheduler")
    if saved is not None:
        sigmas, timesteps = validate_schedule(saved["sigmas"], saved["timesteps"])
    else:
        if generation.get("image") is not None:
            raise ValueError("legacy image-conditioned raw data requires saved solver sigmas")
        sigmas, timesteps = reconstruct_wan_schedule(
            str(generation.sample_solver), int(generation.sample_steps),
            float(generation.sample_shift), num_train_timesteps,
        )
    if len(timesteps) != int(generation.sample_steps):
        raise ValueError(f"schedule length differs from dataset.yaml: {path}")
    return sigmas, timesteps


def iter_raw_pairs(path: Path):
    metadata = read_metadata(path)
    pending = []
    expected_step = 0
    for name in metadata["shards"]:
        # Memory mapping keeps only the accessed tensors resident, instead of
        # loading an entire multi-GB trajectory at once.
        records = torch.load(path / name, map_location="cpu", weights_only=True, mmap=True)
        for record in records:
            branch = "conditional" if len(pending) == 0 else "unconditional"
            if record["step_index"] != expected_step or record["branch"] != branch:
                raise ValueError(f"raw records must be complete ordered CFG pairs: {path}")
            for key in ["model_input", "model_output"]:
                values = record[key]
                if len(values) != 1 or values[0].ndim != 4:
                    raise ValueError(f"{key} must contain one [C,F,H,W] video tensor")
            if record["model_input"][0].shape != record["model_output"][0].shape:
                raise ValueError("raw input/output shapes differ")
            pending.append(record)
            if len(pending) == 2:
                yield tuple(pending)
                pending = []
                expected_step += 1
    if pending:
        raise ValueError(f"incomplete CFG pair: {path}")


class RawPredictorDataset(IterableDataset):
    """Sample k < j < i < t directly; keep both CFG branches and full volumes."""

    def __init__(
        self, paths, generation, channels: int, seed: int = 0, shuffle: bool = False,
        num_train_timesteps: int = 1000,
    ):
        self.paths = [Path(p) for p in paths]
        self.generation = generation
        self.channels = int(channels)
        self.seed, self.shuffle, self.epoch = int(seed), shuffle, 0
        self.num_train_timesteps = int(num_train_timesteps)
        if int(generation.sample_steps) < 4:
            raise ValueError("temporal pretraining requires at least four raw nodes")

    def __iter__(self):
        paths = list(self.paths)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(paths)
        worker = get_worker_info()
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]
        for path in paths:
            yield from self.samples(path)

    def samples(self, path: Path):
        sigmas, timesteps = read_schedule(path, self.generation, self.num_train_timesteps)
        # These are memory-mapped views, not copies of all trajectory tensors.
        pairs = list(iter_raw_pairs(path))
        if len(pairs) != len(timesteps):
            raise ValueError(f"incomplete raw trajectory: {path}: {len(pairs)}/{len(timesteps)}")
        for step, pair in enumerate(pairs):
            for record in pair:
                if float(record["timestep"]) != float(timesteps[step]):
                    raise ValueError(f"recorded timestep differs from solver grid: {path}, step {step}")
                if record["model_input"][0].shape[0] != self.channels:
                    raise ValueError(f"raw channels differ from model channels {self.channels}")
        rng = random.Random(f"{self.seed}:{self.epoch if self.shuffle else 0}:{path.name}")
        targets = list(range(3, len(timesteps)))
        if self.shuffle:
            rng.shuffle(targets)
        for step in targets:
            k, j, i = sorted(rng.sample(range(step), 3))
            for branch, record in enumerate(pairs[step]):
                history = ResidualHistory()
                for anchor in (k, j, i):
                    node = pairs[anchor][branch]
                    history.push(
                        anchor, float(sigmas[anchor]),
                        node["model_input"][0][None].float(),
                        node["model_output"][0][None].float(),
                    )
                bases, q = history.features(step, float(sigmas[step]))
                yield {
                    "x": record["model_input"][0].float(), "bases": bases[0],
                    "q": q[0], "target": record["model_output"][0].float(),
                    "step": step, "anchor": i, "history_steps": (i, j, k),
                    "branch": branch,
                }
