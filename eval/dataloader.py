"""Method-independent trajectory loading for cache-policy training."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Trajectory:
    features: torch.Tensor
    increments: torch.Tensor
    step_indices: torch.Tensor
    valid: torch.Tensor
    step_stride: int


@dataclass
class WindowDataset:
    features: torch.Tensor
    targets: torch.Tensor
    masks: torch.Tensor


@dataclass
class DataBundle:
    trajectory_count: int
    train_window_count: int
    val_window_count: int
    train_loader: DataLoader
    val_loader: DataLoader
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


def infer_step_stride(step_indices: torch.Tensor) -> int:
    if step_indices.numel() < 2:
        return 1
    differences = step_indices[1:] - step_indices[:-1]
    positive = differences[differences > 0]
    if positive.numel() == 0:
        raise ValueError("step_indices contain no positive increment")
    values, counts = torch.unique(positive, return_counts=True)
    return int(values[counts.argmax()].item())


def load_trajectory(
    path: Path,
    raw_feature_dim: int,
    max_target: float | None,
    expected_step_stride: int,
    require_step_indices: bool,
) -> Trajectory:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = torch.as_tensor(payload["features"], dtype=torch.float32).cpu()
    increments = torch.as_tensor(payload["targets"], dtype=torch.float32).flatten().cpu()

    if features.ndim != 2 or features.shape[1] != raw_feature_dim:
        raise ValueError(
            f"features must have shape [N,{raw_feature_dim}], got "
            f"{tuple(features.shape)}"
        )
    if increments.shape != (features.shape[0],):
        raise ValueError(
            f"targets must have shape [{features.shape[0]}], got "
            f"{tuple(increments.shape)}"
        )

    if "step_indices" in payload:
        step_indices = torch.as_tensor(
            payload["step_indices"], dtype=torch.long
        ).flatten().cpu()
        if step_indices.shape != (features.shape[0],):
            raise ValueError(
                f"step_indices must have shape [{features.shape[0]}], got "
                f"{tuple(step_indices.shape)}"
            )
    else:
        if require_step_indices:
            raise KeyError(f"{path} has no step_indices")
        step_indices = torch.arange(features.shape[0], dtype=torch.long)

    valid = torch.isfinite(features).all(dim=1)
    valid &= torch.isfinite(increments)
    valid &= increments >= 0
    if max_target is not None:
        valid &= increments <= max_target
    if not valid.any():
        raise ValueError(f"{path} contains no valid samples")

    step_stride = (
        expected_step_stride
        if expected_step_stride > 0
        else infer_step_stride(step_indices)
    )

    return Trajectory(
        features=features.contiguous(),
        increments=increments.contiguous(),
        step_indices=step_indices.contiguous(),
        valid=valid.contiguous(),
        step_stride=step_stride,
    )


def load_dataset(
    data_dir: Path,
    pattern: str,
    raw_feature_dim: int,
    max_target: float | None,
    expected_step_stride: int,
    require_step_indices: bool,
) -> list[Trajectory]:
    files = sorted(data_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} below {data_dir}")

    trajectories = [
        load_trajectory(
            path=path,
            raw_feature_dim=raw_feature_dim,
            max_target=max_target,
            expected_step_stride=expected_step_stride,
            require_step_indices=require_step_indices,
        )
        for path in files
    ]

    return trajectories


def split_by_trajectory(
    trajectories: list[Trajectory],
    val_ratio: float,
    seed: int,
) -> tuple[list[Trajectory], list[Trajectory]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")

    shuffled = list(trajectories)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1 or val_ratio == 0:
        logging.warning("Validation reuses the training trajectories")
        return shuffled, shuffled

    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def build_windows(
    trajectories: list[Trajectory],
    horizon: int,
    min_valid_horizon: int,
    full_horizon_only: bool,
    history_length: int = 0,
) -> WindowDataset:
    feature_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    for trajectory in trajectories:
        sample_count = trajectory.increments.numel()
        for start in range(sample_count):
            history_start = start - history_length
            if history_start < 0:
                continue

            history_valid = True
            for index in range(history_start, start + 1):
                if not bool(trajectory.valid[index]):
                    history_valid = False
                    break
                if index > history_start:
                    stride = int(
                        trajectory.step_indices[index]
                        - trajectory.step_indices[index - 1]
                    )
                    if stride != trajectory.step_stride:
                        history_valid = False
                        break
            if not history_valid:
                continue

            increments = torch.zeros(horizon, dtype=torch.float32)
            mask = torch.zeros(horizon, dtype=torch.bool)
            for offset in range(horizon):
                index = start + offset
                if index >= sample_count or not bool(trajectory.valid[index]):
                    break
                if offset > 0:
                    stride = int(
                        trajectory.step_indices[index]
                        - trajectory.step_indices[index - 1]
                    )
                    if stride != trajectory.step_stride:
                        break
                increments[offset] = trajectory.increments[index]
                mask[offset] = True

            required = horizon if full_horizon_only else min_valid_horizon
            if int(mask.sum().item()) < required:
                continue

            if history_length == 0:
                feature_rows.append(trajectory.features[start])
            else:
                feature_rows.append(trajectory.features[history_start : start + 1])
            target_rows.append(torch.cumsum(increments, dim=0))
            mask_rows.append(mask)

    if not feature_rows:
        raise RuntimeError("No valid horizon windows were constructed")

    return WindowDataset(
        features=torch.stack(feature_rows).contiguous(),
        targets=torch.stack(target_rows).contiguous(),
        masks=torch.stack(mask_rows).contiguous(),
    )


def make_loader(
    dataset: WindowDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    features = ((dataset.features - mean) / std).contiguous()
    return DataLoader(
        TensorDataset(features, dataset.targets, dataset.masks),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


class TrajectoryDataModule:
    """Hydra-instantiable trajectory data pipeline."""

    def __init__(
        self,
        data_dir: str,
        pattern: str = "*_lazy_data.pt",
        batch_size: int = 256,
        num_workers: int = 0,
        val_ratio: float = 0.2,
        seed: int = 0,
        min_valid_horizon: int = 1,
        full_horizon_only: bool = False,
        expected_step_stride: int = 0,
        require_step_indices: bool = False,
        max_target: float | None = None,
        normalize: bool = True,
        history_length: int = 0,
        pin_memory: bool | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.pattern = pattern
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.seed = seed
        self.min_valid_horizon = min_valid_horizon
        self.full_horizon_only = full_horizon_only
        self.expected_step_stride = expected_step_stride
        self.require_step_indices = require_step_indices
        self.max_target = max_target
        self.normalize = normalize
        self.history_length = int(history_length)
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    def setup(self, raw_feature_dim: int, horizon: int) -> DataBundle:
        if raw_feature_dim <= 0 or horizon <= 0 or self.history_length < 0:
            raise ValueError("feature dimension and horizon must be positive")
        if not 1 <= self.min_valid_horizon <= horizon:
            raise ValueError("min_valid_horizon must be in [1, horizon]")

        trajectories = load_dataset(
            data_dir=self.data_dir,
            pattern=self.pattern,
            raw_feature_dim=raw_feature_dim,
            max_target=self.max_target,
            expected_step_stride=self.expected_step_stride,
            require_step_indices=self.require_step_indices,
        )
        train_trajectories, val_trajectories = split_by_trajectory(
            trajectories,
            self.val_ratio,
            self.seed,
        )
        train_dataset = build_windows(
            train_trajectories,
            horizon,
            self.min_valid_horizon,
            self.full_horizon_only,
            self.history_length,
        )
        val_dataset = build_windows(
            val_trajectories,
            horizon,
            self.min_valid_horizon,
            self.full_horizon_only,
            self.history_length,
        )

        if self.normalize:
            feature_dims = 0 if self.history_length == 0 else (0, 1)
            feature_mean = train_dataset.features.mean(dim=feature_dims)
            feature_std = train_dataset.features.std(
                dim=feature_dims,
                unbiased=False,
            )
            feature_std = torch.where(
                feature_std < 1e-6,
                torch.ones_like(feature_std),
                feature_std,
            )
        else:
            feature_mean = torch.zeros(raw_feature_dim)
            feature_std = torch.ones(raw_feature_dim)

        return DataBundle(
            trajectory_count=len(trajectories),
            train_window_count=int(train_dataset.features.shape[0]),
            val_window_count=int(val_dataset.features.shape[0]),
            train_loader=make_loader(
                train_dataset,
                feature_mean,
                feature_std,
                self.batch_size,
                True,
                self.num_workers,
                self.pin_memory,
            ),
            val_loader=make_loader(
                val_dataset,
                feature_mean,
                feature_std,
                self.batch_size,
                False,
                self.num_workers,
                self.pin_memory,
            ),
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
