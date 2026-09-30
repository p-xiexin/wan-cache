"""Checkpoint serialization for the learned cache model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from torch import nn


SCHEMA_VERSION = 1


def build_predictor_artifact(
    model: nn.Module, model_config: Mapping[str, Any], cache_threshold: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "residual_polynomial",
        "model_config": dict(model_config),
        "model_state_dict": {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        },
        "cache_threshold": float(cache_threshold),
    }


def build_artifact(
    model: nn.Module,
    model_config: Mapping[str, Any],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    calibration_offsets: torch.Tensor,
    cache_threshold: float,
) -> dict[str, Any]:
    """Build a weights-only-safe deployment artifact."""

    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "model_config": dict(model_config),
        "model_state_dict": state_dict,
        "feature_mean": feature_mean.detach().cpu().flatten(),
        "feature_std": feature_std.detach().cpu().flatten(),
        "calibration_offsets": calibration_offsets.detach().cpu().flatten(),
        "cache_threshold": float(cache_threshold),
    }


def load_artifact(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load one artifact dictionary onto the selected device."""

    artifact = torch.load(
        Path(path).expanduser(),
        map_location=torch.device(device),
        weights_only=True,
    )
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported model artifact schema")
    return artifact


def instantiate_artifact_model(
    artifact: Mapping[str, Any],
    device: str | torch.device,
) -> nn.Module:
    """Instantiate the saved Hydra model config and restore its parameters."""

    model = instantiate(artifact["model_config"])
    model.load_state_dict(artifact["model_state_dict"])
    return model.to(device)
