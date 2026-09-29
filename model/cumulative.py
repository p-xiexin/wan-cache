"""Cumulative-risk model, loss and cache method."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from eval.artifacts import instantiate_artifact_model, load_artifact
from eval.model.base import (
    CacheMethod,
    TensorList,
    _model_features,
)

class CacheModel(nn.Module):
    """Predict monotonic cumulative cache risk over a fixed horizon."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        horizon: int = 4,
        initial_prediction: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.horizon = int(horizon)
        hidden_dim = int(hidden_dim)
        num_hidden_layers = int(num_hidden_layers)
        initial_prediction = float(initial_prediction)

        layers: list[nn.Module] = []
        current_dim = self.input_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.SiLU()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.horizon))
        self.network = nn.Sequential(*layers)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        final_layer = self.network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(
            final_layer.bias,
            math.log(math.expm1(initial_prediction)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(F.softplus(self.network(features)), dim=-1)


def _masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    effective = mask.to(values.dtype) * weights.unsqueeze(0)
    return (values * effective).sum() / effective.sum().clamp_min(1.0)


class CacheLoss(nn.Module):
    """Risk-aware quantile loss for cumulative cache-error predictions."""

    def __init__(
        self,
        threshold: float = 0.05,
        horizon_decay: float = 1.0,
        underestimate_weight: float = 2.0,
        unsafe_weight: float = 4.0,
        quantile: float = 0.90,
        near_threshold_weight: float = 2.0,
        threshold_bandwidth: float = 0.02,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.horizon_decay = float(horizon_decay)
        self.underestimate_weight = float(underestimate_weight)
        self.unsafe_weight = float(unsafe_weight)
        self.quantile = float(quantile)
        self.near_threshold_weight = float(near_threshold_weight)
        self.threshold_bandwidth = float(threshold_bandwidth)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        offsets = torch.arange(
            prediction.shape[1],
            device=prediction.device,
            dtype=prediction.dtype,
        )
        weights = torch.pow(prediction.new_tensor(self.horizon_decay), offsets)
        residual = target - prediction
        pointwise = torch.maximum(
            self.quantile * residual,
            (self.quantile - 1.0) * residual,
        )
        if self.near_threshold_weight > 0:
            proximity = torch.exp(
                -(target - self.threshold).abs() / self.threshold_bandwidth
            )
            pointwise = pointwise * (
                1.0 + self.near_threshold_weight * proximity
            )

        regression = _masked_weighted_mean(pointwise, mask, weights)
        underestimation = _masked_weighted_mean(
            F.relu(target - prediction).square(),
            mask,
            weights,
        )
        unsafe_mask = mask.to(torch.bool) & (target >= self.threshold)
        unsafe = _masked_weighted_mean(
            F.relu(self.threshold - prediction).square(),
            unsafe_mask,
            weights,
        )
        return (
            regression
            + self.underestimate_weight * underestimation
            + self.unsafe_weight * unsafe
        )

class CumulativeMethod(CacheMethod):
    """Fixed cumulative-risk method driven by a trained ``CacheModel``."""

    name = "model"

    def __init__(
        self,
        artifact_path: str,
        cache_threshold: float | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        artifact = Path(artifact_path).expanduser().resolve()
        if device is None:
            resolved_device = torch.device(
                "cuda", torch.cuda.current_device()
            ) if torch.cuda.is_available() else torch.device("cpu")
        else:
            resolved_device = torch.device(device)

        self.device = resolved_device
        payload = load_artifact(artifact, resolved_device)
        self.model = instantiate_artifact_model(payload, resolved_device)
        self.model.eval()
        self.cache_threshold = (
            float(payload["cache_threshold"])
            if cache_threshold is None
            else float(cache_threshold)
        )
        self.feature_mean = torch.as_tensor(
            payload["feature_mean"], dtype=torch.float32, device=resolved_device
        ).flatten()
        self.feature_std = torch.as_tensor(
            payload["feature_std"], dtype=torch.float32, device=resolved_device
        ).flatten()
        self.calibration_offsets = torch.as_tensor(
            payload["calibration_offsets"],
            dtype=torch.float32,
            device=resolved_device,
        ).flatten()

    def _reset_method(self) -> None:
        self.skip_remaining = 0
        self.refresh_pending = False

    @torch.no_grad()
    def decide_conditional(
        self,
        raw_input: TensorList,
        timestep: float,
    ) -> bool:
        if self.protected:
            self.skip_remaining = 0
            self.refresh_pending = False
            return False
        if self.skip_remaining > 0:
            self.skip_remaining -= 1
            return True
        if self.refresh_pending:
            self.refresh_pending = False
            return False
        if not self.history_ready:
            return False

        features = _model_features(self, raw_input, timestep, self.device)
        normalized = (features - self.feature_mean) / self.feature_std
        raw_risk = self.model(normalized.unsqueeze(0))[0]
        calibrated_risk = torch.cummax(
            raw_risk + self.calibration_offsets,
            dim=0,
        ).values
        safe = calibrated_risk < self.cache_threshold
        prefix = int(safe.to(torch.int64).cumprod(dim=0).sum().item())
        if prefix > 0:
            self.skip_remaining = prefix - 1
            self.refresh_pending = True
            return True

        return False
