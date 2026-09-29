"""Temporal cumulative-risk model, loss and cache method."""

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
from eval.model.cumulative import CacheLoss


class TemporalModel(nn.Module):
    """Predict cumulative cache risk from a sequence of old 8D features."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        head_dim: int = 64,
        horizon: int = 4,
        history_length: int = 6,
        initial_prediction: float = 0.01,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.history_length = int(history_length)

        self.token_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.input_dim, int(head_dim)),
            nn.SiLU(),
            nn.Linear(int(head_dim), self.horizon),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        final_layer = self.head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(
            final_layer.bias,
            math.log(math.expm1(float(initial_prediction))),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        history = self.token_projection(features[:, :-1])
        _, hidden = self.gru(history)
        query = features[:, -1]
        increments = F.softplus(
            self.head(torch.cat((hidden[-1], query), dim=-1))
        )
        return torch.cumsum(increments, dim=-1)


TemporalLoss = CacheLoss


class TemporalMethod(CacheMethod):
    """Plan cache prefixes from recent full-compute 8D features."""

    name = "temporal"

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
        self.feature_history: list[torch.Tensor] = []
        self.pending_features: torch.Tensor | None = None
        self.skip_remaining = 0
        self.refresh_pending = False

    @torch.no_grad()
    def decide_conditional(
        self,
        raw_input: TensorList,
        timestep: float,
    ) -> bool:
        self.pending_features = None
        if self.protected:
            self.skip_remaining = 0
            self.refresh_pending = False
            if self.history_ready:
                self.pending_features = _model_features(
                    self, raw_input, timestep, self.device
                )
            return False
        if self.skip_remaining > 0:
            self.skip_remaining -= 1
            return True
        if self.refresh_pending:
            self.refresh_pending = False
            self.pending_features = _model_features(
                self, raw_input, timestep, self.device
            )
            return False
        if not self.history_ready:
            return False

        query = _model_features(self, raw_input, timestep, self.device)
        self.pending_features = query
        history_length = self.model.history_length
        if len(self.feature_history) < history_length:
            return False

        sequence = torch.stack(self.feature_history[-history_length:] + [query])
        model_input = ((sequence - self.feature_mean) / self.feature_std).unsqueeze(0)
        raw_risk = self.model(model_input)[0]
        calibrated_risk = torch.cummax(
            raw_risk + self.calibration_offsets,
            dim=0,
        ).values
        safe = calibrated_risk < self.cache_threshold
        prefix = int(safe.to(torch.int64).cumprod(dim=0).sum().item())
        if prefix > 0:
            self.pending_features = None
            self.feature_history.clear()
            self.skip_remaining = prefix - 1
            self.refresh_pending = True
            return True
        return False

    def observe_conditional(
        self,
        _raw_input: TensorList,
        _output: TensorList,
    ) -> None:
        if self.pending_features is not None:
            self.feature_history.append(self.pending_features)
            self.feature_history = self.feature_history[-self.model.history_length :]
        self.pending_features = None
