"""Three-coefficient residual predictor and EasyCache-scheduled deployment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from eval.artifacts import instantiate_artifact_model, load_artifact
from eval.model.base import TensorList
from eval.model.easycache import EasyCacheMethod
from eval.predictor_schedule import validate_schedule


class ConvResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1, groups=width),
            nn.SiLU(),
            nn.Conv3d(width, width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class PolynomialPredictor(nn.Module):
    """[x, P0, P1, P2] -> 32-channel encoder -> pool + q -> three scalars."""

    def __init__(self, channels: int = 48, hidden_dim: int = 32, a_max: float = 2.0):
        super().__init__()
        if channels <= 0 or hidden_dim <= 0 or not math.isfinite(a_max) or a_max <= 1:
            raise ValueError("channels/hidden_dim must be positive and a_max > 1")
        self.channels = int(channels)
        self.a_max = float(a_max)
        self.encoder = nn.Sequential(
            nn.Conv3d(4 * channels, hidden_dim, 1),
            ConvResidualBlock(hidden_dim),
            ConvResidualBlock(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3),
        )
        # Start from standard quadratic extrapolation, with identical treatment
        # of all three coefficients.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, math.atanh(1.0 / self.a_max))

    def forward(self, x, bases, q):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError(f"x must be [B,{self.channels},F,H,W]")
        if bases.shape != (x.shape[0], 3, *x.shape[1:]) or q.shape != (x.shape[0], 5):
            raise ValueError("bases must be [B,3,C,F,H,W] and q must be [B,5]")
        # Keep divided differences and reconstruction in FP32, including inside
        # the outer Wan BF16 autocast context.
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = torch.cat([x.float(), bases.float().flatten(1, 2)], dim=1)
            pooled = self.encoder(z).mean(dim=(2, 3, 4))
            logits = self.head(torch.cat([pooled, q.float()], dim=1))
            return self.a_max * logits.tanh()

    def predict(self, x, bases, q):
        coefficients = self(x, bases, q)
        residual = (bases.float() * coefficients[:, :, None, None, None, None]).sum(1)
        return x.float() + residual, coefficients


class PolynomialLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3, regularization: float = 1e-3):
        super().__init__()
        if epsilon <= 0 or regularization < 0:
            raise ValueError("epsilon must be positive and regularization nonnegative")
        self.epsilon = float(epsilon)
        self.regularization = float(regularization)

    def forward(self, prediction, target, coefficients):
        error = prediction.float() - target.detach().float()
        return (
            (error.square() + self.epsilon**2).sqrt().mean()
            + self.regularization * (coefficients.float() - 1).square().mean()
        )


@dataclass
class ResidualNode:
    step: int
    sigma: float
    residual: torch.Tensor  # [B,C,F,H,W], full DiT only


class ResidualHistory:
    """Latest three full nodes; differences are recomputed only on refresh."""

    def __init__(self):
        self.nodes: list[ResidualNode] = []
        self.first = self.second = None

    def push(self, step: int, sigma: float, x: torch.Tensor, v: torch.Tensor):
        if self.nodes and (step <= self.nodes[0].step or sigma >= self.nodes[0].sigma):
            raise ValueError("full nodes must advance in step and decrease in sigma")
        self.nodes.insert(0, ResidualNode(step, sigma, (v.float() - x.float()).detach()))
        del self.nodes[3:]
        if len(self.nodes) == 3:
            i, j, k = self.nodes
            self.first = (i.residual - j.residual) / (i.sigma - j.sigma)
            older = (j.residual - k.residual) / (j.sigma - k.sigma)
            self.second = (self.first - older) / (i.sigma - k.sigma)

    def features(self, step: int, sigma: float):
        if len(self.nodes) != 3:
            raise RuntimeError("three full nodes are required before prediction")
        i, j, k = self.nodes
        if step <= i.step or sigma >= i.sigma:
            raise ValueError("prediction must follow the latest full node")
        bases = torch.stack([
            i.residual,
            (sigma - i.sigma) * self.first,
            (sigma - i.sigma) * (sigma - j.sigma) * self.second,
        ], dim=1)
        q = bases.new_tensor([
            sigma, sigma - i.sigma, i.sigma - j.sigma, j.sigma - k.sigma, step - i.step,
        ]).expand(bases.shape[0], -1)
        return bases, q


class PolynomialMethod(EasyCacheMethod):
    name = "polynomial"
    requires_sigma_schedule = True

    def __init__(
        self, artifact_path: str | None = None, cache_threshold: float | None = None,
        device=None, model: PolynomialPredictor | None = None,
    ):
        if (artifact_path is None) == (model is None):
            raise ValueError("provide exactly one predictor artifact or model")
        if artifact_path is not None:
            device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            artifact = load_artifact(artifact_path, device)
            if artifact.get("kind") != "residual_polynomial":
                raise ValueError("expected a residual_polynomial predictor artifact")
            model = instantiate_artifact_model(artifact, device).eval()
            if cache_threshold is None:
                cache_threshold = float(artifact["cache_threshold"])
        super().__init__(cache_threshold=0.05 if cache_threshold is None else cache_threshold)
        self.model = model

    def reset(self, sample_steps, warmup_steps, final_full_steps):
        if warmup_steps < 3 or final_full_steps < 0 or warmup_steps + final_full_steps > sample_steps:
            raise ValueError("polynomial cache requires at least three warmup steps")
        super().reset(sample_steps, warmup_steps, final_full_steps)

    def _reset_method(self):
        super()._reset_method()
        self.histories = {True: ResidualHistory(), False: ResidualHistory()}
        self.sigmas = self.timesteps = None
        self.last_coefficients = {}

    def set_schedule(self, sigmas, timesteps):
        self.sigmas, self.timesteps = validate_schedule(sigmas, timesteps)
        if len(self.timesteps) != self.sample_steps:
            raise ValueError("schedule length differs from generation sample_steps")

    @property
    def history_ready(self):
        return super().history_ready and all(len(h.nodes) == 3 for h in self.histories.values())

    def try_skip(self, raw_input, timestep):
        if self.sigmas is None:
            raise RuntimeError("set the actual solver sigma schedule before generation")
        observed = float(torch.as_tensor(timestep).detach().max())
        if not math.isclose(observed, float(self.timesteps[self.pair_index]), abs_tol=1e-4):
            raise ValueError("model timestep does not match the configured solver schedule")
        return super().try_skip(raw_input, timestep)

    def update(self, raw_input: TensorList, output: TensorList):
        branch = self.forward_index % 2 == 0
        self.histories[branch].push(
            self.pair_index, float(self.sigmas[self.pair_index]),
            torch.stack(raw_input), torch.stack(output),
        )
        super().update(raw_input, output)

    def predict_cached_residual(self, raw_input, timestep, is_conditional):
        del timestep
        x = torch.stack(raw_input).float()
        bases, q = self.histories[is_conditional].features(
            self.pair_index, float(self.sigmas[self.pair_index]),
        )
        coefficients = self.model(x, bases, q)
        self.last_coefficients[is_conditional] = coefficients
        residual = (bases * coefficients[:, :, None, None, None, None]).sum(1)
        return list(residual.unbind(0))
