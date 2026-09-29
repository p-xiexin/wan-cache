"""Shared cache state machine for Wan CFG forwards."""

from __future__ import annotations

from typing import Any

import torch


TensorList = list[torch.Tensor]


def _mean_abs(values: TensorList) -> float:
    total = sum(float(value.abs().sum().item()) for value in values)
    count = sum(value.numel() for value in values)
    return total / count


def _mean_abs_difference(left: TensorList, right: TensorList) -> float:
    total = sum(
        float((lhs - rhs).abs().sum().item())
        for lhs, rhs in zip(left, right)
    )
    count = sum(value.numel() for value in left)
    return total / count


class CacheMethod:
    """Own the cache state and decide one conditional/unconditional CFG pair."""

    name = "base"
    accelerated = True
    cache_threshold: float | None = None

    def reset(
        self,
        sample_steps: int,
        warmup_steps: int,
        final_full_steps: int,
    ) -> None:
        self.sample_steps = int(sample_steps)
        self.warmup_steps = int(warmup_steps)
        self.final_full_steps = int(final_full_steps)
        self.forward_index = 0
        self.skip_current_pair = False
        self.previous_raw_input_even: TensorList | None = None
        self.previous_raw_output_even: TensorList | None = None
        self.prev_previous_raw_input_even: TensorList | None = None
        self.prev_previous_raw_output_even: TensorList | None = None
        self.cache_even: TensorList | None = None
        self.cache_odd: TensorList | None = None
        self.calculated_pairs: list[int] = []
        self.skipped_pairs: list[int] = []
        self._reset_method()

    @property
    def pair_index(self) -> int:
        return self.forward_index // 2

    @property
    def history_ready(self) -> bool:
        return (
            self.previous_raw_input_even is not None
            and self.previous_raw_output_even is not None
            and self.cache_even is not None
            and self.cache_odd is not None
        )

    @property
    def protected(self) -> bool:
        return (
            self.pair_index < self.warmup_steps
            or self.pair_index >= self.sample_steps - self.final_full_steps
        )

    def try_skip(
        self,
        raw_input: TensorList,
        timestep: Any,
    ) -> TensorList | None:
        is_conditional = self.forward_index % 2 == 0
        if is_conditional:
            timestep_value = (
                float(timestep.detach().flatten()[0].item())
                if torch.is_tensor(timestep)
                else float(timestep)
            )
            wants_skip = self.decide_conditional(raw_input, timestep_value)
            self.skip_current_pair = (
                wants_skip and self.history_ready and not self.protected
            )
            if self.skip_current_pair:
                self.skipped_pairs.append(self.pair_index)
            else:
                self.calculated_pairs.append(self.pair_index)

        if not self.skip_current_pair:
            return None

        cached_residual = self.predict_cached_residual(
            raw_input,
            timestep,
            is_conditional,
        )
        if is_conditional:
            self.prev_previous_raw_input_even = self.previous_raw_input_even
            self.previous_raw_input_even = raw_input
        self.forward_index += 1
        return [
            (input_tensor + residual).float()
            for input_tensor, residual in zip(raw_input, cached_residual)
        ]

    def predict_cached_residual(
        self,
        raw_input: TensorList,
        timestep: Any,
        is_conditional: bool,
    ) -> TensorList:
        return self.cache_even if is_conditional else self.cache_odd

    def update(self, raw_input: TensorList, output: TensorList) -> None:
        is_conditional = self.forward_index % 2 == 0
        if is_conditional:
            self.observe_conditional(raw_input, output)
            self.prev_previous_raw_input_even = self.previous_raw_input_even
            self.previous_raw_input_even = raw_input
            self.prev_previous_raw_output_even = self.previous_raw_output_even
            self.previous_raw_output_even = [value.clone() for value in output]
            self.cache_even = [
                output_tensor - input_tensor
                for output_tensor, input_tensor in zip(output, raw_input)
            ]
        else:
            self.cache_odd = [
                output_tensor - input_tensor
                for output_tensor, input_tensor in zip(output, raw_input)
            ]
        self.forward_index += 1

    def decide_conditional(
        self,
        raw_input: TensorList,
        timestep: float,
    ) -> bool:
        """Return True when the current CFG pair should use the cache."""

        raise NotImplementedError

    def observe_conditional(
        self,
        raw_input: TensorList,
        output: TensorList,
    ) -> None:
        """Observe a conditional output produced by the full DiT."""

    def _reset_method(self) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "cache_threshold": self.cache_threshold,
            "sample_steps": self.sample_steps,
            "calculated_pair_indices": self.calculated_pairs,
            "skipped_pair_indices": self.skipped_pairs,
            "calculated_pairs": len(self.calculated_pairs),
            "skipped_pairs": len(self.skipped_pairs),
            "skip_ratio_all_pairs": len(self.skipped_pairs) / self.sample_steps,
        }


def _model_features(
    method: CacheMethod,
    raw_input: TensorList,
    timestep: float,
    device: torch.device,
) -> torch.Tensor:
    current_change = _mean_abs_difference(
        raw_input,
        method.previous_raw_input_even,
    )
    previous_input_norm = _mean_abs(method.previous_raw_input_even)
    if method.prev_previous_raw_input_even is None:
        previous_change_relative = 0.0
    else:
        previous_change_relative = _mean_abs_difference(
            method.previous_raw_input_even,
            method.prev_previous_raw_input_even,
        ) / (_mean_abs(method.prev_previous_raw_input_even) + 1e-8)

    current_mean = _mean_abs(raw_input)
    if method.prev_previous_raw_output_even is None:
        previous_output_change_relative = 0.0
    else:
        previous_output_change_relative = _mean_abs_difference(
            method.previous_raw_output_even,
            method.prev_previous_raw_output_even,
        ) / (_mean_abs(method.prev_previous_raw_output_even) + 1e-8)

    return torch.tensor(
        [
            timestep / 1000.0,
            current_change / (previous_input_norm + 1e-8),
            previous_change_relative,
            current_mean,
            previous_input_norm,
            previous_output_change_relative,
            _mean_abs(method.cache_even) / (current_mean + 1e-8),
            method.pair_index / method.sample_steps,
        ],
        dtype=torch.float32,
        device=device,
    )
