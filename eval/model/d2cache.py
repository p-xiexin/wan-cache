"""Output-residual adaptation of D2Cache for Wan.

Original implementation
https://github.com/VG-Huai/D2Cache/blob/main/D2Cache4Wan2.1/delta2cache_generate.py

The original method corrects cached transformer-block feature residuals with
their latest delta. This research variant applies the correction to the full
DiT output residual and keeps the existing output-level EasyCache schedule.
"""

from __future__ import annotations

from typing import Any

from eval.model.base import TensorList, _mean_abs, _mean_abs_difference
from eval.model.easycache import EasyCacheMethod


class D2CacheMethod(EasyCacheMethod):
    name = "d2cache_output"

    def _reset_method(self) -> None:
        super()._reset_method()
        self.previous_error_score = 0.0
        self.correction_scale = 1.0
        self.correction_scales: list[float] = []
        self.residual_delta_even: TensorList | None = None
        self.residual_delta_odd: TensorList | None = None

    def decide_conditional(
        self,
        raw_input: TensorList,
        timestep: float,
    ) -> bool:
        if self.protected:
            self.previous_error_score = self.accumulated_error
            self.accumulated_error = 0.0
            return False
        if not self.history_ready or self.transformation_rate is None:
            return False

        predicted_change = (
            self.transformation_rate
            * _mean_abs_difference(raw_input, self.previous_raw_input_even)
            / (_mean_abs(self.previous_raw_output_even) + self.epsilon)
        )
        self.accumulated_error += predicted_change
        if self.accumulated_error < self.cache_threshold:
            denominator = self.previous_error_score or self.accumulated_error
            self.correction_scale = self.accumulated_error / denominator
            self.correction_scales.append(self.correction_scale)
            return True

        self.previous_error_score = self.accumulated_error
        self.accumulated_error = 0.0
        self.correction_scale = 1.0
        return False

    def update(self, raw_input: TensorList, output: TensorList) -> None:
        is_conditional = self.forward_index % 2 == 0
        previous_residual = self.cache_even if is_conditional else self.cache_odd
        super().update(raw_input, output)
        current_residual = self.cache_even if is_conditional else self.cache_odd
        if previous_residual is not None:
            residual_delta = [
                current - previous
                for current, previous in zip(current_residual, previous_residual)
            ]
            if is_conditional:
                self.residual_delta_even = residual_delta
            else:
                self.residual_delta_odd = residual_delta

    def predict_cached_residual(
        self,
        raw_input: TensorList,
        timestep: Any,
        is_conditional: bool,
    ) -> TensorList:
        residual = self.cache_even if is_conditional else self.cache_odd
        residual_delta = (
            self.residual_delta_even if is_conditional else self.residual_delta_odd
        )
        if residual_delta is None:
            return residual
        return [
            cached + self.correction_scale * delta
            for cached, delta in zip(residual, residual_delta)
        ]

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result["mean_correction_scale"] = (
            sum(self.correction_scales) / len(self.correction_scales)
            if self.correction_scales
            else 0.0
        )
        return result
