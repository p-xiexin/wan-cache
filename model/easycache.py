"""Formula-aligned EasyCache method."""

from __future__ import annotations

from eval.model.base import (
    CacheMethod,
    TensorList,
    _mean_abs,
    _mean_abs_difference,
)


class EasyCacheMethod(CacheMethod):
    """Residual-cache method from the original EasyCache TI2V script."""

    name = "easycache"

    def __init__(self, cache_threshold: float, epsilon: float = 1e-8) -> None:
        self.cache_threshold = float(cache_threshold)
        self.epsilon = float(epsilon)

    def _reset_method(self) -> None:
        self.accumulated_error = 0.0
        self.transformation_rate = None
        self.last_full_input_even = None

    def decide_conditional(
        self,
        raw_input: TensorList,
        timestep: float,
    ) -> bool:
        if self.protected:
            self.accumulated_error = 0.0
            return False
        if not self.history_ready:
            return False
        if self.transformation_rate is None:
            return False

        output_norm = _mean_abs(self.previous_raw_output_even)
        input_change = _mean_abs_difference(
            raw_input,
            self.previous_raw_input_even,
        )
        predicted_change = (
            self.transformation_rate * input_change / (output_norm + self.epsilon)
        )
        self.accumulated_error += predicted_change
        if self.accumulated_error < self.cache_threshold:
            return True
        self.accumulated_error = 0.0
        return False

    def observe_conditional(
        self,
        raw_input: TensorList,
        output: TensorList,
    ) -> None:
        if (
            self.previous_raw_output_even is not None
            and self.last_full_input_even is not None
        ):
            output_change = _mean_abs_difference(
                output,
                self.previous_raw_output_even,
            )
            input_change = _mean_abs_difference(
                raw_input,
                self.last_full_input_even,
            )
            if input_change > self.epsilon:
                self.transformation_rate = output_change / input_change
        self.last_full_input_even = raw_input
