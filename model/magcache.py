"""Output-residual adaptation of MagCache for Wan2.2.

Original implementation
https://github.com/Zehong-Ma/MagCache/blob/main/MagCache4Wan2.2/magcache_generate.py

The original method caches transformer-block feature residuals. This research
variant applies the same magnitude-ratio schedule to the full DiT output
residual exposed by the evaluation pipeline.
"""

from __future__ import annotations

from typing import Any

from eval.model.base import CacheMethod, TensorList


class MagCacheMethod(CacheMethod):
    name = "magcache_output"

    def __init__(
        self,
        cache_threshold: float,
        magnitude_ratios: list[float],
        max_cached_pairs: int = 2,
        retention_ratio: float = 0.2,
    ) -> None:
        self.cache_threshold = float(cache_threshold)
        self.magnitude_ratios = [float(value) for value in magnitude_ratios]
        self.max_cached_pairs = int(max_cached_pairs)
        self.retention_ratio = float(retention_ratio)

    def _reset_method(self) -> None:
        self.accumulated_ratio = 1.0
        self.accumulated_error = 0.0
        self.cached_pairs = 0

    def decide_conditional(
        self,
        raw_input: TensorList,
        timestep: float,
    ) -> bool:
        if self.protected or self.pair_index < int(
            self.sample_steps * self.retention_ratio
        ):
            self._reset_method()
            return False

        self.accumulated_ratio *= self.magnitude_ratios[self.pair_index]
        self.cached_pairs += 1
        self.accumulated_error += abs(1.0 - self.accumulated_ratio)
        if (
            self.accumulated_error < self.cache_threshold
            and self.cached_pairs <= self.max_cached_pairs
        ):
            return True

        self._reset_method()
        return False

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "max_cached_pairs": self.max_cached_pairs,
                "retention_ratio": self.retention_ratio,
            }
        )
        return result
