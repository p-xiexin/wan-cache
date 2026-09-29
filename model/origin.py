"""Full-compute Origin method."""

from __future__ import annotations

from typing import Any

from eval.model.base import CacheMethod, TensorList


class OriginMethod(CacheMethod):
    name = "origin"
    accelerated = False

    def decide_conditional(
        self,
        raw_input: TensorList,
        timestep: float,
    ) -> bool:
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "cache_threshold": None,
            "sample_steps": self.sample_steps,
            "calculated_pair_indices": list(range(self.sample_steps)),
            "skipped_pair_indices": [],
            "calculated_pairs": self.sample_steps,
            "skipped_pairs": 0,
            "skip_ratio_all_pairs": 0.0,
        }
