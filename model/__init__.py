"""Models and cache methods used by the evaluation pipeline."""

from eval.model.base import CacheMethod
from eval.model.cumulative import CacheLoss, CacheModel, CumulativeMethod
from eval.model.d2cache import D2CacheMethod
from eval.model.easycache import EasyCacheMethod
from eval.model.magcache import MagCacheMethod
from eval.model.origin import OriginMethod
from eval.model.temporal import TemporalLoss, TemporalMethod, TemporalModel

__all__ = [
    "CacheLoss",
    "CacheMethod",
    "CacheModel",
    "CumulativeMethod",
    "D2CacheMethod",
    "EasyCacheMethod",
    "MagCacheMethod",
    "OriginMethod",
    "TemporalLoss",
    "TemporalMethod",
    "TemporalModel",
]
