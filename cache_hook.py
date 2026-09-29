"""Temporary Wan forward replacement used during one video generation."""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Any, Callable, Iterator


@contextmanager
def patch_forward(model: Any, forward: Callable[..., Any]) -> Iterator[None]:
    original_forward = model.forward
    model.forward = types.MethodType(forward, model)
    try:
        yield
    finally:
        model.forward = original_forward
