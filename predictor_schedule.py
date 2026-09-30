"""Wan flow schedules; never infer a sigma from a rounded model timestep."""

from __future__ import annotations

import numpy as np
import torch


def validate_schedule(sigmas, timesteps) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32).detach().cpu().flatten()
    timesteps = torch.as_tensor(timesteps).detach().cpu().flatten()
    if len(sigmas) != len(timesteps) + 1 or len(timesteps) < 3:
        raise ValueError("expected N model timesteps and N+1 solver sigmas")
    if not torch.isfinite(sigmas).all() or not torch.isfinite(timesteps).all():
        raise ValueError("schedule contains non-finite values")
    if not (sigmas[:-1] > sigmas[1:]).all():
        raise ValueError("polynomial prediction requires strictly decreasing sigmas")
    return sigmas, timesteps


def reconstruct_wan_schedule(
    sample_solver: str, sample_steps: int, sample_shift: float,
    num_train_timesteps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the official Wan2.2 static schedules for legacy raw data.

    UniPC starts at float32(1 - 1/N_train); DPM++ starts at 1.
    Timesteps are truncated to int64 *before* the sigma array is cast to float32.
    See wan/utils/fm_solvers{,_unipc}.py and WanTI2V.t2v.
    """
    if sample_steps < 3 or num_train_timesteps < 2 or sample_shift <= 0:
        raise ValueError("invalid Wan schedule parameters")
    if sample_solver == "unipc":
        start = float(np.float32(1.0 - 1.0 / num_train_timesteps))
    elif sample_solver == "dpm++":
        start = 1.0
    else:
        raise ValueError(f"unsupported Wan solver: {sample_solver}")
    sigmas = np.linspace(start, 0, sample_steps + 1)[:-1]
    sigmas = sample_shift * sigmas / (1 + (sample_shift - 1) * sigmas)
    timesteps = (sigmas * num_train_timesteps).astype(np.int64)
    return validate_schedule(
        np.concatenate([sigmas, [0]]).astype(np.float32), timesteps,
    )


def make_wan_scheduler(generation, device, num_train_timesteps: int = 1000):
    """Use the installed Wan implementation for inference and closed-loop training."""
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    solver = str(generation.sample_solver)
    kwargs = dict(
        num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False,
    )
    if solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(**kwargs)
        scheduler.set_timesteps(
            int(generation.sample_steps), device=device,
            shift=float(generation.sample_shift),
        )
    elif solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(**kwargs)
        retrieve_timesteps(
            scheduler, device=device,
            sigmas=get_sampling_sigmas(
                int(generation.sample_steps), float(generation.sample_shift),
            ),
        )
    else:
        raise ValueError(f"unsupported Wan solver: {solver}")
    validate_schedule(scheduler.sigmas, scheduler.timesteps)
    return scheduler
