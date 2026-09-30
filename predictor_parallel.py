"""Single-node DDP, memory-based batch sizing and visible training progress."""

from __future__ import annotations

import copy
import logging
import math
import os
import threading
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch.utils.data import Sampler


LOGGER = logging.getLogger(__name__)


@dataclass
class Process:
    rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self):
        return self.rank == 0

    def minimum(self, value):
        tensor = torch.tensor(value, dtype=torch.int64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return int(tensor.item())

    def totals(self, values):
        tensor = torch.tensor(list(values.values()), dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor)
        result = dict(zip(values, tensor.tolist()))
        result["samples"] = int(result["samples"])
        return result

    def broadcast_model(self, model):
        if self.world_size > 1:
            for value in model.state_dict().values():
                dist.broadcast(value, src=0)


class RolloutGradients:
    """Synchronize at common solver nodes, independent of local cache decisions.

    Each finished local segment contributes its loss sum and prediction count.
    Empty ranks contribute zeros. One reduction averages over all actual
    prediction nodes before clipping and applying the same update on every GPU.
    """

    def __init__(self, model, optimizer, process, grad_clip=1.0):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer, self.process, self.grad_clip = optimizer, process, grad_clip
        self.count = 0
        self.optimizer_steps = 0
        self.optimizer.zero_grad(set_to_none=True)

    def backward(self, losses):
        torch.stack(losses).sum().backward()
        self.count += len(losses)

    def step(self):
        packed = torch.cat([
            p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
            for p in self.parameters
        ] + [self.parameters[0].new_tensor([self.count])])
        if self.process.world_size > 1:
            dist.all_reduce(packed)
        count = int(packed[-1].item())
        if count:
            if not torch.isfinite(packed).all():
                raise FloatingPointError("non-finite distributed rollout gradients")
            packed[:-1].div_(count)
            offset = 0
            for parameter in self.parameters:
                end = offset + parameter.numel()
                parameter.grad = packed[offset:end].view_as(parameter)
                offset = end
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
            self.optimizer.step()
            self.optimizer_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.count = 0
        return count


class RankSampler(Sampler):
    """Contiguous sample ranges preserve raw-file locality.

    Training pads at most world_size-1 samples so every rank takes the same
    number of optimizer steps. Validation never pads or repeats samples.
    """

    def __init__(self, size, rank=0, world_size=1, training=False):
        self.size, self.rank, self.world_size = size, rank, world_size
        if training:
            self.count = math.ceil(size / world_size)
            self.start = rank * self.count
        else:
            self.start = size * rank // world_size
            self.count = size * (rank + 1) // world_size - self.start

    def __len__(self):
        return self.count

    def __iter__(self):
        return ((self.start + index) % self.size for index in range(self.count))


class Progress:
    """Keep reporting the current phase even during a slow read or CUDA call."""

    def __init__(self, phase, enabled=True, interval=30):
        self.phase, self.enabled, self.interval = phase, enabled, interval
        self.started = self.last_log = time.monotonic()
        self.stopped = threading.Event()

    def __enter__(self):
        if self.enabled:
            self.log(self.phase)
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        return self

    def update(self, phase):
        self.phase = phase

    def log(self, message, *args):
        if self.enabled:
            LOGGER.info(message, *args)
            self.last_log = time.monotonic()

    def _heartbeat(self):
        while not self.stopped.wait(min(self.interval, 1.0)):
            if time.monotonic() - self.last_log >= self.interval:
                self.log("%s | elapsed %.0fs", self.phase, time.monotonic() - self.started)

    def __exit__(self, *args):
        self.stopped.set()
        if self.enabled:
            self.thread.join()


def process_count(cfg):
    if str(cfg.train.stage) == "validate":
        if str(cfg.parallel.num_processes) not in {"auto", "1"}:
            raise ValueError("standalone video validation currently requires parallel.num_processes=1")
        return 1
    requested = cfg.parallel.num_processes
    device = torch.device(str(cfg.device))
    available = torch.cuda.device_count() if device.type == "cuda" else 1
    count = (1 if device.index is not None else max(1, available)) if str(requested) == "auto" else int(requested)
    if device.type == "cuda" and device.index is not None and count > 1:
        raise ValueError("use device=cuda and CUDA_VISIBLE_DEVICES to select GPUs for DDP")
    if count < 1 or (device.type == "cuda" and count > available):
        raise ValueError(f"parallel.num_processes={count}, visible CUDA devices={available}")
    return count


def loader_workers(value, world_size):
    if str(value) != "auto":
        if int(value) < 0:
            raise ValueError("data.num_workers must be nonnegative or auto")
        return int(value)
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if hasattr(os, "sched_getaffinity"):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    return min(2, max(0, cpus // world_size - 1))


def search_batch_size(probe, maximum):
    """Choose a measured power of two; never infer capacity from parameter count."""
    best, candidate = 0, 1
    while candidate <= maximum:
        try:
            fits = probe(candidate)
        except torch.cuda.OutOfMemoryError:
            fits = False
        if not fits:
            break
        best = candidate
        candidate *= 2
    if not best:
        raise RuntimeError(
            "A complete latent at batch_size=1 exceeds the CUDA memory budget. "
            "Free GPU memory or increase data.auto_batch_memory_fraction; spatial cropping is disabled."
        )
    return best


def resolve_batch_size(cfg, model, criterion, shape, process, max_samples):
    requested = cfg.data.batch_size
    if str(requested) != "auto":
        if int(requested) < 1:
            raise ValueError("data.batch_size must be positive or auto")
        return int(requested)
    if process.device.type != "cuda":
        if process.primary:
            LOGGER.info("Automatic batch size: CPU fallback = 1")
        return 1
    fraction = float(cfg.data.auto_batch_memory_fraction)
    maximum = min(int(cfg.data.auto_batch_max), max_samples)
    if not 0 < fraction < 1 or maximum < 1:
        raise ValueError("auto batch needs 0 < memory_fraction < 1 and auto_batch_max >= 1")
    # Establish the communicator before measuring available CUDA memory.
    process.minimum(1)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(process.device)
    budget = free * fraction

    def trial(size):
        # Probe an independent copy, including Adam states, without updating
        # training parameters, optimizer, RNG or the raw-data iterator.
        probe_model = copy.deepcopy(model).train()
        optimizer = instantiate(cfg.optimizer, params=probe_model.parameters())
        x = torch.zeros((size, *shape), device=process.device)
        bases = torch.zeros((size, 3, *shape), device=process.device)
        q = torch.zeros(size, 5, device=process.device)
        target = torch.zeros_like(x)
        prediction, coefficients = probe_model.predict(x, bases, q)
        criterion(prediction, target, coefficients).backward()
        optimizer.step()
        torch.cuda.synchronize(process.device)

    with Progress(f"GPU {process.rank}: probing full latent {tuple(shape)} batch size", process.primary) as progress:
        def probe(size):
            progress.update(f"GPU {process.rank}: batch_size={size} forward/backward memory probe")
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated(process.device)
            torch.cuda.reset_peak_memory_stats(process.device)
            try:
                with torch.random.fork_rng(devices=[process.device]):
                    trial(size)
                peak = torch.cuda.max_memory_allocated(process.device) - baseline
                progress.log("Auto batch %d | peak additional %.2f GiB / budget %.2f GiB",
                             size, peak / 2**30, budget / 2**30)
                return peak <= budget
            except torch.cuda.OutOfMemoryError:
                progress.log("Auto batch %d | CUDA OOM; selecting the previous successful size", size)
                return False
            finally:
                torch.cuda.empty_cache()

        try:
            selected = search_batch_size(probe, maximum)
        except RuntimeError as error:
            LOGGER.error("GPU %d batch probe failed: %s", process.rank, error)
            selected = 0
        progress.update("Waiting for batch-size probes on the other GPUs")
        selected = process.minimum(selected)
        if selected == 0:
            raise RuntimeError("At least one GPU cannot fit batch_size=1 within the auto batch memory budget")
    if process.primary:
        LOGGER.info("Automatic batch size: %d per GPU, %d global, %.0f%% free-memory budget",
                    selected, selected * process.world_size, 100 * fraction)
    return selected
