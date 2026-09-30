"""Offline pretraining and short on-policy fine-tuning for PLAN.md."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from eval.artifacts import build_predictor_artifact, load_artifact
from eval.predictor_data import (
    RawPredictorDataset, find_raw_trajectories, iter_raw_pairs, read_metadata, split_by_prompt,
)
from eval.predictor_parallel import (
    Process, Progress, RankSampler, RolloutGradients, loader_workers, process_count, resolve_batch_size,
)


LOGGER = logging.getLogger(__name__)


def offline_epoch(model, criterion, loader, device, optimizer=None, grad_clip=1.0,
                  *, label="offline", log_every=10, primary=True):
    totals = dict(loss=0.0, mae=0.0, reuse_mae=0.0, quadratic_mae=0.0, samples=0)
    model.train(optimizer is not None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    read_seconds = compute_seconds = 0.0
    with torch.set_grad_enabled(optimizer is not None), Progress(f"{label}: waiting for first batch", primary) as progress:
        iterator = iter(loader)
        batch_index = 0
        while True:
            progress.update(f"{label} batch {batch_index + 1}/{len(loader)}: reading raw tensors"
                            if batch_index < len(loader) else f"{label}: finishing data workers")
            before = time.monotonic()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch_index += 1
            read_seconds += time.monotonic() - before
            progress.update(f"{label} batch {batch_index}/{len(loader)}: forward/backward")
            before = time.monotonic()
            x, bases, q, target = [
                batch[key].to(device, non_blocking=True) for key in ["x", "bases", "q", "target"]
            ]
            # Call DDP.forward so gradient synchronization is not bypassed by
            # the convenience model.predict method used during deployment.
            coefficients = model(x, bases, q)
            prediction = x.float() + (bases.float() * coefficients[:, :, None, None, None, None]).sum(1)
            loss = criterion(prediction, target, coefficients)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite predictor loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            count = x.shape[0]
            totals["samples"] += count
            totals["loss"] += float(loss.detach()) * count
            totals["mae"] += float((prediction.detach() - target).abs().mean()) * count
            totals["reuse_mae"] += float((x + bases[:, 0] - target).abs().mean()) * count
            totals["quadratic_mae"] += float((x + bases.sum(1) - target).abs().mean()) * count
            compute_seconds += time.monotonic() - before
            if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(loader):
                elapsed = time.monotonic() - started
                peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
                progress.log(
                    "%s | batch %d/%d | samples %d | loss %.6f | %.2f samples/s | "
                    "read %.1fs / compute %.1fs | peak %.2f GiB | ETA %.0fs",
                    label, batch_index, len(loader), totals["samples"], totals["loss"] / totals["samples"],
                    totals["samples"] / max(elapsed, 1e-9), read_seconds, compute_seconds, peak,
                    elapsed / batch_index * (len(loader) - batch_index),
                )
            # Release the previous full-volume batch before loading the next.
            del batch, x, bases, q, target, prediction, coefficients, loss
    return totals


def mean_metrics(totals):
    count = totals["samples"]
    if count == 0:
        raise ValueError("no prediction samples; check trajectory length or rollout skip settings")
    return {key: value / count for key, value in totals.items() if key != "samples"}


def closed_loop_epoch(cfg, generation, runtime, paths, model, criterion, process, optimizer=None, epoch=1):
    from eval.predictor_rollout import empty_rollout
    from eval.predictor_validation import run_trajectory

    training = optimizer is not None
    window = int(cfg.train.rollout_steps)
    if not 1 <= window <= 4:
        raise ValueError("train.rollout_steps must be between 1 and 4")
    model.train(training)
    selected = list(paths)
    if training:
        random.Random(int(cfg.seed) + epoch).shuffle(selected)
    local_paths = selected[process.rank::process.world_size]
    rounds = ((len(selected) + process.world_size - 1) // process.world_size
              if training else len(local_paths))
    gradients = RolloutGradients(model, optimizer, process, float(cfg.train.grad_clip)) if training else None
    totals = dict(loss=0.0, mae=0.0, reuse_mae=0.0, quadratic_mae=0.0, samples=0)
    drift_totals = dict(latent_mae_mean=0.0, latent_mae_max=0.0, latent_mae_final=0.0,
                        latent_rmse_final=0.0, samples=0)
    phase = f"Epoch {epoch} rollout {'train' if training else 'validation'}"
    with Progress(f"{phase}: preparing trajectories", process.primary) as progress:
        for index in range(rounds):
            if index >= len(local_paths):
                progress.update(f"{phase}: joining updates for the remaining trajectories")
                empty_rollout(int(generation.sample_steps), int(cfg.cache.warmup_steps), window, gradients)
                continue
            path = local_paths[index]
            label = f"{phase} trajectory {index + 1}/{rounds} ({path.name})"
            progress.update(f"{label}: loading latent and text conditions")
            progress.log(progress.phase)

            def on_progress(step, total):
                progress.update(f"{label}: node {step + 1}/{total}" if step < total else f"{label}: complete")
                if step == 0 or step == total or (step + 1) % int(cfg.train.log_every) == 0:
                    progress.log(progress.phase)

            values, _, _, drift = run_trajectory(
                cfg, generation, runtime, path, model, criterion,
                gradients=gradients, measure_drift=not training, on_progress=on_progress,
            )
            for key in totals:
                totals[key] += values[key]
            if drift is not None:
                for key, value in drift.metrics().items():
                    drift_totals[key] += value
                drift_totals["samples"] += 1
            progress.log("%s | predicted nodes %d", label, values["samples"] // 2)
        progress.update(f"{phase}: aggregating metrics across ranks")
        totals = process.totals(totals)
        metrics = (mean_metrics(totals) if training or totals["samples"]
                   else {key: None for key in totals if key != "samples"})
        if not training:
            drift_totals = process.totals(drift_totals)
            metrics.update(mean_metrics(drift_totals))
        else:
            metrics["optimizer_steps"] = gradients.optimizer_steps
    return totals, metrics


def fit_predictor(cfg, output_dir: Path):
    """Launch one training process per visible GPU for either training stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    external_world = int(os.environ.get("WORLD_SIZE", "1"))
    if external_world > 1:
        if str(cfg.train.stage) not in {"offline", "rollout"}:
            raise ValueError("multi-process training supports offline and rollout stages")
        if int(os.environ.get("LOCAL_WORLD_SIZE", external_world)) != external_world:
            raise ValueError("predictor launcher currently supports a single node")
        return _predictor_worker(int(os.environ["RANK"]), external_world, cfg, str(output_dir), "env://")
    count = process_count(cfg)
    if count == 1:
        return _fit_predictor(cfg, output_dir, Process(0, 1, torch.device(str(cfg.device))))
    LOGGER.info("Launching %s training on %d devices", cfg.train.stage, count)
    with tempfile.TemporaryDirectory(prefix="wan-predictor-ddp-") as rendezvous:
        mp.spawn(
            _predictor_worker,
            args=(count, OmegaConf.to_container(cfg, resolve=True), str(output_dir),
                  (Path(rendezvous) / "rendezvous").as_uri()),
            nprocs=count, join=True,
        )


def _predictor_worker(rank, world_size, config, output_dir, rendezvous):
    cfg = OmegaConf.create(config)
    local_rank = int(os.environ.get("LOCAL_RANK", rank)) if rendezvous == "env://" else rank
    device = torch.device(f"cuda:{local_rank}" if torch.device(str(cfg.device)).type == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, stream=sys.stdout,
                        format=f"[rank {rank}] %(asctime)s %(message)s", force=True)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", init_method=rendezvous,
                            rank=rank, world_size=world_size)
    try:
        _fit_predictor(cfg, Path(output_dir), Process(rank, world_size, device))
    finally:
        dist.destroy_process_group()


def _fit_predictor(cfg, output_dir: Path, process):
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(cfg.train.epochs) < 1:
        raise ValueError("train.epochs must be positive")
    if int(cfg.train.log_every) < 1:
        raise ValueError("train.log_every must be positive")
    stage = str(cfg.train.stage)
    if stage not in {"offline", "rollout", "validate"}:
        raise ValueError("train.stage must be offline, rollout or validate")
    paths, generation = find_raw_trajectories(cfg.data.data_dir)
    train_paths, val_paths = split_by_prompt(paths, float(cfg.data.val_ratio), int(cfg.seed))
    prompt_split = {
        name: sorted({read_metadata(p)["prompt"].strip() for p in selected})
        for name, selected in [("train", train_paths), ("validation", val_paths)]
    }
    device = process.device
    model = instantiate(cfg.model).to(device)
    criterion = instantiate(cfg.loss).to(device)
    model_config = dict(OmegaConf.to_container(cfg.model, resolve=True))
    if cfg.init_artifact is not None:
        saved = load_artifact(cfg.init_artifact, device)
        if saved.get("kind") != "residual_polynomial" or saved["model_config"] != model_config:
            raise ValueError("init_artifact architecture does not match cfg.model")
        model.load_state_dict(saved["model_state_dict"])
        summary_path = Path(str(cfg.init_artifact)).expanduser().parent / "summary.json"
        if summary_path.is_file():
            prior_split = json.loads(summary_path.read_text(encoding="utf-8")).get("prompt_split")
            if prior_split is not None and prior_split != prompt_split:
                raise ValueError("prompt split differs from init_artifact; reuse its data, seed and val_ratio")
    elif stage != "offline":
        raise ValueError("rollout and validate require a trained init_artifact")
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    def make_data(selected, shuffle):
        return RawPredictorDataset(
            selected, generation, int(cfg.model.channels), seed=int(cfg.seed),
            shuffle=shuffle, num_train_timesteps=int(cfg.data.num_train_timesteps),
        )

    if process.primary:
        LOGGER.info("Polynomial %s | %d train / %d validation trajectories | %d parameters | %d device(s)",
                    stage, len(train_paths), len(val_paths), sum(p.numel() for p in model.parameters()), process.world_size)
    train_data, val_data = make_data(train_paths, True), make_data(val_paths, False)
    training_model = model
    run_settings = {"world_size": process.world_size}
    if stage == "offline":
        train_sampler = RankSampler(len(train_data), process.rank, process.world_size, training=True)
        val_sampler = RankSampler(len(val_data), process.rank, process.world_size)
        with Progress("Reading latent shapes from raw shard metadata", process.primary):
            shape_info = [None]
            if process.primary:
                shape_info[0] = {tuple(next(iter_raw_pairs(path))[0]["model_input"][0].shape) for path in paths}
            if process.world_size > 1:
                dist.broadcast_object_list(shape_info, src=0, device=device)
            shapes = shape_info[0]
        if len(shapes) != 1:
            raise ValueError("offline batching requires a common latent shape; group raw data by resolution")
        shape = shapes.pop()
        if shape[0] != int(cfg.model.channels):
            raise ValueError(f"raw channels {shape[0]} != model channels {cfg.model.channels}")
        batch_size = resolve_batch_size(cfg, model, criterion, shape, process, len(train_sampler))
        workers = (0 if device.type == "cpu" and str(cfg.data.num_workers) == "auto"
                   else loader_workers(cfg.data.num_workers, process.world_size))
        loader_kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
        if workers:
            loader_kwargs.update(prefetch_factor=1, multiprocessing_context="spawn")
        train_loader = DataLoader(train_data, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_data, sampler=val_sampler, **loader_kwargs)
        if process.world_size > 1:
            training_model = DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        run_settings.update(batch_size_per_device=batch_size, global_batch_size=batch_size * process.world_size,
                            workers_per_device=workers,
                            train_padding_samples=len(train_sampler) * process.world_size - len(train_data))
        if process.primary:
            LOGGER.info("Offline loader | batch %d per device / %d global | workers %d per device | "
                        "train %d / validation %d samples | %d train batches per rank",
                        batch_size, batch_size * process.world_size, workers, len(train_data), len(val_data), len(train_loader))

    runtime = None
    if stage in {"rollout", "validate"}:
        from eval.pipeline import WanWorkerRuntime

        if device.type != "cuda" or generation.get("image") is not None:
            raise ValueError("Wan rollout/validation requires CUDA and text-to-video raw data")
        runtime_cfg = OmegaConf.create({
            "generation": OmegaConf.to_container(generation, resolve=True),
            "paths": {"checkpoint_dir": str(cfg.paths.checkpoint_dir)},
        })
        if stage == "rollout":
            process.broadcast_model(model)
            run_settings.update(
                trajectory_batch_size_per_device=1, global_trajectory_batch_size=process.world_size,
                gradient_sync_steps=int(cfg.train.rollout_steps), train_padding_samples=0,
            )
            if process.primary:
                LOGGER.info("Closed-loop training | 1 trajectory per GPU / up to %d global | "
                            "synchronize every %d solver nodes | no duplicated tail trajectories",
                            process.world_size, int(cfg.train.rollout_steps))
        runtime = WanWorkerRuntime(runtime_cfg, Path(str(cfg.project_root)).resolve(), device.index or 0)
        if stage == "validate":
            from eval.predictor_validation import validate_predictor

            validate_predictor(cfg, generation, runtime, val_paths, model, criterion, output_dir)
            LOGGER.info("Full trajectory validation: %s", output_dir / "validation.json")
            return
        with Progress("Loading frozen Wan teacher on each training GPU", process.primary):
            runtime.ensure_loaded()

    selection_metric = "loss" if stage == "offline" else "latent_mae_mean"
    best_score, best_epoch, stale = float("inf"), 0, 0
    artifact_path = output_dir / "model.pth"
    with ((output_dir / "history.csv").open("w", newline="", encoding="utf-8")
          if process.primary else open(os.devnull, "w")) as handle:
        writer = None
        for epoch in range(1, int(cfg.train.epochs) + 1):
            train_data.epoch = epoch
            if stage == "offline":
                train_totals = offline_epoch(
                    training_model, criterion, train_loader, device, optimizer, float(cfg.train.grad_clip),
                    label=f"Epoch {epoch} train", log_every=int(cfg.train.log_every), primary=process.primary,
                )
                val_totals = offline_epoch(
                    model, criterion, val_loader, device, label=f"Epoch {epoch} validation",
                    log_every=int(cfg.train.log_every), primary=process.primary,
                )
                with Progress(f"Epoch {epoch}: aggregating metrics across ranks", process.primary):
                    train_totals, val_totals = process.totals(train_totals), process.totals(val_totals)
                train_metrics, val_metrics = mean_metrics(train_totals), mean_metrics(val_totals)
            else:
                train_totals, train_metrics = closed_loop_epoch(
                    cfg, generation, runtime, train_paths, model, criterion, process, optimizer, epoch,
                )
                val_totals, val_metrics = closed_loop_epoch(
                    cfg, generation, runtime, val_paths, model, criterion, process, epoch=epoch,
                )
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if process.primary:
                LOGGER.info("Epoch %d | train loss %.6f | validation %s", epoch, train_metrics["loss"], val_metrics)
            if val_metrics[selection_metric] < best_score:
                best_score, best_epoch, stale = val_metrics[selection_metric], epoch, 0
                if not process.primary:
                    continue
                temporary = output_dir / "model.tmp.pth"
                torch.save(build_predictor_artifact(model, model_config, float(cfg.cache.threshold)), temporary)
                temporary.replace(artifact_path)
                (output_dir / "summary.json").write_text(json.dumps({
                    "training_config": OmegaConf.to_container(cfg, resolve=True),
                    "generation": OmegaConf.to_container(generation, resolve=True),
                    "prompt_split": prompt_split,
                    "best_epoch": best_epoch, "val_metrics": val_metrics,
                    "selection_metric": selection_metric,
                    "runtime": run_settings,
                    "train_trajectories": [str(p) for p in train_paths],
                    "val_trajectories": [str(p) for p in val_paths],
                    "train_samples": train_totals["samples"], "val_samples": val_totals["samples"],
                    "artifact": str(artifact_path),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                stale += 1
            if int(cfg.train.patience) > 0 and stale >= int(cfg.train.patience):
                break
    if process.primary:
        LOGGER.info("Best predictor: %s (epoch %d)", artifact_path, best_epoch)
