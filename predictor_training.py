"""Offline pretraining and short on-policy fine-tuning for PLAN.md."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from eval.artifacts import build_predictor_artifact, load_artifact
from eval.predictor_data import (
    RawPredictorDataset, find_raw_trajectories, read_metadata, split_by_prompt,
)


LOGGER = logging.getLogger(__name__)


def offline_epoch(model, criterion, loader, device, optimizer=None, grad_clip=1.0):
    totals = dict(loss=0.0, mae=0.0, reuse_mae=0.0, quadratic_mae=0.0, samples=0)
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            x, bases, q, target = [
                batch[key].to(device) for key in ["x", "bases", "q", "target"]
            ]
            prediction, coefficients = model.predict(x, bases, q)
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
    return totals


def mean_metrics(totals):
    count = totals["samples"]
    if count == 0:
        raise ValueError("no prediction samples; check trajectory length or rollout skip settings")
    return {key: value / count for key, value in totals.items() if key != "samples"}


def fit_predictor(cfg, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(cfg.train.epochs) < 1:
        raise ValueError("train.epochs must be positive")
    stage = str(cfg.train.stage)
    if stage not in {"offline", "rollout", "validate"}:
        raise ValueError("train.stage must be offline, rollout or validate")
    paths, generation = find_raw_trajectories(cfg.data.data_dir)
    train_paths, val_paths = split_by_prompt(paths, float(cfg.data.val_ratio), int(cfg.seed))
    prompt_split = {
        name: sorted({read_metadata(p)["prompt"].strip() for p in selected})
        for name, selected in [("train", train_paths), ("validation", val_paths)]
    }
    device = torch.device(str(cfg.device))
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

    train_data, val_data = make_data(train_paths, True), make_data(val_paths, False)
    loader_kwargs = dict(batch_size=int(cfg.data.batch_size), num_workers=int(cfg.data.num_workers))
    train_loader, val_loader = DataLoader(train_data, **loader_kwargs), DataLoader(val_data, **loader_kwargs)

    runtime = None
    if stage in {"rollout", "validate"}:
        from eval.pipeline import WanWorkerRuntime

        if device.type != "cuda" or generation.get("image") is not None:
            raise ValueError("Wan rollout/validation requires CUDA and text-to-video raw data")
        runtime_cfg = OmegaConf.create({
            "generation": OmegaConf.to_container(generation, resolve=True),
            "paths": {"checkpoint_dir": str(cfg.paths.checkpoint_dir)},
        })
        runtime = WanWorkerRuntime(runtime_cfg, Path(str(cfg.project_root)).resolve(), device.index or 0)
        if stage == "validate":
            from eval.predictor_validation import validate_predictor

            validate_predictor(cfg, generation, runtime, val_paths, model, criterion, output_dir)
            LOGGER.info("Full trajectory validation: %s", output_dir / "validation.json")
            return
        runtime.ensure_loaded()

    def rollout_epoch(selected, training):
        from eval.predictor_validation import run_trajectory

        model.train(training)
        totals = dict(loss=0.0, mae=0.0, reuse_mae=0.0, quadratic_mae=0.0, samples=0)
        drift_metrics = []
        for path in selected:
            values, _, _, drift = run_trajectory(
                cfg, generation, runtime, path, model, criterion,
                optimizer=optimizer if training else None, measure_drift=not training,
            )
            if drift is not None:
                drift_metrics.append(drift.metrics())
            for key in totals:
                totals[key] += values[key]
        if training or totals["samples"]:
            metrics = mean_metrics(totals)
        else:
            metrics = {key: None for key in totals if key != "samples"}
        if drift_metrics:
            metrics.update({
                key: sum(row[key] for row in drift_metrics) / len(drift_metrics)
                for key in drift_metrics[0]
            })
        return totals, metrics

    selection_metric = "loss" if stage == "offline" else "latent_mae_mean"
    best_score, best_epoch, stale = float("inf"), 0, 0
    artifact_path = output_dir / "model.pth"
    LOGGER.info(
        "Polynomial %s | %d train / %d validation trajectories | %d parameters",
        stage, len(train_paths), len(val_paths), sum(p.numel() for p in model.parameters()),
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, int(cfg.train.epochs) + 1):
            train_data.epoch = epoch
            if stage == "offline":
                train_totals = offline_epoch(
                    model, criterion, train_loader, device, optimizer, float(cfg.train.grad_clip),
                )
                val_totals = offline_epoch(model, criterion, val_loader, device)
                train_metrics, val_metrics = mean_metrics(train_totals), mean_metrics(val_totals)
            else:
                train_totals, train_metrics = rollout_epoch(train_paths, True)
                val_totals, val_metrics = rollout_epoch(val_paths, False)
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
            LOGGER.info(
                "Epoch %d | train loss %.6f | validation %s",
                epoch, train_metrics["loss"], val_metrics,
            )
            if val_metrics[selection_metric] < best_score:
                best_score, best_epoch, stale = val_metrics[selection_metric], epoch, 0
                temporary = output_dir / "model.tmp.pth"
                torch.save(build_predictor_artifact(model, model_config, float(cfg.cache.threshold)), temporary)
                temporary.replace(artifact_path)
                (output_dir / "summary.json").write_text(json.dumps({
                    "training_config": OmegaConf.to_container(cfg, resolve=True),
                    "generation": OmegaConf.to_container(generation, resolve=True),
                    "prompt_split": prompt_split,
                    "best_epoch": best_epoch, "val_metrics": val_metrics,
                    "selection_metric": selection_metric,
                    "train_trajectories": [str(p) for p in train_paths],
                    "val_trajectories": [str(p) for p in val_paths],
                    "train_samples": train_totals["samples"], "val_samples": val_totals["samples"],
                    "artifact": str(artifact_path),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                stale += 1
            if int(cfg.train.patience) > 0 and stale >= int(cfg.train.patience):
                break
    LOGGER.info("Best predictor: %s (epoch %d)", artifact_path, best_epoch)
