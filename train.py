"""Hydra training entrypoint for the learned cache model."""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from eval.artifacts import build_artifact
from eval.dataloader import DataBundle, TrajectoryDataModule


@dataclass
class Metrics:
    loss: float
    mae: float
    rmse: float
    bias: float
    underestimation_rate: float
    prefix_mae: float
    prefix_exact_accuracy: float
    unsafe_over_skip_rate: float
    conservative_rate: float
    mean_predicted_prefix: float
    mean_true_prefix: float


def safe_prefix_length(
    cumulative_risk: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    safe = mask & (cumulative_risk < threshold)
    return safe.to(torch.long).cumprod(dim=1).sum(dim=1)


def prefix_decision_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    predicted_prefix = safe_prefix_length(prediction, mask, threshold)
    true_prefix = safe_prefix_length(target, mask, threshold)
    error = predicted_prefix.float() - true_prefix.float()
    return {
        "prefix_mae": error.abs().mean().item(),
        "prefix_exact_accuracy": (predicted_prefix == true_prefix).float().mean().item(),
        "unsafe_over_skip_rate": (predicted_prefix > true_prefix).float().mean().item(),
        "conservative_rate": (predicted_prefix < true_prefix).float().mean().item(),
        "mean_predicted_prefix": predicted_prefix.float().mean().item(),
        "mean_true_prefix": true_prefix.float().mean().item(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Metrics:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for features, target, mask in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        prediction = model(features)
        loss = criterion(prediction, target, mask)
        count = target.shape[0]
        total_loss += float(loss.item()) * count
        total_samples += count
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        masks.append(mask.cpu())

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    valid = mask.float()
    valid_count = valid.sum().clamp_min(1.0)
    error = prediction - target
    prefixes = prefix_decision_metrics(
        prediction,
        target,
        mask,
        threshold,
    )
    return Metrics(
        loss=total_loss / max(total_samples, 1),
        mae=((error.abs() * valid).sum() / valid_count).item(),
        rmse=((error.square() * valid).sum() / valid_count).sqrt().item(),
        bias=((error * valid).sum() / valid_count).item(),
        underestimation_rate=(
            ((prediction < target) & mask).float().sum() / valid_count
        ).item(),
        **prefixes,
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for features, target, mask in loader:
        target = target.to(device, non_blocking=True)
        prediction = model(features.to(device, non_blocking=True))
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        masks.append(mask)
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def calibrate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    coverage: float,
) -> torch.Tensor:
    offsets = []
    for horizon_index in range(prediction.shape[1]):
        valid = mask[:, horizon_index].to(torch.bool)
        residual = target[valid, horizon_index] - prediction[valid, horizon_index]
        if residual.numel() == 0:
            offsets.append(prediction.new_zeros(()))
            continue
        ordered = residual.sort().values
        rank = math.ceil((ordered.numel() + 1) * coverage) - 1
        offsets.append(ordered[min(rank, ordered.numel() - 1)].clamp_min(0.0))
    return torch.cummax(torch.stack(offsets), dim=0).values


def append_history(
    path: Path,
    epoch: int,
    lr: float,
    train_metrics: Metrics,
    val_metrics: Metrics,
) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(
                ["epoch", "lr"]
                + [f"train_{key}" for key in asdict(train_metrics)]
                + [f"val_{key}" for key in asdict(val_metrics)]
            )
        writer.writerow(
            [epoch, lr]
            + list(asdict(train_metrics).values())
            + list(asdict(val_metrics).values())
        )


def fit(
    cfg: DictConfig,
    model: nn.Module,
    criterion: nn.Module,
    data_module: TrajectoryDataModule,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(cfg.device))
    model = model.to(device)
    criterion = criterion.to(device)
    training_config = OmegaConf.to_container(cfg, resolve=True)
    model_config = dict(OmegaConf.to_container(cfg.model, resolve=True))
    data: DataBundle = data_module.setup(
        raw_feature_dim=int(model.input_dim),
        horizon=int(model.horizon),
    )
    threshold = float(cfg.cache.threshold)

    logging.info(
        "Train %s on %s | %d trajectories | %d train windows | "
        "%d val windows | %d parameters",
        type(model).__name__,
        device,
        data.trajectory_count,
        data.train_window_count,
        data.val_window_count,
        sum(parameter.numel() for parameter in model.parameters()),
    )

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    history_path = output_dir / "history.csv"
    history_path.unlink(missing_ok=True)
    artifact_path = output_dir / "model.pth"
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    best_val_loss = float("inf")
    stale_epochs = 0
    best_state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        for features, target, mask in data.train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            prediction = model(features)
            loss = criterion(prediction, target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.train.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg.train.grad_clip),
                )
            optimizer.step()

        train_metrics = evaluate(
            model,
            criterion,
            data.train_loader,
            device,
            threshold,
        )
        val_metrics = evaluate(
            model,
            criterion,
            data.val_loader,
            device,
            threshold,
        )
        scheduler.step(val_metrics.loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        append_history(
            history_path,
            epoch,
            current_lr,
            train_metrics,
            val_metrics,
        )
        logging.info(
            "Epoch %04d | lr=%.2e | train loss=%.6f | val loss=%.6f | "
            "prefix MAE=%.4f | unsafe over-skip=%.4f",
            epoch,
            current_lr,
            train_metrics.loss,
            val_metrics.loss,
            val_metrics.prefix_mae,
            val_metrics.unsafe_over_skip_rate,
        )

        selection_key = (
            max(
                val_metrics.unsafe_over_skip_rate
                - float(cfg.cache.max_unsafe_rate),
                0.0,
            ),
            -val_metrics.mean_predicted_prefix,
            val_metrics.loss,
        )
        if best_key is None or selection_key < best_key:
            best_key = selection_key
            best_epoch = epoch
            best_val_loss = val_metrics.loss
            stale_epochs = 0
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1

        patience = int(cfg.train.patience)
        if patience > 0 and stale_epochs >= patience:
            break

    model.load_state_dict(best_state_dict)
    final_model = model
    final_metrics = evaluate(
        final_model,
        criterion,
        data.val_loader,
        device,
        threshold,
    )
    prediction, target, mask = collect_predictions(
        final_model,
        data.val_loader,
        device,
    )
    calibration_offsets = calibrate(
        prediction,
        target,
        mask,
        float(cfg.cache.calibration_coverage),
    )
    calibrated = torch.cummax(
        prediction + calibration_offsets.unsqueeze(0),
        dim=1,
    ).values
    calibrated_prefix_metrics = prefix_decision_metrics(
        calibrated,
        target,
        mask,
        threshold,
    )
    final_artifact = build_artifact(
        final_model,
        model_config,
        data.feature_mean,
        data.feature_std,
        calibration_offsets,
        threshold,
    )
    torch.save(final_artifact, artifact_path)

    valid = mask.float()
    valid_count = valid.sum(dim=0)
    error = prediction - target
    denominator = valid_count.clamp_min(1.0)
    per_horizon = {
        "mae": ((error.abs() * valid).sum(dim=0) / denominator).tolist(),
        "rmse": (
            (error.square() * valid).sum(dim=0) / denominator
        ).sqrt().tolist(),
        "valid_count": valid_count.long().tolist(),
    }

    summary = {
        "training_config": training_config,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_metrics": asdict(final_metrics),
        "per_horizon_val_metrics": per_horizon,
        "calibration_offsets": calibration_offsets.detach().cpu().tolist(),
        "calibrated_prefix_metrics": calibrated_prefix_metrics,
        "trajectory_files": data.trajectory_count,
        "train_windows": data.train_window_count,
        "val_windows": data.val_window_count,
        "artifact": str(artifact_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info("Deployment artifact: %s", artifact_path)


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    output_dir = Path(str(cfg.output_dir)).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(get_original_cwd()) / output_dir
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = instantiate(cfg.model)
    criterion = instantiate(cfg.loss)
    data_module = instantiate(cfg.data)
    fit(cfg, model, criterion, data_module, output_dir.resolve())


if __name__ == "__main__":
    main()
