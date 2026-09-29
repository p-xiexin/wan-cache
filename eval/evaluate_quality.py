"""Compute paired image metrics and distribution-level FVD for videos."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf


@dataclass(frozen=True)
class VideoPair:
    pair_id: str
    origin_video: Path
    target_video: Path
    origin_index: int
    method: str
    cache_threshold: float | None


VIDEO_FIELDS = [
    "pair_id",
    "origin_index",
    "method",
    "cache_threshold",
    "origin_video",
    "target_video",
    "frame_count",
    "width",
    "height",
    "psnr",
    "ssim",
    "lpips",
]
SUMMARY_FIELDS = [
    "method",
    "cache_threshold",
    "videos",
    "psnr_mean",
    "ssim_mean",
    "lpips_mean",
    "fvd",
]


def _resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_config(config_path: Path, overrides: list[str] | None = None) -> Any:
    config_path = config_path.expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    if overrides:
        invalid = [value for value in overrides if "=" not in value]
        if invalid:
            raise ValueError(
                "evaluation overrides must use key=value syntax: "
                + " ".join(invalid)
            )
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def load_pairs(
    config_path: Path,
    preview: bool = False,
    overrides: list[str] | None = None,
) -> tuple[Any, list[VideoPair], Path]:
    config_path = config_path.expanduser().resolve()
    cfg = load_config(config_path, overrides)
    project_root = _resolve(str(cfg.project_root), config_path.parent)
    output_dir = _resolve(str(cfg.output_dir), project_root)

    if not OmegaConf.is_list(cfg.origins) or not OmegaConf.is_list(cfg.targets):
        raise ValueError("origins and targets must be lists")
    origins = [_resolve(str(path), project_root) for path in cfg.origins]
    if not origins:
        raise ValueError("origins is empty")

    groups = []
    for target in cfg.targets:
        if not OmegaConf.is_dict(target):
            raise ValueError("each target must be a mapping")
        method = str(target.method).strip()
        videos = list(target.videos)
        if not method:
            raise ValueError("target method is empty")
        if len(videos) != len(origins):
            raise ValueError("each target must provide one video per origin")
        threshold = target.get("cache_threshold")
        groups.append(
            (
                method,
                None if threshold is None else float(threshold),
                [_resolve(str(path), project_root) for path in videos],
            )
        )
    if not groups:
        raise ValueError("targets is empty")

    pairs = []
    for origin_index, origin in enumerate(origins):
        for method, threshold, videos in groups:
            pairs.append(
                VideoPair(
                    pair_id=str(len(pairs)),
                    origin_video=origin,
                    target_video=videos[origin_index],
                    origin_index=origin_index,
                    method=method,
                    cache_threshold=threshold,
                )
            )
    if not preview:
        for pair in pairs:
            for path in (pair.origin_video, pair.target_video):
                if not path.is_file():
                    raise FileNotFoundError(path)
    return cfg, pairs, output_dir


def _load_alexnet_weights(net: torch.nn.Module, path: Path) -> None:
    from torchvision.models import alexnet

    backbone = alexnet(weights=None)
    backbone.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    features = backbone.features
    for index, (start, stop) in enumerate(
        ((0, 2), (2, 5), (5, 8), (8, 10), (10, 12)),
        start=1,
    ):
        setattr(net, f"slice{index}", features[start:stop])
    net.requires_grad_(False)


class MetricComputer:
    def __init__(self, device: str, alexnet_path: Path) -> None:
        try:
            import lpips
            from torchmetrics.functional.image import (
                structural_similarity_index_measure,
            )
        except ImportError as error:
            raise RuntimeError("install lpips and torchmetrics") from error
        self.device = torch.device(device)
        self.ssim = structural_similarity_index_measure
        self.lpips = lpips.LPIPS(
            net="alex",
            spatial=True,
            pnet_rand=True,
        )
        _load_alexnet_weights(self.lpips.net, alexnet_path)
        self.lpips = self.lpips.to(self.device).eval()

    @torch.inference_mode()
    def __call__(
        self,
        origin_rgb: np.ndarray,
        target_rgb: np.ndarray,
    ) -> tuple[float, int, float, float]:
        origin = torch.from_numpy(origin_rgb).permute(0, 3, 1, 2)
        target = torch.from_numpy(target_rgb).permute(0, 3, 1, 2)
        origin = origin.to(self.device, dtype=torch.float32).div_(255)
        target = target.to(self.device, dtype=torch.float32).div_(255)
        squared_error = float((origin - target).square().sum().item())
        ssim_sum = sum(
            float(self.ssim(t[None], o[None], data_range=1.0).item())
            for o, t in zip(origin, target)
        )
        lpips_values = self.lpips(origin * 2 - 1, target * 2 - 1)
        lpips_sum = float(lpips_values.flatten(1).mean(1).sum().item())
        return squared_error, origin.numel(), ssim_sum, lpips_sum


class FVDComputer:
    """Extract canonical Kinetics-400 I3D features from fixed video clips."""

    def __init__(
        self,
        device: str,
        i3d_path: Path,
        num_frames: int,
        frame_stride: int,
    ) -> None:
        if num_frames < 1:
            raise ValueError("fvd_num_frames must be positive")
        if frame_stride < 1:
            raise ValueError("fvd_frame_stride must be positive")
        if not i3d_path.is_file():
            raise FileNotFoundError(i3d_path)
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.i3d = torch.jit.load(str(i3d_path), map_location=self.device)
        self.i3d = self.i3d.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, paths: list[Path]) -> np.ndarray:
        features = []
        for path in paths:
            clip = _read_fvd_clip(
                path,
                num_frames=self.num_frames,
                frame_stride=self.frame_stride,
            )
            video = (
                torch.from_numpy(clip)
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
                .contiguous()
            )
            feature = self.i3d(
                video.to(self.device),
                rescale=True,
                resize=True,
                return_features=True,
            )
            features.append(feature.reshape(feature.shape[0], -1).cpu().numpy())
        return np.concatenate(features, axis=0)


def _read_fvd_clip(
    path: Path,
    num_frames: int,
    frame_stride: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open FVD video {path}")
    selected = []
    wanted = {index * frame_stride for index in range(num_frames)}
    last_index = (num_frames - 1) * frame_stride
    try:
        for index in range(last_index + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(selected) != num_frames:
        required = last_index + 1
        raise ValueError(
            f"FVD video {path} has fewer than {required} decodable frames"
        )
    return np.stack(selected)


def frechet_distance(
    origin_features: np.ndarray,
    target_features: np.ndarray,
) -> float:
    origin_features = np.asarray(origin_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    if origin_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("FVD features must be rank-2 arrays")
    if origin_features.shape[1] != target_features.shape[1]:
        raise ValueError("FVD feature dimensions do not match")
    if min(len(origin_features), len(target_features)) < 2:
        raise ValueError("FVD requires at least two videos in each distribution")

    origin_mean = origin_features.mean(axis=0)
    target_mean = target_features.mean(axis=0)
    origin_centered = origin_features - origin_mean
    target_centered = target_features - target_mean
    origin_factor = origin_centered.T / math.sqrt(len(origin_features) - 1)
    target_factor = target_centered.T / math.sqrt(len(target_features) - 1)
    covariance_overlap = np.linalg.svd(
        origin_factor.T @ target_factor,
        compute_uv=False,
    ).sum()
    mean_distance = np.square(origin_mean - target_mean).sum()
    distance = (
        mean_distance
        + np.square(origin_factor).sum()
        + np.square(target_factor).sum()
        - 2 * covariance_overlap
    )
    return max(float(distance), 0.0)


def evaluate_fvd(
    pairs: list[VideoPair],
    computer: FVDComputer,
) -> dict[tuple[str, float | None], float]:
    groups: dict[tuple[str, float | None], list[VideoPair]] = defaultdict(list)
    for pair in pairs:
        groups[(pair.method, pair.cache_threshold)].append(pair)

    paths = list(
        dict.fromkeys(
            [pair.origin_video for pair in pairs]
            + [pair.target_video for pair in pairs]
        )
    )
    extracted = computer(paths)
    if extracted.ndim != 2 or len(extracted) != len(paths):
        raise ValueError("FVD extractor must return one feature vector per video")
    features = {path: feature for path, feature in zip(paths, extracted)}
    scores = {}
    for key, group in groups.items():
        group = sorted(group, key=lambda pair: pair.origin_index)
        origin = np.stack([features[pair.origin_video] for pair in group])
        target = np.stack([features[pair.target_video] for pair in group])
        scores[key] = frechet_distance(origin, target)
    return scores


def _read_batch(
    origin: cv2.VideoCapture,
    target: cv2.VideoCapture,
    batch_size: int,
    pair_id: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    origin_frames, target_frames = [], []
    for _ in range(batch_size):
        origin_ok, origin_frame = origin.read()
        target_ok, target_frame = target.read()
        if origin_ok != target_ok:
            raise ValueError(f"frame count mismatch in pair {pair_id}")
        if not origin_ok:
            break
        if origin_frame.shape != target_frame.shape:
            raise ValueError(f"resolution mismatch in pair {pair_id}")
        origin_frames.append(cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB))
        target_frames.append(cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB))
    if not origin_frames:
        return None, None
    return np.stack(origin_frames), np.stack(target_frames)


def evaluate_pair(
    pair: VideoPair,
    computer: MetricComputer,
    batch_size: int,
) -> dict[str, Any]:
    origin = cv2.VideoCapture(str(pair.origin_video))
    target = cv2.VideoCapture(str(pair.target_video))
    if not origin.isOpened() or not target.isOpened():
        raise RuntimeError(f"cannot open pair {pair.pair_id}")

    squared_error = value_count = frame_count = 0
    ssim_sum = lpips_sum = 0.0
    width = height = 0
    try:
        while True:
            origin_batch, target_batch = _read_batch(
                origin, target, batch_size, pair.pair_id
            )
            if origin_batch is None:
                break
            height, width = origin_batch.shape[1:3]
            error, values, ssim, lpips = computer(origin_batch, target_batch)
            squared_error += error
            value_count += values
            ssim_sum += ssim
            lpips_sum += lpips
            frame_count += len(origin_batch)
    finally:
        origin.release()
        target.release()
    if frame_count == 0:
        raise RuntimeError(f"no frames decoded in pair {pair.pair_id}")

    mse = squared_error / value_count
    return {
        "pair_id": pair.pair_id,
        "origin_index": pair.origin_index,
        "method": pair.method,
        "cache_threshold": pair.cache_threshold,
        "origin_video": str(pair.origin_video),
        "target_video": str(pair.target_video),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "psnr": math.inf if mse == 0 else -10 * math.log10(mse),
        "ssim": ssim_sum / frame_count,
        "lpips": lpips_sum / frame_count,
    }


def summarize(
    rows: list[dict[str, Any]],
    fvd_scores: dict[tuple[str, float | None], float] | None = None,
) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["cache_threshold"])].append(row)
    summaries = []
    for (method, threshold), group in groups.items():
        key = (method, threshold)
        summaries.append(
            {
                "method": method,
                "cache_threshold": threshold,
                "videos": len(group),
                **{
                    f"{metric}_mean": float(np.mean([row[metric] for row in group]))
                    for metric in ("psnr", "ssim", "lpips")
                },
                "fvd": None if fvd_scores is None else fvd_scores[key],
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    config_path: Path,
    overrides: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg, pairs, output_dir = load_pairs(config_path, overrides=overrides)
    config_path = config_path.expanduser().resolve()
    project_root = _resolve(str(cfg.project_root), config_path.parent)
    alexnet_path = _resolve(str(cfg.alexnet_path), project_root)
    i3d_path = _resolve(str(cfg.i3d_path), project_root)
    computer = MetricComputer(str(cfg.device), alexnet_path)
    rows = []
    for pair in pairs:
        row = evaluate_pair(pair, computer, int(cfg.batch_size))
        rows.append(row)
        print(
            f"{pair.pair_id} {pair.method} threshold={pair.cache_threshold} "
            f"PSNR={row['psnr']:.4f} SSIM={row['ssim']:.4f} "
            f"LPIPS={row['lpips']:.4f}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fvd_computer = FVDComputer(
        str(cfg.device),
        i3d_path,
        int(cfg.fvd_num_frames),
        int(cfg.fvd_frame_stride),
    )
    fvd_scores = evaluate_fvd(pairs, fvd_computer)
    summaries = summarize(rows, fvd_scores)
    for row in summaries:
        print(
            f"{row['method']} threshold={row['cache_threshold']} "
            f"FVD={row['fvd']:.4f}"
        )
    _write_csv(output_dir / "quality_per_video.csv", rows, VIDEO_FIELDS)
    _write_csv(output_dir / "quality_summary.csv", summaries, SUMMARY_FIELDS)
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.check_config:
        _, pairs, _ = load_pairs(
            args.config,
            preview=True,
            overrides=args.overrides,
        )
        print(f"{len(pairs)} video pairs")
        for pair in pairs:
            print(
                pair.pair_id,
                pair.method,
                pair.cache_threshold,
                pair.origin_video,
                pair.target_video,
            )
        return
    run(args.config, args.overrides)


if __name__ == "__main__":
    main()
