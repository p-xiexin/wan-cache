"""SVD-Cache-inspired subspace diagnostics on raw Wan trajectories.

Reference: https://arxiv.org/html/2601.07396v1 (equations 5, 6, 9, 11).
This is an output-level diagnostic adaptation, not the original block cache.
Run inside analyze/: python analyze_raw_svd.py (no model weights or GPU required).
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .raw_analysis import cosine, divide, find_trajectories, load_trajectory, provenance, resolve_path, write_csv, write_json
else:
    from raw_analysis import cosine, divide, find_trajectories, load_trajectory, provenance, resolve_path, write_csv, write_json


# Configuration constants (edit here, or set the two path environment variables).
RAW_TRAJ_ROOT = os.environ.get("RAW_TRAJ_ROOT", "/path/to/raw_wan_trajectories")
OUTPUT_ROOT = os.environ.get("RAW_ANALYSIS_OUTPUT", "outputs/raw_analysis")
TARGET = "residual"  # "x", "v", "residual" (= v - x)
MAX_TRAJECTORIES = 3  # 0 = all
MAX_TOKENS = 4096  # 0 = all latent positions
SEED = 0
WARMUP_STEPS = 3  # Fit a fixed basis using these first steps only.
ENERGY_THRESHOLD = 0.90
RANK = None  # None = select by ENERGY_THRESHOLD; otherwise integer 1..C
EMA_BETA = 0.9  # State EMA from SVD-Cache eq. 11, not an EMA of derivatives.
HORIZONS = (1, 2, 4)  # Steps after an observed anchor; no intervening truth used.


def fit_basis(values, warmup_steps, energy_threshold, rank=None):
    if not 2 <= warmup_steps < len(values):
        raise ValueError("WARMUP_STEPS must be >=2 and smaller than the trajectory length")
    if not 0 < energy_threshold <= 1:
        raise ValueError("ENERGY_THRESHOLD must be in (0, 1]")
    channels = values.shape[-1]
    if rank is not None and (not isinstance(rank, int) or not 1 <= rank <= channels):
        raise ValueError("RANK must be None or an integer in 1..C")
    # Uncentered SVD via the small C x C Gram matrix; no future steps enter it.
    gram = np.zeros((channels, channels), dtype=np.float64)
    for frame in values[:warmup_steps]:
        frame = frame.astype(np.float64)
        gram += frame.T @ frame
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues, vectors = np.maximum(eigenvalues[::-1], 0), vectors[:, ::-1]
    total = eigenvalues.sum()
    if total <= 1e-20:
        raise ValueError("Warmup has zero energy; no meaningful SVD basis can be fitted")
    cumulative = np.cumsum(eigenvalues) / total
    selected = rank if rank is not None else min(channels, int(np.searchsorted(cumulative, energy_threshold)) + 1)
    return vectors[:, :selected], eigenvalues, cumulative


def forecast_errors(values, principal, timesteps, warmup_steps, beta, horizons):
    """Rolling-origin offline errors. Only steps <= anchor update EMA/history."""
    if not 0 <= beta < 1:
        raise ValueError("EMA_BETA must be in [0, 1)")
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("HORIZONS must contain positive integers")
    ema_full = values[0].copy()
    ema_principal = principal[0].copy()
    rows = []
    for i in range(len(values) - 1):
        if i:
            ema_full = beta * ema_full + (1 - beta) * values[i]
            ema_principal = beta * ema_principal + (1 - beta) * principal[i]
        if i < warmup_steps - 1:
            continue
        for h in sorted(set(horizons)):
            t = i + h
            if t >= len(values):
                continue
            ratio = (timesteps[t] - timesteps[i]) / (timesteps[i] - timesteps[i - 1])
            predictions = {
                "reuse": values[i],
                "full_linear": values[i] + ratio * (values[i] - values[i - 1]),
                "full_ema": ema_full,
                "principal_ema_reuse_tail": ema_principal + values[i] - principal[i],
                "principal_linear_reuse_tail": values[i] + ratio * (principal[i] - principal[i - 1]),
            }
            target_energy = np.mean(values[t] ** 2)
            for method, prediction in predictions.items():
                mse = np.mean((prediction - values[t]) ** 2)
                rows.append({"anchor_step": i, "target_step": t, "horizon": h,
                             "method": method, "mse": float(mse),
                             "target_mean_square": float(target_energy),
                             "nrmse": float(np.sqrt(divide(mse, target_energy)))})
    return rows


def analyze(series, output, source):
    output.mkdir(parents=True, exist_ok=True)
    values = series.values.astype(np.float64)
    basis, eigenvalues, cumulative = fit_basis(values, WARMUP_STEPS, ENERGY_THRESHOLD, RANK)
    principal = (values @ basis) @ basis.T
    tail = values - principal
    components = {"full": values, "principal": principal, "tail": tail}
    stats = {}
    for name, component in components.items():
        energy = np.sum(component ** 2, axis=(1, 2))
        delta = np.diff(component, axis=0)
        delta_energy = np.sum(delta ** 2, axis=(1, 2))
        stats[f"{name}_energy"] = energy
        stats[f"{name}_delta_energy"] = delta_energy
        stats[f"{name}_relative_delta"] = np.sqrt(divide(delta_energy, energy[:-1]))
        stats[f"{name}_direction_cosine"] = cosine(delta[1:], delta[:-1], axis=(1, 2))
    stats["principal_energy_fraction"] = divide(stats["principal_energy"], stats["full_energy"])
    stats["principal_delta_energy_fraction"] = divide(stats["principal_delta_energy"], stats["full_delta_energy"])
    np.savez_compressed(output / "statistics.npz", **stats, basis=basis,
                        singular_values=np.sqrt(eigenvalues), cumulative_energy=cumulative,
                        steps=series.steps, timesteps=series.timesteps, token_indices=series.token_indices)
    write_csv(output / "subspaces.csv", [
        {"step": int(step), "model_timestep": series.timesteps[t],
         "principal_energy_fraction": stats["principal_energy_fraction"][t],
         "principal_delta_energy_fraction": stats["principal_delta_energy_fraction"][t - 1] if t else np.nan,
         **{f"{name}_relative_delta": stats[f"{name}_relative_delta"][t - 1] if t else np.nan
            for name in components},
         **{f"{name}_direction_cosine": stats[f"{name}_direction_cosine"][t - 2] if t >= 2 else np.nan
            for name in components}}
        for t, step in enumerate(series.steps)
    ])
    rows = forecast_errors(values, principal, series.timesteps, WARMUP_STEPS, EMA_BETA, HORIZONS)
    if not rows:
        raise ValueError("No prediction targets remain; reduce WARMUP_STEPS or HORIZONS")
    write_csv(output / "prediction_errors.csv", rows)
    scores = []
    for h in sorted({r["horizon"] for r in rows}):
        baseline = sum(r["mse"] for r in rows if r["horizon"] == h and r["method"] == "reuse")
        for method in dict.fromkeys(r["method"] for r in rows):
            selected = [r for r in rows if r["horizon"] == h and r["method"] == method]
            error = sum(r["mse"] for r in selected)
            energy = sum(r["target_mean_square"] for r in selected)
            scores.append({"horizon": h, "method": method, "count": len(selected),
                           "nrmse": float(np.sqrt(divide(error, energy))),
                           "mse_relative_to_reuse": float(divide(error, baseline))})
    write_csv(output / "prediction_summary.csv", scores)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].plot(np.arange(1, len(cumulative) + 1), cumulative)
    axes[0, 0].axvline(basis.shape[1], color="tab:red", linestyle="--", label=f"rank={basis.shape[1]}")
    axes[0, 0].set(title="Warmup SVD energy", xlabel="Rank", ylabel="Cumulative energy", ylim=(0, 1.05))
    axes[0, 0].legend()
    axes[0, 1].plot(series.steps, stats["principal_energy_fraction"], label="Feature energy")
    axes[0, 1].plot(series.steps[1:], stats["principal_delta_energy_fraction"], label="Increment energy")
    axes[0, 1].axvline(WARMUP_STEPS - 0.5, color="gray", linestyle="--")
    axes[0, 1].set(title="Energy captured by fixed basis", xlabel="Denoising step", ylabel="Fraction", ylim=(0, 1.05))
    axes[0, 1].legend()
    for name in components:
        axes[0, 2].plot(series.steps[1:], stats[f"{name}_relative_delta"], label=name)
        axes[1, 0].plot(series.steps[2:], stats[f"{name}_direction_cosine"], label=name)
    axes[0, 2].set(title="Relative increment norm", xlabel="Denoising step", ylabel="||delta|| / ||previous||")
    axes[1, 0].set(title="Consecutive increment directions", xlabel="Denoising step", ylabel="Cosine", ylim=(-1.05, 1.05))
    axes[0, 2].legend()
    axes[1, 0].legend()
    for method in dict.fromkeys(r["method"] for r in rows):
        selected = [r for r in scores if r["method"] == method]
        axes[1, 1].plot([r["horizon"] for r in selected], [r["mse_relative_to_reuse"] for r in selected], marker="o", label=method)
    axes[1, 1].set(title="Held-out forecast error", xlabel="Prediction horizon (steps)", ylabel="MSE / reuse MSE (lower is better)")
    axes[1, 1].legend(fontsize=7)
    im = axes[1, 2].imshow(basis, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    axes[1, 2].set(title="Fixed channel basis", xlabel="Principal component", ylabel="Latent channel")
    fig.colorbar(im, ax=axes[1, 2], shrink=0.8)
    fig.suptitle(f"SVD-Cache-inspired diagnostics | {source.name} | {series.branch} | {TARGET}")
    fig.savefig(output / "subspaces.png", dpi=160)
    plt.close(fig)
    summary = provenance(source, series, TARGET, SEED)
    summary.update({
        "method": "SVD-Cache-inspired fixed warmup basis; state EMA + tail reuse; output-level adaptation",
        "reference": "https://arxiv.org/html/2601.07396v1",
        "warmup_steps": WARMUP_STEPS, "rank": basis.shape[1], "channels": values.shape[-1],
        "full_rank": basis.shape[1] == values.shape[-1], "energy_threshold": ENERGY_THRESHOLD,
        "ema_beta": EMA_BETA, "requested_horizons": HORIZONS, "prediction_scores": scores,
        "evaluation": "Rolling origins on full-compute trajectories; basis uses only first warmup steps. Each origin uses true history up to that origin, no intervening truth at horizons >1. No closed-loop quality/speed claim.",
        "basis_scope": "Fitted separately per trajectory and CFG branch; cross-prompt reuse is not tested.",
    })
    write_json(output / "summary.json", summary)
    return str(output)


def main():
    trajectories = find_trajectories(RAW_TRAJ_ROOT, MAX_TRAJECTORIES)
    output = resolve_path(OUTPUT_ROOT) / "svd" / TARGET
    outputs = []
    for source in trajectories:
        print(f"SVD analysis: {source}", flush=True)
        for series in load_trajectory(source, TARGET, MAX_TOKENS, SEED):
            folder = output / source.name / f"{series.branch}_item{series.item:02d}"
            outputs.append(analyze(series, folder, source))
    write_json(output / "index.json", {"source": str(resolve_path(RAW_TRAJ_ROOT)), "outputs": outputs})
    print(f"Saved SVD diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
