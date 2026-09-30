"""ICC-inspired channel diagnostics on raw trajectories, not ICC reproduction.

Reference: https://github.com/ccccczzy/icc
Run inside analyze/: python analyze_raw_icc.py (no model weights or GPU required).
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
MAX_TRAJECTORIES = 3  # 0 = all; start small
MAX_TOKENS = 4096  # 0 = all latent positions; fixed positions across steps/CFG
SEED = 0


def channel_statistics(values: np.ndarray, timesteps: np.ndarray) -> dict:
    values = values.astype(np.float64)
    delta = np.diff(values, axis=0)
    rms = np.sqrt(np.mean(values ** 2, axis=1))
    delta_rms = np.sqrt(np.mean(delta ** 2, axis=1))
    flat_delta = delta.reshape(-1, delta.shape[-1])
    centered = flat_delta - flat_delta.mean(axis=0)
    gram = centered.T @ centered
    norms = np.sqrt(np.diag(gram))
    correlation = divide(gram, norms[:, None] * norms[None, :])
    return {
        "mean_abs": np.mean(np.abs(values), axis=1),
        "rms": rms,
        "abs_p99_over_rms": divide(np.quantile(np.abs(values), 0.99, axis=1), rms),
        "delta_mean_abs": np.mean(np.abs(delta), axis=1),
        "delta_rms": delta_rms,
        "relative_delta_rms": divide(delta_rms, rms[:-1]),
        "slope_rms": delta_rms / np.abs(np.diff(timesteps))[:, None],
        "delta_direction_cosine": cosine(delta[1:], delta[:-1], axis=1),
        "delta_correlation": np.clip(correlation, -1, 1),
        "delta_energy_share": divide(np.sum(delta ** 2, axis=(0, 1)), np.sum(delta ** 2)),
    }


def analyze(series, output, source):
    output.mkdir(parents=True, exist_ok=True)
    stats = channel_statistics(series.values, series.timesteps)
    np.savez_compressed(output / "statistics.npz", **stats, steps=series.steps,
                        timesteps=series.timesteps, token_indices=series.token_indices)
    rows = []
    for t, step in enumerate(series.steps):
        for c in range(series.values.shape[-1]):
            rows.append({
                "step": int(step), "model_timestep": series.timesteps[t], "channel": c,
                **{name: stats[name][t, c] for name in ("mean_abs", "rms", "abs_p99_over_rms")},
                **{name: stats[name][t - 1, c] if t else np.nan for name in
                   ("delta_mean_abs", "delta_rms", "relative_delta_rms", "slope_rms")},
                "delta_direction_cosine": stats["delta_direction_cosine"][t - 2, c] if t >= 2 else np.nan,
            })
    write_csv(output / "per_step_channel.csv", rows)
    write_csv(output / "channels.csv", [
        {"channel": c, "mean_abs": stats["mean_abs"][:, c].mean(),
         "delta_mean_abs": stats["delta_mean_abs"][:, c].mean(),
         "delta_energy_share": stats["delta_energy_share"][c]}
        for c in range(series.values.shape[-1])
    ])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, name, title, start in zip(axes.flat[:4],
            ("rms", "delta_rms", "relative_delta_rms", "slope_rms"),
            ("Channel RMS", "Increment RMS", "Increment / previous RMS", "Increment RMS / |timestep gap|"),
            (0, 1, 1, 1)):
        data = stats[name]
        im = ax.imshow(data.T, aspect="auto", origin="lower",
                       extent=(start - 0.5, len(series.steps) - 0.5, -0.5, data.shape[1] - 0.5))
        ax.set(title=title, xlabel="Denoising step", ylabel="Latent channel")
        fig.colorbar(im, ax=ax, shrink=0.8)
    im = axes[1, 1].imshow(stats["delta_correlation"], vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
    axes[1, 1].set(title="Increment channel correlation", xlabel="Latent channel", ylabel="Latent channel")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
    axes[1, 2].bar(np.arange(series.values.shape[-1]), stats["delta_energy_share"])
    axes[1, 2].set(title="Share of total increment energy", xlabel="Latent channel", ylabel="Energy fraction")
    fig.suptitle(f"ICC-inspired diagnostics | {source.name} | {series.branch} | {TARGET}")
    fig.savefig(output / "channels.png", dpi=160)
    plt.close(fig)
    summary = provenance(source, series, TARGET, SEED)
    summary.update({
        "method": "ICC-inspired activation/delta channel statistics; no weight SVD or ICC calibration",
        "reference": "https://github.com/ccccczzy/icc",
        "top_delta_channels": [int(c) for c in np.argsort(-stats["delta_energy_share"])
                               if np.isfinite(stats["delta_energy_share"][c]) and stats["delta_energy_share"][c] > 0][:5],
        "delta_energy_share": stats["delta_energy_share"],
        "mean_abs_per_channel": stats["mean_abs"].mean(axis=0),
        "delta_mean_abs_per_channel": stats["delta_mean_abs"].mean(axis=0),
    })
    write_json(output / "summary.json", summary)
    return str(output)


def main():
    trajectories = find_trajectories(RAW_TRAJ_ROOT, MAX_TRAJECTORIES)
    output = resolve_path(OUTPUT_ROOT) / "icc" / TARGET
    outputs = []
    for source in trajectories:
        print(f"ICC analysis: {source}", flush=True)
        for series in load_trajectory(source, TARGET, MAX_TOKENS, SEED):
            folder = output / source.name / f"{series.branch}_item{series.item:02d}"
            outputs.append(analyze(series, folder, source))
    write_json(output / "index.json", {"source": str(resolve_path(RAW_TRAJ_ROOT)), "outputs": outputs})
    print(f"Saved ICC diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
