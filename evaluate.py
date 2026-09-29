"""Run performance and quality evaluation for one experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval import evaluate_quality, summarize


def write_table(
    experiment_root: Path,
    performance: list[dict],
    quality: list[dict],
) -> Path:
    quality_by_key = {
        (row["method"], row["cache_threshold"]): row
        for row in quality
    }
    performance = sorted(
        performance,
        key=lambda row: (
            row["method"] != "origin",
            row["method"],
            -1 if row["cache_threshold"] is None else row["cache_threshold"],
        ),
    )

    lines = [
        "<table>",
        "  <thead>",
        "    <tr>",
        '      <th rowspan="2">Method</th>',
        '      <th colspan="4">Efficiency</th>',
        '      <th colspan="4">Visual Quality Retention</th>',
        "    </tr>",
        "    <tr>",
        "      <th>Latency (s) ↓</th>",
        "      <th>Skip ↑</th>",
        "      <th>Speedup ↑</th>",
        "      <th>DiT total / video (s) ↓</th>",
        "      <th>PSNR ↑</th>",
        "      <th>SSIM ↑</th>",
        "      <th>LPIPS ↓</th>",
        "      <th>FVD ↓</th>",
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ]
    for row in performance:
        method = str(row["method"])
        threshold = row["cache_threshold"]
        display_method = {
            "easycache": "EasyCache",
            "magcache_output": "MagCache output",
            "d2cache_output": "D2Cache output",
        }.get(method, method.title())
        label = (
            display_method
            if threshold is None
            else f"{display_method} ({threshold:g})"
        )
        if method == "origin":
            psnr = ssim = lpips = fvd = "-"
        else:
            quality_row = quality_by_key[(method, threshold)]
            psnr = f"{quality_row['psnr_mean']:.4f}"
            ssim = f"{quality_row['ssim_mean']:.4f}"
            lpips = f"{quality_row['lpips_mean']:.4f}"
            fvd = f"{quality_row['fvd']:.4f}"
        lines.extend(
            [
                "    <tr>",
                f"      <td>{label}</td>",
                f"      <td>{row['latency_mean_seconds']:.4f}</td>",
                f"      <td>{row['skipped_pairs_mean']:.2f}</td>",
                f"      <td>{row['speedup_mean']:.4f}×</td>",
                f"      <td>{row['dit_forward_seconds_mean']:.4f}</td>",
                f"      <td>{psnr}</td>",
                f"      <td>{ssim}</td>",
                f"      <td>{lpips}</td>",
                f"      <td>{fvd}</td>",
                "    </tr>",
            ]
        )
    lines.extend(["  </tbody>", "</table>"])

    output_path = experiment_root / "metrics" / "evaluation_summary.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run(
    config_path: Path,
    overrides: list[str] | None = None,
) -> Path:
    config_path = config_path.expanduser().resolve()
    cfg = evaluate_quality.load_config(config_path, overrides)
    project_root = evaluate_quality._resolve(str(cfg.project_root), config_path.parent)
    experiment_root = evaluate_quality._resolve(str(cfg.experiment_root), project_root)
    output_path = experiment_root / "metrics" / "evaluation_summary.md"
    output_path.unlink(missing_ok=True)
    _, performance = summarize.run(experiment_root)
    _, quality = evaluate_quality.run(config_path, overrides)
    output_path = write_table(experiment_root, performance, quality)
    print(f"Combined evaluation table written to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run(args.config, args.overrides)


if __name__ == "__main__":
    main()
