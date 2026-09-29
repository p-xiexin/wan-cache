"""Summarize generation, DiT, cache, and skip metrics for one experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PER_RUN_FIELDS = [
    "run_id",
    "prompt_id",
    "method",
    "cache_threshold",
    "generation_seconds",
    "origin_generation_seconds",
    "speedup",
    "dit_forward_seconds",
    "dit_forward_calls",
    "dit_forward_mean_milliseconds",
    "cache_forward_seconds",
    "cache_forward_calls",
    "cache_forward_mean_milliseconds",
    "dit_to_cache_speedup",
    "skipped_pairs",
    "skip_ratio",
    "skipped_pair_indices",
]

SUMMARY_FIELDS = [
    "method",
    "cache_threshold",
    "videos",
    "latency_mean_seconds",
    "dit_forward_seconds_total",
    "dit_forward_seconds_mean",
    "dit_forward_calls_mean",
    "cache_forward_seconds_mean",
    "cache_forward_calls_mean",
    "dit_forward_mean_milliseconds",
    "cache_forward_mean_milliseconds",
    "dit_to_cache_speedup",
    "skipped_pairs_mean",
    "skip_ratio_mean",
    "speedup_mean",
]


def load_results(output_root: Path) -> list[dict[str, Any]]:
    paths = sorted(output_root.glob("*/*.result.json"))
    if not paths:
        raise FileNotFoundError(f"no result files found under {output_root}")

    results = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(f"incomplete result at {path}")
        results.append(result)
    return results


def build_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    origins = {}
    for result in results:
        if result["method"] == "origin":
            prompt_id = str(result["prompt_id"])
            if prompt_id in origins:
                raise ValueError(f"duplicate Origin for prompt {prompt_id}")
            origins[prompt_id] = result

    rows = []
    for result in results:
        prompt_id = str(result["prompt_id"])
        if prompt_id not in origins:
            raise ValueError(f"missing Origin for prompt {prompt_id}")
        origin = origins[prompt_id]
        latency = float(result["timing"]["generation_seconds"])
        origin_latency = float(origin["timing"]["generation_seconds"])
        if latency <= 0 or origin_latency <= 0:
            raise ValueError("generation latency must be positive")

        timing = result["timing"]
        try:
            dit_seconds = float(timing["dit_forward_seconds"])
            dit_calls = int(timing["dit_forward_calls"])
            cache_seconds = float(timing["cache_forward_seconds"])
            cache_calls = int(timing["cache_forward_calls"])
        except KeyError as error:
            raise ValueError(
                "result is missing forward timing; rerun video generation"
            ) from error
        if dit_seconds < 0 or cache_seconds < 0:
            raise ValueError("forward timing must be non-negative")
        if dit_calls <= 0 or cache_calls < 0:
            raise ValueError("invalid forward timing call counts")
        dit_mean_ms = 1000.0 * dit_seconds / dit_calls
        cache_mean_ms = (
            1000.0 * cache_seconds / cache_calls if cache_calls else None
        )
        dit_to_cache_speedup = (
            dit_mean_ms / cache_mean_ms
            if cache_mean_ms is not None and cache_mean_ms > 0
            else None
        )

        cache = result["cache"]
        rows.append(
            {
                "run_id": result["run_id"],
                "prompt_id": prompt_id,
                "method": result["method"],
                "cache_threshold": result.get("cache_threshold"),
                "generation_seconds": latency,
                "origin_generation_seconds": origin_latency,
                "speedup": origin_latency / latency,
                "dit_forward_seconds": dit_seconds,
                "dit_forward_calls": dit_calls,
                "dit_forward_mean_milliseconds": dit_mean_ms,
                "cache_forward_seconds": cache_seconds,
                "cache_forward_calls": cache_calls,
                "cache_forward_mean_milliseconds": cache_mean_ms,
                "dit_to_cache_speedup": dit_to_cache_speedup,
                "skipped_pairs": int(cache["skipped_pairs"]),
                "skip_ratio": float(cache["skip_ratio_all_pairs"]),
                "skipped_pair_indices": " ".join(
                    str(index) for index in cache["skipped_pair_indices"]
                ),
            }
        )
    return sorted(rows, key=lambda row: (int(row["prompt_id"]), int(row["run_id"])))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["cache_threshold"])].append(row)

    summaries = []
    for (method, threshold), group in groups.items():
        dit_seconds = sum(row["dit_forward_seconds"] for row in group)
        dit_calls = sum(row["dit_forward_calls"] for row in group)
        cache_seconds = sum(row["cache_forward_seconds"] for row in group)
        cache_calls = sum(row["cache_forward_calls"] for row in group)
        dit_mean_ms = 1000.0 * dit_seconds / dit_calls
        cache_mean_ms = (
            1000.0 * cache_seconds / cache_calls if cache_calls else None
        )
        summaries.append(
            {
                "method": method,
                "cache_threshold": threshold,
                "videos": len(group),
                "latency_mean_seconds": statistics.fmean(
                    row["generation_seconds"] for row in group
                ),
                "dit_forward_seconds_total": dit_seconds,
                "dit_forward_seconds_mean": statistics.fmean(
                    row["dit_forward_seconds"] for row in group
                ),
                "dit_forward_calls_mean": statistics.fmean(
                    row["dit_forward_calls"] for row in group
                ),
                "cache_forward_seconds_mean": statistics.fmean(
                    row["cache_forward_seconds"] for row in group
                ),
                "cache_forward_calls_mean": statistics.fmean(
                    row["cache_forward_calls"] for row in group
                ),
                "dit_forward_mean_milliseconds": dit_mean_ms,
                "cache_forward_mean_milliseconds": cache_mean_ms,
                "dit_to_cache_speedup": (
                    dit_mean_ms / cache_mean_ms
                    if cache_mean_ms is not None and cache_mean_ms > 0
                    else None
                ),
                "skipped_pairs_mean": statistics.fmean(
                    row["skipped_pairs"] for row in group
                ),
                "skip_ratio_mean": statistics.fmean(
                    row["skip_ratio"] for row in group
                ),
                "speedup_mean": statistics.fmean(row["speedup"] for row in group),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = output_root.expanduser().resolve()
    rows = build_rows(load_results(output_root))
    summaries = summarize(rows)

    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_dir / "performance_per_run.csv", rows, PER_RUN_FIELDS)
    _write_csv(metrics_dir / "performance_summary.csv", summaries, SUMMARY_FIELDS)

    for row in summaries:
        print(
            f"{row['method']} threshold={row['cache_threshold']} "
            f"latency={row['latency_mean_seconds']:.4f}s "
            f"DiT_total_per_video={row['dit_forward_seconds_mean']:.2f}s "
            f"DiT_calls={row['dit_forward_calls_mean']:.2f} "
            f"cache_total={row['cache_forward_seconds_mean']:.4f}s "
            f"cache_calls={row['cache_forward_calls_mean']:.2f} "
            f"skip={row['skip_ratio_mean']:.4f} "
            f"speedup={row['speedup_mean']:.4f}x"
        )
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root)


if __name__ == "__main__":
    main()
