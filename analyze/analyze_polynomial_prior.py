"""Causal scalar-trajectory experiments; NOT full DiT output forecasting.

Run from the repository root:
  .venv/bin/python analyze/analyze_polynomial_prior.py

Only past observations from a held-out trajectory enter a forecast. Offline
priors use training trajectories; hyperparameters use validation trajectories.
The archive's eight summary features cannot recover tensor directions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = np.array([1, 2, 4, 8])
SEED = 20260930


def load_data(root):
    files = sorted(Path(root).glob("*.pt"))
    if not files:
        raise ValueError(f"No trajectories under {root}")
    objects = [torch.load(p, map_location="cpu", weights_only=True) for p in files]
    ref = objects[0]
    names = ref["feature_names"]
    column = names.index("output_change_prev")
    settings = ("task", "size", "frame_num", "sampling_steps", "sample_solver",
                "sample_shift", "guide_scale", "target_definition")
    for obj in objects:
        assert obj["feature_names"] == names
        assert torch.equal(obj["step_indices"], ref["step_indices"])
        assert torch.equal(obj["timesteps"], ref["timesteps"])
        assert obj["features"].shape == (49, 8)
        assert obj["targets"].shape == (49,)
        assert obj["features"][0, column] == 0  # unavailable history, not data
        assert all(obj["metadata"][k] == ref["metadata"][k] for k in settings)
    steps = ref["step_indices"].numpy()
    times = ref["timesteps"].double().numpy()
    assert np.array_equal(steps, np.arange(1, 50))
    assert np.all(np.diff(times) < 0)
    # Row with step r stores q_(r-1), so q_1..q_48 align to times[0:48].
    q = np.stack([o["features"][1:, column].double().numpy() for o in objects])
    errors = np.stack([o["targets"].double().numpy() for o in objects])
    for values in (q, errors):
        assert np.isfinite(values).all() and (values > 0).all()
    prompts = [o["metadata"]["prompt"] for o in objects]
    if len(set(prompts)) != len(prompts):
        raise ValueError("Duplicate prompts require a grouped split")
    return files, objects, {
        "output_change": (q, steps[1:] - 1, times[:-1]),
        "one_step_cache_error": (errors, steps, times),
    }


def basis(x, kind, degree=3, knots=()):
    if kind == "polynomial":
        return np.polynomial.chebyshev.chebvander(2 * x - 1, degree)
    if kind == "spline":
        # Cubic regression spline: C2 continuous truncated-power basis.
        return np.column_stack([np.ones_like(x), x, x*x, x*x*x]
                               + [np.maximum(x-k, 0)**3 for k in knots])
    raise ValueError(kind)


def fit_prior(train_log, x, kind, degree=3, knot_indices=(), ridge=0.):
    if kind == "table":
        return train_log.mean(axis=0), {"kind": kind}
    knots = x[list(knot_indices)] if knot_indices else []
    design = basis(x, kind, degree, knots)
    # Prior is fitted to the train log-mean; never use a test trajectory's future.
    penalty = np.eye(design.shape[1]) * np.sqrt(ridge)
    penalty[0, 0] = 0
    augmented = np.vstack([design, penalty])
    target = np.concatenate([train_log.mean(axis=0), np.zeros(len(penalty))])
    coefficients = np.linalg.lstsq(augmented, target, rcond=1e-12)[0]
    return design @ coefficients, {"kind": kind, "degree": degree,
            "knots": list(map(float, knots)), "ridge": ridge,
            "coefficients": coefficients.tolist()}


def forecast(log_values, x, anchors, prior, kind, degree=1, window=5, ridge=0.):
    """Return [trajectory, anchor, horizon]; reads no values beyond each anchor."""
    result = np.empty((len(log_values), len(anchors), len(HORIZONS)))
    for ai, anchor in enumerate(anchors):
        future = anchor + HORIZONS
        if kind == "prior":
            result[:, ai] = prior[future]
            continue
        if kind == "persistence":
            result[:, ai] = log_values[:, anchor, None]
            continue
        start = max(0, anchor-window+1)
        scale = x[anchor] - x[start]
        if scale <= 0:
            raise ValueError("Need at least two historical timestamps")
        past_x = (x[start:anchor+1] - x[anchor]) / scale
        future_x = (x[future] - x[anchor]) / scale
        past = log_values[:, start:anchor+1]
        if kind == "local":
            design = np.polynomial.polynomial.polyvander(past_x, degree)
            weights = np.linalg.lstsq(design, past.T, rcond=1e-12)[0]
            result[:, ai] = (np.polynomial.polynomial.polyvander(future_x, degree) @ weights).T
        elif kind == "correction":
            residual = past - prior[start:anchor+1]
            offset = residual[:, -1]
            pred = prior[future][None, :] + offset[:, None]
            if degree:
                design = np.polynomial.polynomial.polyvander(past_x, degree)[:, 1:]
                augmented = np.vstack([design, np.sqrt(ridge) * np.eye(degree)])
                target = np.vstack([(residual-offset[:, None]).T,
                                    np.zeros((degree, len(log_values)))])
                weights = np.linalg.lstsq(augmented, target, rcond=1e-12)[0]
                pred = pred + (np.polynomial.polynomial.polyvander(future_x, degree)[:, 1:] @ weights).T
            result[:, ai] = pred
        else:
            raise ValueError(kind)
    assert np.isfinite(result).all()
    return result


def metrics(pred, truth):
    delta = pred - truth
    relative = np.abs(np.expm1(delta))
    return {"male": float(np.mean(np.abs(delta))),
            "mape_percent": float(100 * relative.mean()),
            "p95_ape_percent": float(100 * np.quantile(relative, .95)),
            "rmse": float(np.sqrt(np.mean((np.exp(pred)-np.exp(truth))**2)))}


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def experiment(name, values, steps, times, splits, output):
    log_values = np.log(values)
    train, val, test = (log_values[splits[k]] for k in ("train", "validation", "test"))
    x = (times[0]-times)/(times[0]-times[-1])
    anchors = np.arange(4, len(x)-int(HORIZONS.max()))
    truth_val = val[:, anchors[:, None] + HORIZONS]
    truth_test = test[:, anchors[:, None] + HORIZONS]
    selected, priors, tuning = {}, {}, []
    # Global polynomial and piecewise cubic are selected independently.
    configurations = {
        "offline_polynomial": [dict(kind="polynomial", degree=d) for d in (2, 3, 5, 7)],
        "offline_spline": [dict(kind="spline", knot_indices=knots, ridge=r)
                           for knots in ((7, 23, 39), (3, 7, 15, 23, 31, 39, 43))
                           for r in (0., 1e-10, 1e-8, 1e-6)],
        "offline_table": [dict(kind="table")],
    }
    for method, configs in configurations.items():
        candidates = []
        for cfg in configs:
            prior, spec = fit_prior(train, x, **cfg)
            score = metrics(forecast(val, x, anchors, prior, "prior"), truth_val)["male"]
            candidates.append((score, prior, spec))
            tuning.append({"method": method, "validation_male": score, "config": spec})
        score, prior, spec = min(candidates, key=lambda c: c[0])
        priors[method] = prior
        selected[method] = {"prior": spec, "forecast": {"kind": "prior"}, "validation_male": score}

    spline = priors["offline_spline"]
    candidates = {
        "persistence": [dict(kind="persistence")],
        "local_linear": [dict(kind="local", degree=1, window=w) for w in (3, 5)],
        "local_quadratic": [dict(kind="local", degree=2, window=w) for w in (3, 5)],
        "polynomial_anchor": [dict(kind="correction", degree=0)],
        "table_anchor": [dict(kind="correction", degree=0)],
        "spline_anchor": [dict(kind="correction", degree=0)],
        "spline_online": [dict(kind="correction", degree=d, window=w, ridge=r)
                          for d in (0, 1, 2) for w in (3, 5)
                          for r in ((0.,) if d == 0 else (.01, .1, 1., 10., 100.))],
    }
    prior_sources = {"polynomial_anchor": "offline_polynomial", "table_anchor": "offline_table"}
    for method, configs in candidates.items():
        scored = []
        prior_source = prior_sources.get(method, "offline_spline")
        for cfg in configs:
            score = metrics(forecast(val, x, anchors, priors[prior_source], **cfg), truth_val)["male"]
            tuning.append({"method": method, "validation_male": score, "config": cfg})
            scored.append((score, cfg))
        score, cfg = min(scored, key=lambda c: c[0])
        selected[method] = {"forecast": cfg, "prior_source": prior_source, "validation_male": score}

    predictions, rows = {}, []
    for method, spec in selected.items():
        prior = priors.get(spec.get("prior_source", method), spline)
        predictions[method] = forecast(test, x, anchors, prior, **spec["forecast"])
        for hi, horizon in enumerate(HORIZONS):
            rows.append({"series": name, "method": method, "horizon": int(horizon),
                         "forecast_count": int(len(test)*len(anchors)),
                         **metrics(predictions[method][:, :, hi], truth_test[:, :, hi])})
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "metrics.csv", rows)
    phases = []
    target_steps = steps[anchors[:, None] + HORIZONS]
    for phase, low, high in (("early", 1, 15), ("middle", 16, 35), ("late", 36, 49)):
        mask = (target_steps >= low) & (target_steps <= high)
        for method, pred in predictions.items():
            phases.append({"phase": phase, "method": method,
                           "forecast_count": int(mask.sum()*len(test)),
                           **metrics(pred[:, mask], truth_test[:, mask])})
    save_csv(output / "phase_metrics.csv", phases)
    aggregates = {m: metrics(p, truth_test) for m, p in predictions.items()}
    rng = np.random.default_rng(SEED+1)
    draws = rng.integers(0, len(test), size=(2000, len(test)))
    ref_errors = np.abs(np.expm1(predictions["offline_spline"]-truth_test)).mean(axis=(1, 2))
    online_errors = np.abs(np.expm1(predictions["spline_online"]-truth_test)).mean(axis=(1, 2))
    improvement = 100 * (1-online_errors[draws].mean(axis=1)/ref_errors[draws].mean(axis=1))
    summary = {"series": name, "hyperparameter_selection": "minimum validation MALE, never test",
               "selected": selected, "test_metrics": aggregates,
               "online_mape_reduction_vs_offline_spline_percent": 100*(1-online_errors.mean()/ref_errors.mean()),
               "paired_trajectory_bootstrap_95_percent_interval": np.quantile(improvement, [.025, .975]).tolist(),
               "anchor_steps": steps[anchors].tolist(), "horizons": HORIZONS.tolist(),
               "forecast_space": "log of a positive scalar; local baselines are not EasyCache/D2Cache tensor implementations",
               "limitations": ["Scalar prediction only, not tensor prediction or cached inference.",
                               "Each rolling-origin experiment observes true scalar history through the anchor.",
                               "No new truth is observed during the entire forecast horizon.",
                               "q at a node requires two adjacent actual outputs; isolated refreshes do not expose it.",
                               "One-step cache error labels do not measure long skip errors or closed-loop drift."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output / "validation_search.json").write_text(json.dumps(tuning, indent=2)+"\n")
    np.savez_compressed(output / "forecasts.npz", truth_log=truth_test, anchors=anchors,
                        steps=steps, timesteps=times, horizons=HORIZONS,
                        **{f"log_{k}": v for k, v in predictions.items()})

    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(steps, values.T, color="#287bb5", alpha=.035, lw=.6)
    ax.plot(steps, np.median(values, axis=0), color="#144e7a", lw=2, label="Median (500 trajectories)")
    ax.set(yscale="log", title="Observed scalar trajectories", xlabel="Transition end step", ylabel=name)
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.plot(steps, np.exp(train.mean(axis=0)), "k--", label="Training log-mean")
    for m in ("offline_polynomial", "offline_spline"):
        ax.plot(steps, np.exp(priors[m]), label=m)
    ax.set(yscale="log", title="Offline priors (350 training trajectories)", xlabel="Transition end step")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    shown = ("persistence", "local_linear", "local_quadratic", "offline_spline", "spline_anchor", "table_anchor")
    for m in shown:
        ax.plot(HORIZONS, [r["male"] for r in rows if r["method"] == m], "o-", label=m)
    ax.set(yscale="log", title="Held-out forecast errors (75 trajectories)", xlabel="Forecast horizon (steps)", ylabel="Mean absolute log error")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=7, ncols=2)
    ax = axs[1, 1]
    # First test example and a predetermined anchor; no cherry-picking by error.
    ai = min(19, len(anchors)-1)
    anchor = anchors[ai]
    segment = slice(max(0, anchor-6), min(len(steps), anchor+11))
    ax.plot(steps[segment], np.exp(test[0, segment]), color="#222", label="Held-out truth")
    for m in ("offline_spline", "spline_anchor", "spline_online"):
        ax.plot(steps[anchor+HORIZONS], np.exp(predictions[m][0, ai]), "o--", label=m)
    ax.axvline(steps[anchor], color="gray", linestyle=":", label="Last observed node")
    ax.set(yscale="log", xlim=(steps[max(0, anchor-6)], steps[min(len(steps)-1, anchor+10)]),
           title="First held-out example; no future observations", xlabel="Transition end step")
    ax.legend(fontsize=7)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Offline polynomial prior + causal node correction | SCALAR DIAGNOSTIC", fontsize=13)
    fig.savefig(output / "diagnostic.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/videos500_experiments/run_train/lazy_dataset_500")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/polynomial_prior")
    args = parser.parse_args()
    files, objects, series = load_data(args.data)
    if len(files) != 500:
        raise ValueError("This fixed 350/75/75 experiment expects 500 trajectories")
    order = np.random.default_rng(SEED).permutation(len(files))
    splits = {"train": order[:350], "validation": order[350:425], "test": order[425:]}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "data_root": str(args.data.resolve()),
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                           "prompt_index": o["metadata"]["prompt_index"], "generation_seed": o["metadata"]["seed"]}
                          for p, o in zip(files, objects)],
                "splits": {k: v.tolist() for k, v in splits.items()},
                "output_change_alignment": "features[1:, output_change_prev] -> transition end step step_indices[1:]-1; timestep timesteps[:-1]",
                "target_definition": objects[0]["metadata"]["target_definition"],
                "coordinate": "normalized recorded model timestep, not uniform step index",
                "tensor_outputs_present": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for name, (values, steps, times) in series.items():
        result = experiment(name, values, steps, times, splits, args.output / name)
        print(name, json.dumps(result["test_metrics"], indent=2), flush=True)
        print("online improvement", result["online_mape_reduction_vs_offline_spline_percent"],
              result["paired_trajectory_bootstrap_95_percent_interval"], flush=True)


if __name__ == "__main__":
    main()
