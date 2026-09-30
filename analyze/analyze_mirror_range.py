"""Expanded mirror stress test: long horizons, sparse observations and OOF splits.

No new trajectories are generated. Conditions are scalar replay experiments,
not independent samples or measured cache speedups. Each of 500 trajectories
is tested out of fold in two random partitions and one prompt-order partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analyze_polynomial_prior import ROOT, SEED, load_data, save_csv
    from .analyze_mirror_prior import symmetry_scan
except ImportError:
    from analyze_polynomial_prior import ROOT, SEED, load_data, save_csv
    from analyze_mirror_prior import symmetry_scan


HORIZONS = (1, 2, 4, 8, 12, 16, 20, 24, 32)
STRIDES = (1, 2, 4, 8)
METHODS = ("anchor", "local", "geometry", "mirror", "direct_mirror")


def observed_nodes(anchor, stride):
    """First three observations plus a stride-spaced history ending at anchor."""
    return np.unique(np.r_[np.arange(min(3, anchor+1)), np.arange(anchor, 2, -stride)]).astype(int)


def conditions(length=48, horizons=HORIZONS, strides=STRIDES):
    rows = [(stride, anchor, h) for stride in strides for anchor in range(2, length-1)
            for h in horizons if anchor+h < length]
    return np.asarray(rows, dtype=int)


def condition_groups(query, steps, center):
    stride, anchor, horizon = query.T
    mirror_source = 2*center-steps[anchor+horizon]
    available = ((steps[anchor+horizon] > center) & (mirror_source >= steps[0])
                 & (mirror_source <= steps[anchor]))
    # Availability includes interpolation support only; all actual virtual
    # node values below are taken from observed nodes, never dense history.
    band = np.where(horizon <= 4, 0, np.where(horizon <= 12, 1, 2))
    keys = [(int(s), int(b), bool(a)) for s, b, a in zip(stride, band, available)]
    groups = {key: np.array([k == key for k in keys]) for key in sorted(set(keys))}
    return groups, available


def correction_operator(query, steps, center, degree=1, window=5,
                        ridge=1., weight=0., mode="local"):
    """Linear map from sample residual history to every predicted residual.

    Virtual observations have total weight `weight`; local observations have
    total weight 1. Subtracting the anchor residual makes the latest real node
    exact. Mirror sources are restricted to the sparse observation set.
    """
    result = np.zeros((len(query), len(steps)), dtype=float)
    for stride, anchor in sorted(set(map(tuple, query[:, :2]))):
        rows = np.flatnonzero((query[:, 0] == stride) & (query[:, 1] == anchor))
        observed = observed_nodes(anchor, stride)
        local = observed[-window:]
        x = (steps[local]-steps[anchor])/8
        design = np.polynomial.polynomial.polyvander(x, degree)[:, 1:]/np.sqrt(len(local))
        target_map = np.zeros((len(local), len(steps)))
        target_map[np.arange(len(local)), local] += 1
        target_map[:, anchor] -= 1
        target_map /= np.sqrt(len(local))
        mirror_steps = 2*center-steps[observed]
        mask = (mirror_steps > steps[anchor]) & (mirror_steps <= steps[-1])
        sources, mirror_steps = observed[mask], mirror_steps[mask]
        if weight > 0 and len(sources):
            factor = np.sqrt(weight/len(sources))
            virtual_x = (mirror_steps-steps[anchor])/8
            virtual_design = np.polynomial.polynomial.polyvander(virtual_x, degree)[:, 1:]
            virtual_map = np.zeros((len(sources), len(steps)))
            if mode == "mirror":
                virtual_map[np.arange(len(sources)), sources] += 1
                virtual_map[:, anchor] -= 1
            elif mode != "geometry":
                raise ValueError(mode)
            design = np.vstack([design, virtual_design*factor])
            target_map = np.vstack([target_map, virtual_map*factor])
        design = np.vstack([design, np.sqrt(ridge)*np.eye(degree)])
        target_map = np.vstack([target_map, np.zeros((degree, len(steps)))])
        mapping = np.linalg.lstsq(design, target_map, rcond=1e-12)[0]
        future = np.polynomial.polynomial.polyvander(query[rows, 2]/8, degree)[:, 1:]
        result[rows] = future @ mapping
        result[rows, anchor] += 1
        forbidden = np.setdiff1d(np.arange(len(steps)), observed)
        assert np.max(np.abs(result[np.ix_(rows, forbidden)]), initial=0.) < 1e-12
    return result


def raw_mirror(log_values, fallback, query, steps, center, available):
    result = fallback.copy()
    for qi in np.flatnonzero(available):
        stride, anchor, h = query[qi]
        observed = observed_nodes(anchor, stride)
        source = 2*center-steps[anchor+h]
        hi = int(np.searchsorted(steps[observed], source))
        if hi == 0 or steps[observed[hi]] == source:
            result[:, qi] = log_values[:, observed[hi]]
        else:
            left, right = observed[hi-1], observed[hi]
            w = (source-steps[left])/(steps[right]-steps[left])
            result[:, qi] = (1-w)*log_values[:, left] + w*log_values[:, right]
    return result


def make_splits(count=500):
    protocols = {"random_1": np.random.default_rng(SEED+101).permutation(count),
                 "random_2": np.random.default_rng(SEED+202).permutation(count),
                 "prompt_order": np.arange(count)}
    result = []
    for protocol, order in protocols.items():
        folds = np.array_split(order, 5)
        for fold in range(5):
            test, val = folds[fold], folds[(fold+1) % 5]
            train = np.concatenate([folds[j] for j in range(5) if j not in (fold, (fold+1) % 5)])
            assert not (set(train) & set(test) or set(val) & set(test) or set(train) & set(val))
            result.append({"protocol": protocol, "fold": fold,
                           "train": train, "validation": val, "test": test})
    return result


def choose_models(train, val, query, steps, center):
    prior = train.mean(axis=0)
    targets = query[:, 1]+query[:, 2]
    residual = val-prior
    truth = val[:, targets]
    groups, available = condition_groups(query, steps, center)
    operators, scores, choices = {}, {}, {}
    for mode in ("local", "geometry", "mirror"):
        if mode == "local":
            operators[mode] = np.zeros((len(query), len(steps)))
            scores[mode] = {key: float("inf") for key in groups}
            choices[mode] = {}
        else:
            # Zero mirror weight remains a valid candidate; validation can
            # reject mirror information separately for each condition group.
            operators[mode] = operators["local"].copy()
            scores[mode] = scores["local"].copy()
            choices[mode] = {k: dict(v) for k, v in choices["local"].items()}
        for degree in (1, 2):
            for window in (3, 5):
                for ridge in (.01, .1, 1., 10., 100.):
                    for weight in ((0.,) if mode == "local" else (.01, .1, 1.)):
                        cfg = dict(degree=degree, window=window, ridge=ridge, weight=weight, mode=mode)
                        operator = correction_operator(query, steps, center, **cfg)
                        pred = prior[targets] + residual @ operator.T
                        point_loss = np.abs(pred-truth).mean(axis=0)
                        for key, mask in groups.items():
                            if mode != "local" and not key[2]:
                                continue
                            score = float(point_loss[mask].mean())
                            if score < scores[mode][key]:
                                operators[mode][mask] = operator[mask]
                                scores[mode][key] = score
                                choices[mode][key] = {**cfg, "validation_male": score}
    serializable = {mode: [{"stride": key[0], "horizon_band": key[1], "mirror_available": key[2], **cfg}
                          for key, cfg in selection.items()] for mode, selection in choices.items()}
    return prior, operators, serializable, available


def scalar_metrics(pred, truth):
    delta = pred-truth
    ape = np.abs(np.expm1(delta))
    assert np.isfinite(ape).all(), "Nonfinite relative errors; inspect extrapolation instability"
    return {"male": float(np.abs(delta).mean()), "mape_percent": float(100*ape.mean()),
            "median_ape_percent": float(100*np.median(ape)),
            "p95_ape_percent": float(100*np.quantile(ape, .95))}


def bootstrap_reduction(mirror_errors, baseline_errors):
    # Inputs already average conditions and repeated partitions within each
    # trajectory. Resample 500 trajectory IDs, not millions of replay windows.
    draws = np.random.default_rng(SEED+909).integers(0, len(mirror_errors), (2000, len(mirror_errors)))
    improvement = 100*(1-mirror_errors[draws].mean(axis=1)/baseline_errors[draws].mean(axis=1))
    return {"relative_mape_reduction_percent": float(100*(1-mirror_errors.mean()/baseline_errors.mean())),
            "trajectory_bootstrap_95_interval": np.quantile(improvement, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/videos500_experiments/run_train/lazy_dataset_500")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/mirror_range")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files, objects, data = load_data(args.data)
    values, steps, times = data["output_change"]
    if len(files) != 500:
        raise ValueError("Expected 500 trajectories")
    logs = np.log(values)
    query = conditions(len(steps))
    targets = query[:, 1]+query[:, 2]
    truth = logs[:, targets]
    protocols = ("random_1", "random_2", "prompt_order")
    oof = {p: {m: np.full(truth.shape, np.nan) for m in METHODS} for p in protocols}
    eligible = {p: np.zeros(truth.shape, dtype=bool) for p in protocols}
    selections, fold_rows, split_manifest = [], [], []
    for split in make_splits(len(files)):
        protocol, fold = split["protocol"], split["fold"]
        train, val, test = (logs[split[k]] for k in ("train", "validation", "test"))
        axis, _ = symmetry_scan(train, steps)
        center = axis["center_step"]
        prior, operators, selected, available = choose_models(train, val, query, steps, center)
        pred = {"anchor": prior[targets]+(test-prior)[:, query[:, 1]]}
        for mode, operator in operators.items():
            pred[mode] = prior[targets] + (test-prior) @ operator.T
        pred["direct_mirror"] = raw_mirror(test, pred["local"], query, steps, center, available)
        test_truth = test[:, targets]
        for method in METHODS:
            oof[protocol][method][split["test"]] = pred[method]
            fold_rows.append({"protocol": protocol, "fold": fold, "method": method,
                              **scalar_metrics(pred[method], test_truth)})
        eligible[protocol][split["test"]] = available
        selections.append({"protocol": protocol, "fold": fold, "center": center, "models": selected})
        split_manifest.append({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in split.items()})
        print(f"{protocol} fold {fold+1}/5: local={fold_rows[-4]['mape_percent']:.3f}% "
              f"mirror={fold_rows[-2]['mape_percent']:.3f}% axis={center:.1f}", flush=True)

    for methods in oof.values():
        assert all(np.isfinite(v).all() for v in methods.values())
    condition_rows = [{"stride": int(s), "anchor_step": int(steps[a]), "target_step": int(steps[a+h]),
                       "horizon": int(h), "observed_nodes": len(observed_nodes(a, s))} for s, a, h in query]
    save_csv(args.output/"conditions.csv", condition_rows)
    save_csv(args.output/"fold_metrics.csv", fold_rows)
    (args.output/"selected_models.json").write_text(json.dumps(selections, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps({
        "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
        "splits": split_manifest, "horizons": HORIZONS, "strides": STRIDES,
        "conditions_per_trajectory": len(query), "distinct_trajectories": 500,
        "generation_settings": {k: objects[0]["metadata"][k] for k in
             ("task", "size", "frame_num", "sampling_steps", "sample_solver", "sample_shift", "guide_scale")},
        "model_selection": "Validation MALE within stride, horizon band and mirror availability; same groups for controls",
        "observation_rule": "First three scalar observations plus stride-spaced past ending at anchor; no intermediate observations during forecast",
    }, indent=2)+"\n")

    masks = {"all": np.ones(len(query), dtype=bool)}
    masks.update({f"horizon_{h}": query[:, 2] == h for h in HORIZONS})
    masks.update({f"stride_{s}": query[:, 0] == s for s in STRIDES})
    masks.update({f"stride_{s}_horizon_{h}": (query[:, 0] == s) & (query[:, 2] == h) for s in STRIDES for h in HORIZONS})
    masks.update({"target_early_1_15": steps[targets] <= 15,
                  "target_middle_16_35": (steps[targets] >= 16) & (steps[targets] <= 35),
                  "target_late_36_48": steps[targets] >= 36})
    rows, protocol_metrics = [], {}
    error_arrays = {m: [] for m in METHODS}
    for protocol, methods in oof.items():
        protocol_metrics[protocol] = {}
        for method, pred in methods.items():
            error_arrays[method].append(np.abs(np.expm1(pred-truth)))
            protocol_metrics[protocol][method] = scalar_metrics(pred, truth)
            for group, mask in masks.items():
                rows.append({"protocol": protocol, "group": group, "method": method,
                             "forecast_count": int(mask.sum()*500), **scalar_metrics(pred[:, mask], truth[:, mask])})
            mask = eligible[protocol]
            rows.append({"protocol": protocol, "group": "mirror_available", "method": method,
                         "forecast_count": int(mask.sum()), **scalar_metrics(pred[mask], truth[mask])})
        np.savez_compressed(args.output/(protocol+"_forecasts.npz"), truth_log=truth,
                            query=query, mirror_available=eligible[protocol],
                            **{f"log_{m}": pred for m, pred in methods.items()})
    save_csv(args.output/"group_metrics.csv", rows)
    # Average repeated evaluations for each trajectory before any uncertainty
    # estimate. This does not turn 500 trajectories into 1,500 independent ones.
    pooled_errors = {m: np.mean(v, axis=0) for m, v in error_arrays.items()}
    comparisons = []
    for group, mask in masks.items():
        for baseline in ("anchor", "local", "geometry"):
            comparison = bootstrap_reduction(pooled_errors["mirror"][:, mask].mean(axis=1),
                                             pooled_errors[baseline][:, mask].mean(axis=1))
            comparisons.append({"group": group, "baseline": baseline, **comparison})
    (args.output/"paired_comparisons.json").write_text(json.dumps(comparisons, indent=2)+"\n")

    # Descriptive strata only. Full trajectories define strata for auditing;
    # these labels never select a model or enter a prediction.
    amplitude = logs[:, 5:].mean(axis=1)
    right = steps >= 31
    reflected = np.stack([np.interp(52.8-steps[right], steps, row) for row in logs])
    asymmetry = np.abs(reflected-logs[:, right]).mean(axis=1)
    strata_rows, strata_labels = [], {}
    for name, score in (("amplitude", amplitude), ("asymmetry", asymmetry)):
        labels = np.empty(500, dtype=int)
        for quartile, ids in enumerate(np.array_split(np.argsort(score, kind="stable"), 4)):
            labels[ids] = quartile
            for method in METHODS:
                strata_rows.append({"stratum": name, "quartile": quartile+1, "trajectories": len(ids),
                                    "method": method, "mape_percent": float(100*pooled_errors[method][ids].mean())})
        strata_labels[name] = labels
    save_csv(args.output/"distribution_strata.csv", strata_rows)
    np.savez_compressed(args.output/"trajectory_strata.npz", amplitude=amplitude, asymmetry=asymmetry, **{f"quartile_{k}": v for k, v in strata_labels.items()})
    summary = {"distinct_trajectories": 500, "protocols": len(protocols), "folds": 15,
               "train_validation_test_per_fold": [300, 100, 100],
               "conditions_per_trajectory": len(query), "forecast_evaluations_per_method": 500*len(query)*len(protocols),
               "pooled_mape_percent": {m: float(100*v.mean()) for m, v in pooled_errors.items()},
               "protocol_metrics": protocol_metrics,
               "overall_comparisons": [r for r in comparisons if r["group"] == "all"],
               "limitations": ["Same 500 underlying trajectories; expanded replay conditions, not new generation data.",
                               "All trajectories share model, solver, step count, resolution, CFG and shift.",
                               "Only scalar output-change values exist, not full output tensors or closed-loop runs.",
                               "Each q observation itself requires adjacent real model outputs; observation stride is not a measured cache speedup.",
                               "Previous exploratory work already inspected these trajectories; OOF results are not an untouched external test set.",
                               "Confidence intervals cluster by trajectory; they do not retrain models under bootstrap or correct subgroup multiplicity.",
                               "Amplitude and asymmetry quartiles are post-hoc diagnostic strata, never model inputs."]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.7), constrained_layout=True)
    ax = axs[0, 0]
    labels = {"anchor": "Prior + latest node", "local": "Local polynomial (matched tuning)",
              "geometry": "Virtual geometry, no mirror values", "mirror": "Soft mirror nodes"}
    for m in labels:
        ax.plot(HORIZONS, [100*pooled_errors[m][:, query[:, 2] == h].mean() for h in HORIZONS], "o-", label=labels[m])
    ax.set(title="Longer forecast horizons", xlabel="Future steps without new observations", ylabel="MAPE (%)")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    gains = np.array([[100*(1-pooled_errors["mirror"][:, masks[f"stride_{s}_horizon_{h}"]].mean()/pooled_errors["local"][:, masks[f"stride_{s}_horizon_{h}"]].mean())
                       for h in HORIZONS] for s in STRIDES])
    limit = max(2., float(np.abs(gains).max()))
    im = ax.imshow(gains, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(HORIZONS)), HORIZONS)
    ax.set_yticks(range(len(STRIDES)), STRIDES)
    ax.set(title="Mirror vs matched local: relative error reduction (%)", xlabel="Forecast horizon", ylabel="Historical observation stride")
    for y in range(len(STRIDES)):
        for x in range(len(HORIZONS)):
            ax.text(x, y, f"{gains[y,x]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(gains[y,x]) > limit*.6 else "black")
    fig.colorbar(im, ax=ax, shrink=.8, label="Positive = mirror improves")
    ax = axs[1, 0]
    individual = 100*(1-pooled_errors["mirror"].mean(axis=1)/pooled_errors["local"].mean(axis=1))
    ax.hist(individual, bins=30, color="#387fac", alpha=.85)
    ax.axvline(0, color="#333", ls="--")
    ax.set(title="Distribution across 500 distinct trajectories", xlabel="Relative MAPE reduction from mirror (%)", ylabel="Trajectory count")
    ax = axs[1, 1]
    for offset, m in ((-.18, "local"), (.18, "mirror")):
        vals = [r["mape_percent"] for r in strata_rows if r["stratum"] == "asymmetry" and r["method"] == m]
        ax.bar(np.arange(4)+offset, vals, width=.36, label=labels[m])
    ax.set_xticks(range(4), ["Most symmetric", "Q2", "Q3", "Least symmetric"])
    ax.set(title="Asymmetry strata (125 trajectories each)", ylabel="MAPE (%)")
    ax.legend(fontsize=8)
    for ax in (axs[0, 0], axs[1, 0], axs[1, 1]):
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Expanded scalar mirror test | 15 OOF folds | 1-32 step horizons | 1-8 step observation stride", fontsize=12)
    fig.savefig(args.output/"expanded_mirror_diagnostic.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"summary": summary["pooled_mape_percent"], "comparisons": summary["overall_comparisons"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
