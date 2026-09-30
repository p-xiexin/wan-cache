"""Test causal mirror correction on scalar DiT output-change trajectories.

The 350/75/75 split is reused from the earlier exploratory experiment. No test
labels select the mirror axis, polynomial, blend, or history length. This is a
scalar replay with fully observed history, not a cached generation benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .analyze_polynomial_prior import ROOT, SEED, HORIZONS, load_data, basis, metrics, save_csv
except ImportError:
    from analyze_polynomial_prior import ROOT, SEED, HORIZONS, load_data, basis, metrics, save_csv


def anchor_forecast(log_values, prior, anchors):
    anchors = np.asarray(anchors, dtype=int)
    targets = anchors[:, None] + HORIZONS
    return prior[targets][None] + log_values[:, anchors, None] - prior[anchors][None, :, None]


def interpolate_observed(values, steps, query, anchor):
    """Return one interpolated observation per sample, or None if unavailable.

    Both interpolation endpoints must already have been observed. In particular,
    a fractional mirror position just beyond anchor must not read anchor+1.
    """
    if query < steps[0] or query > steps[anchor]:
        return None
    hi = int(np.searchsorted(steps[:anchor+1], query, side="left"))
    if hi == 0 or steps[hi] == query:
        return values[:, hi]
    lo = hi-1
    w = (query-steps[lo])/(steps[hi]-steps[lo])
    return values[:, lo]*(1-w) + values[:, hi]*w


def mirror_forecasts(log_values, prior, steps, anchors, center):
    base = anchor_forecast(log_values, prior, anchors)
    result = {key: base.copy() for key in ("raw", "level", "difference")}
    masks = {key: np.zeros(base.shape[1:], dtype=bool) for key in result}
    residual = log_values - prior
    for ai, anchor in enumerate(anchors):
        source_at_anchor = 2*center-steps[anchor]
        observed_anchor_mirror = interpolate_observed(residual, steps, source_at_anchor, anchor)
        for hi, horizon in enumerate(HORIZONS):
            target = anchor+int(horizon)
            if steps[target] <= center:
                continue
            mirror = 2*center-steps[target]
            source = interpolate_observed(log_values, steps, mirror, anchor)
            source_residual = interpolate_observed(residual, steps, mirror, anchor)
            if source is None:
                continue
            result["raw"][:, ai, hi] = source
            result["level"][:, ai, hi] = prior[target] + source_residual
            masks["raw"][ai, hi] = masks["level"][ai, hi] = True
            if observed_anchor_mirror is not None:
                result["difference"][:, ai, hi] += source_residual-observed_anchor_mirror
                masks["difference"][ai, hi] = True
    return base, result, masks


def history_forecast(log_values, prior, anchors, window):
    out = anchor_forecast(log_values, prior, anchors)
    residual = log_values-prior
    for ai, anchor in enumerate(anchors):
        offset = residual[:, max(0, anchor-window+1):anchor+1].mean(axis=1)
        out[:, ai] = prior[anchor+HORIZONS] + offset[:, None]
    return out


def virtual_node_forecast(log_values, prior, steps, anchors, center,
                          degree=2, window=5, ridge=1., mirror_weight=.1,
                          virtual_target="mirror"):
    """Fit an anchored correction polynomial with optional soft virtual nodes.

    Observed residuals e_j = log(q_j)-mu_j are reflected to step 2*c-j.
    Thus the mean left/right asymmetry is removed before making virtual nodes.
    Actual and virtual groups have total weights 1 and mirror_weight. Latest
    real node remains exact because the correction polynomial has no constant.
    """
    result = anchor_forecast(log_values, prior, anchors)
    residual = log_values-prior
    for ai, anchor in enumerate(anchors):
        observed = np.arange(max(0, anchor-window+1), anchor+1)
        local_x = (steps[observed]-steps[anchor])/8
        design = np.polynomial.polynomial.polyvander(local_x, degree)[:, 1:]
        target = (residual[:, observed]-residual[:, anchor, None]).T
        design = design/np.sqrt(len(observed))
        target = target/np.sqrt(len(observed))
        # A virtual node can only originate from an observed source.
        sources = np.arange(anchor+1)
        reflected_steps = 2*center-steps[sources]
        mask = (reflected_steps > steps[anchor]) & (reflected_steps <= steps[-1])
        sources, reflected_steps = sources[mask], reflected_steps[mask]
        if mirror_weight > 0 and len(sources):
            x_virtual = (reflected_steps-steps[anchor])/8
            factor = np.sqrt(mirror_weight/len(sources))
            virtual_design = np.polynomial.polynomial.polyvander(x_virtual, degree)[:, 1:]
            if virtual_target == "mirror":
                virtual_y = (residual[:, sources]-residual[:, anchor, None]).T
            elif virtual_target == "repeat_anchor":
                # Geometry-only control: no reflected sample information.
                virtual_y = np.zeros((len(sources), len(log_values)))
            else:
                raise ValueError(virtual_target)
            design = np.vstack([design, virtual_design*factor])
            target = np.vstack([target, virtual_y*factor])
        design = np.vstack([design, np.sqrt(ridge)*np.eye(degree)])
        target = np.vstack([target, np.zeros((degree, len(log_values)))])
        coef = np.linalg.lstsq(design, target, rcond=1e-12)[0]
        # Equal step spacing is intentional: the mirror hypothesis is about
        # the plot's step coordinate, not the recorded nonuniform model time.
        future_design = np.polynomial.polynomial.polyvander(HORIZONS/8, degree)[:, 1:]
        result[:, ai] += (future_design @ coef).T
    return result


def fit_step_prior(train, val, steps, anchors, symmetric):
    """Coefficients from train; family choices from validation forecasts only."""
    truth = val[:, anchors[:, None]+HORIZONS]
    center_grid = np.arange(24.5, 29.51, .25) if symmetric else [(steps[0]+steps[-1])/2]
    degrees = (2, 4, 6, 8, 10, 12) if symmetric else (2, 3, 5, 7, 9, 12)
    best = None
    search = []
    for center in center_grid:
        u = (steps-center)/24
        for degree in degrees:
            if symmetric:
                design = np.column_stack([u**d for d in range(0, degree+1, 2)])
            else:
                design = np.polynomial.chebyshev.chebvander(u, degree)
            for first_step in (1, 4):
                mask = steps >= first_step
                coef = np.linalg.lstsq(design[mask], train.mean(axis=0)[mask], rcond=1e-12)[0]
                prior = design @ coef
                score = metrics(anchor_forecast(val, prior, anchors), truth)["male"]
                spec = {"symmetric": symmetric, "center_step": float(center), "degree": degree,
                        "fit_start_step": first_step, "coefficients": coef.tolist(),
                        "validation_male": score}
                search.append(spec)
                if best is None or score < best[0]:
                    best = score, prior, spec
    return best[1], best[2], search


def symmetry_scan(train, steps):
    # Compare identical right-arm steps 31..48 for every center, avoiding a
    # changing-overlap objective that rewards trivial near-axis comparisons.
    mu = train.mean(axis=0)
    right = steps >= 31
    rows = []
    for center in np.round(np.arange(24.5, 30.001, .1), 8):
        reflected = np.interp(2*center-steps[right], steps, mu)
        rows.append({"center_step": float(center),
                     "train_mean_curve_symmetry_male": float(np.abs(reflected-mu[right]).mean())})
    best = min(rows, key=lambda row: row["train_mean_curve_symmetry_male"])
    return best, rows


def paired_interval(candidate, reference, truth, mask=None):
    if mask is None:
        mask = np.ones(truth.shape[1:], dtype=bool)
    c = np.abs(np.expm1(candidate[:, mask]-truth[:, mask])).mean(axis=1)
    r = np.abs(np.expm1(reference[:, mask]-truth[:, mask])).mean(axis=1)
    rng = np.random.default_rng(SEED+7)
    draws = rng.integers(0, len(c), size=(3000, len(c)))
    reduction = 100*(1-c[draws].mean(axis=1)/r[draws].mean(axis=1))
    return {"mape_reduction_percent": float(100*(1-c.mean()/r.mean())),
            "paired_trajectory_bootstrap_95_interval": np.quantile(reduction, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/videos500_experiments/run_train/lazy_dataset_500")
    parser.add_argument("--previous", type=Path, default=ROOT/"outputs/polynomial_prior")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/mirror_prior")
    args = parser.parse_args()
    files, _, data = load_data(args.data)
    manifest = json.loads((args.previous/"manifest.json").read_text())
    assert [p.name for p in files] == [f["name"] for f in manifest["files"]]
    values, steps, times = data["output_change"]
    logs = np.log(values)
    splits = manifest["splits"]
    train, val, test = (logs[splits[k]] for k in ("train", "validation", "test"))
    anchors = np.arange(4, len(steps)-8)
    targets = anchors[:, None]+HORIZONS
    truth_val, truth = val[:, targets], test[:, targets]
    args.output.mkdir(parents=True, exist_ok=True)
    selected_axis, scan = symmetry_scan(train, steps)
    center = selected_axis["center_step"]
    save_csv(args.output/"symmetry_scan.csv", scan)

    prior_specs = {}
    priors = {"table_anchor": train.mean(axis=0)}
    previous = json.loads((args.previous/"output_change/summary.json").read_text())
    for old in ("offline_polynomial", "offline_spline"):
        spec = previous["selected"][old]["prior"]
        x = (times[0]-times)/(times[0]-times[-1])
        prior = basis(x, spec["kind"], spec["degree"], spec["knots"]) @ np.array(spec["coefficients"])
        priors["old_"+old+"_anchor"] = prior
    for method, symmetric in (("step_polynomial_anchor", False), ("symmetric_polynomial_anchor", True)):
        prior, spec, search = fit_step_prior(train, val, steps, anchors, symmetric)
        priors[method], prior_specs[method] = prior, spec
        (args.output/(method+"_validation.json")).write_text(json.dumps(search, indent=2)+"\n")

    validation, forecasts, choices, mask_cache = {}, {}, {}, {}
    for method, prior in priors.items():
        validation[method] = anchor_forecast(val, prior, anchors)
        forecasts[method] = anchor_forecast(test, prior, anchors)

    # Fixed-axis mirrors of three priors, so the effects of prior representation
    # and of mirrored observations can be separated.
    mirror_priors = ("table_anchor", "old_offline_spline_anchor", "symmetric_polynomial_anchor")
    weight_grid = (0., .1, .25, .5, .75, 1.)
    search_rows = []
    for parent in mirror_priors:
        prior = priors[parent]
        base_val, pred_val, masks = mirror_forecasts(val, prior, steps, anchors, center)
        base_test, pred_test, _ = mirror_forecasts(test, prior, steps, anchors, center)
        for kind in ("raw", "level", "difference"):
            method = parent+"__mirror_"+kind
            # Direct copying is an untuned diagnostic. Other mirror variants
            # blend with their exact corresponding no-mirror baseline.
            weights = (1.,) if kind == "raw" else weight_grid
            scores = []
            for weight in weights:
                pred = base_val + weight*(pred_val[kind]-base_val)
                score = metrics(pred, truth_val)["male"]
                scores.append((score, weight))
                search_rows.append({"method": method, "weight": weight, "validation_male": score})
            score, weight = min(scores)
            validation[method] = base_val+weight*(pred_val[kind]-base_val)
            forecasts[method] = base_test+weight*(pred_test[kind]-base_test)
            choices[method] = {"parent": parent, "kind": kind, "weight": weight,
                               "center_step": center, "validation_male": score,
                               "eligible_fraction": float(masks[kind].mean())}
            mask_cache[method] = masks[kind]
    # Check whether any mirror benefit is simply averaging noisy observations.
    parent = "table_anchor"
    history_candidates = [(metrics(history_forecast(val, priors[parent], anchors, w), truth_val)["male"], w)
                          for w in (2, 3, 5, 8)]
    score, window = min(history_candidates)
    method = "table_recent_history"
    validation[method] = history_forecast(val, priors[parent], anchors, window)
    forecasts[method] = history_forecast(test, priors[parent], anchors, window)
    choices[method] = {"window": window, "validation_male": score}
    save_csv(args.output/"mirror_validation.csv", search_rows)

    virtual_search = []
    virtual_best = {}
    for degree in (1, 2, 3):
        for window in (3, 5):
            for ridge in (.01, .1, 1., 10., 100.):
                for weight in (0., .01, .1, 1.):
                    cfg = {"degree": degree, "window": window, "ridge": ridge, "mirror_weight": weight}
                    pred = virtual_node_forecast(val, priors["table_anchor"], steps, anchors, center, **cfg)
                    score = metrics(pred, truth_val)["male"]
                    virtual_search.append({**cfg, "validation_male": score})
                    method = "table_virtual_mirror_nodes" if weight > 0 else "table_local_correction"
                    if method not in virtual_best or score < virtual_best[method][0]:
                        virtual_best[method] = score, cfg, pred
    save_csv(args.output/"virtual_node_validation.csv", virtual_search)
    for method, (score, cfg, pred) in virtual_best.items():
        choices[method] = {**cfg, "validation_male": score}
        validation[method] = pred
        forecasts[method] = virtual_node_forecast(test, priors["table_anchor"], steps, anchors, center, **cfg)

    # Reflection becomes directly usable on the descending-to-ascending
    # transition: only enable virtual-node fitting after the observed anchor
    # has crossed the TRAIN-selected symmetry axis. The switch is not tuned
    # against test phases or test errors.
    ready = np.broadcast_to((steps[anchors] >= center)[:, None], targets.shape)
    gated_best = None
    gated_search = []
    for row in virtual_search:
        if row["mirror_weight"] == 0:
            continue
        cfg = {k: v for k, v in row.items() if k != "validation_male"}
        pred = virtual_node_forecast(val, priors["table_anchor"], steps, anchors, center, **cfg)
        pred = np.where(ready[None], pred, validation["table_local_correction"])
        score = metrics(pred, truth_val)["male"]
        gated_search.append({**cfg, "validation_male": score})
        if gated_best is None or score < gated_best[0]:
            gated_best = score, cfg, pred
    score, cfg, pred = gated_best
    method = "table_gated_virtual_mirror"
    choices[method] = {**cfg, "validation_male": score, "enable_after_axis": center}
    validation[method] = pred
    test_pred = virtual_node_forecast(test, priors["table_anchor"], steps, anchors, center, **cfg)
    forecasts[method] = np.where(ready[None], test_pred, forecasts["table_local_correction"])
    save_csv(args.output/"gated_virtual_node_validation.csv", gated_search)

    # Matched controls: allow the same phase switch and independent parameter
    # selection without reflected observations. This prevents attributing an
    # effect of weaker late-phase regularization to mirror information.
    control_search = []
    for method, target_type, weights in (
            ("table_gated_local", "mirror", (0.,)),
            ("table_gated_geometry_control", "repeat_anchor", (.01, .1, 1.))):
        best = None
        for row in virtual_search:
            if row["mirror_weight"] not in weights:
                continue
            cfg = {k: v for k, v in row.items() if k != "validation_male"}
            cfg["virtual_target"] = target_type
            pred = virtual_node_forecast(val, priors["table_anchor"], steps, anchors, center, **cfg)
            pred = np.where(ready[None], pred, validation["table_local_correction"])
            score = metrics(pred, truth_val)["male"]
            control_search.append({"method": method, **cfg, "validation_male": score})
            if best is None or score < best[0]:
                best = score, cfg, pred
        score, cfg, pred = best
        choices[method] = {**cfg, "validation_male": score, "enable_after_axis": center}
        validation[method] = pred
        test_pred = virtual_node_forecast(test, priors["table_anchor"], steps, anchors, center, **cfg)
        forecasts[method] = np.where(ready[None], test_pred, forecasts["table_local_correction"])
    save_csv(args.output/"gated_control_validation.csv", control_search)

    validation_scores = {m: metrics(p, truth_val)["male"] for m, p in validation.items()}
    pure_baselines = list(priors)+["table_recent_history", "table_local_correction", "table_gated_local", "table_gated_geometry_control"]
    mirror_methods = [m for m in forecasts if "__mirror_" in m]
    best_baseline = min(pure_baselines, key=validation_scores.get)
    best_mirror = min(mirror_methods+["table_virtual_mirror_nodes", "table_gated_virtual_mirror"], key=validation_scores.get)
    eligible_mask = mask_cache["table_anchor__mirror_level"]
    target_steps = steps[targets]
    rows = []
    for method, pred in forecasts.items():
        for hi, horizon in enumerate(HORIZONS):
            rows.append({"method": method, "horizon": int(horizon),
                         **metrics(pred[:, :, hi], truth[:, :, hi])})
    save_csv(args.output/"metrics.csv", rows)
    phases = []
    for phase, mask in (("all", np.ones(targets.shape, dtype=bool)),
                        ("late_36_48", target_steps >= 36),
                        ("mirror_eligible", eligible_mask)):
        for method, pred in forecasts.items():
            phases.append({"phase": phase, "method": method,
                           "forecast_count": int(mask.sum()*len(test)),
                           **metrics(pred[:, mask], truth[:, mask])})
    save_csv(args.output/"phase_metrics.csv", phases)
    summary = {"split": {k: len(v) for k, v in splits.items()}, "seed": SEED,
               "mirror_axis": selected_axis, "prior_models": prior_specs, "online_choices": choices,
               "best_nonmirror_by_validation": best_baseline, "best_mirror_by_validation": best_mirror,
               "validation_male": validation_scores,
               "test_metrics": {m: metrics(p, truth) for m, p in forecasts.items()},
               "comparison_best_mirror_vs_best_nonmirror": paired_interval(forecasts[best_mirror], forecasts[best_baseline], truth),
               "comparison_virtual_vs_local_correction": paired_interval(forecasts["table_virtual_mirror_nodes"], forecasts["table_local_correction"], truth),
               "comparison_gated_mirror_vs_local_correction": paired_interval(forecasts["table_gated_virtual_mirror"], forecasts["table_local_correction"], truth),
               "comparison_gated_mirror_late_36_48": paired_interval(forecasts["table_gated_virtual_mirror"], forecasts["table_local_correction"], truth, target_steps >= 36),
               "comparison_gated_mirror_vs_matched_phase_local": paired_interval(forecasts["table_gated_virtual_mirror"], forecasts["table_gated_local"], truth),
               "comparison_gated_mirror_vs_geometry_control": paired_interval(forecasts["table_gated_virtual_mirror"], forecasts["table_gated_geometry_control"], truth),
               "comparison_same_prior": {m: paired_interval(forecasts[m], forecasts[choices[m]["parent"]], truth)
                                          for m in mirror_methods},
               "limitations": ["Same exploratory test split as the earlier analysis; no independent replication.",
                               "Scalar log-output-change only. No tensor direction or generation quality evidence.",
                               "Rolling origin: every scalar through the anchor is observed.",
                               "Mirror observations must lie inside the observed prefix, including both interpolation endpoints.",
                               "Mirror values are predictions, never added to the true-node history.",
                               "No observations arrive over the 1/2/4/8-step forecasting interval."]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    np.savez_compressed(args.output/"forecasts.npz", truth_log=truth, steps=steps,
                        anchors=anchors, horizons=HORIZONS, mirror_eligible=eligible_mask,
                        **{f"log_{m}": pred for m, pred in forecasts.items()})

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    ax = axs[0, 0]
    mu = priors["table_anchor"]
    ax.plot(steps, np.exp(mu), color="#222", lw=2, label="Training geometric mean")
    right = steps >= 31
    ax.plot(steps[right], np.exp(np.interp(2*center-steps[right], steps, mu)), "--", color="#df7d27", label="Left arm mirrored to right")
    ax.axvline(center, color="gray", ls=":", label=f"Axis = {center:.1f}")
    ax.set(yscale="log", title="Approximate step symmetry; not exact", xlabel="Transition end step", ylabel="Output-change ratio q")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    ax.plot(steps, np.exp(mu), "k--", lw=2, label="Training geometric mean")
    for key, label in (("old_offline_polynomial_anchor", "Polynomial in model time (previous)"),
                       ("step_polynomial_anchor", "Polynomial in step coordinate"),
                       ("symmetric_polynomial_anchor", "Even polynomial around fitted axis")):
        ax.plot(steps, np.exp(priors[key]), label=label)
    ax.set(yscale="log", title="Coordinate and symmetry ablation", xlabel="Transition end step")
    ax.axvspan(1, 3.5, color="#ddd", alpha=.5, label="Warmup; new fits start at step 4")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    shown = ["old_offline_spline_anchor", "table_anchor", "table_anchor__mirror_raw",
             "table_gated_local", "table_gated_virtual_mirror"]
    labels = {"old_offline_spline_anchor": "Previous spline + anchor",
              "symmetric_polynomial_anchor": "Even polynomial + anchor",
              "table_anchor": "Table + anchor", "table_anchor__mirror_raw": "Direct mirror (fallback if unavailable)",
              "table_virtual_mirror_nodes": "Soft mirror nodes + polynomial correction",
              "table_gated_virtual_mirror": "Soft mirror nodes after symmetry axis",
              "table_gated_local": "Matched phase switch, no mirror information",
              "table_anchor__mirror_level": "Table + calibrated mirror", "table_recent_history": "Table + recent-history average"}
    for method in dict.fromkeys(shown):
        ax.plot(HORIZONS, [r["mape_percent"] for r in rows if r["method"] == method], "o-", label=labels.get(method, method))
    ax.set(title="Held-out causal forecasts (75 trajectories)", xlabel="Forecast horizon (steps)", ylabel="MAPE (%)")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    # The same first held-out trajectory as before; fix anchor 36 before seeing errors.
    ai = int(np.flatnonzero(steps[anchors] == 36)[0])
    anchor = anchors[ai]
    target = anchor+8
    mirrored = 2*center-steps[target]
    ax.plot(steps, np.exp(test[0]), color="#222", lw=1.7, label="Held-out truth")
    ax.axvspan(steps[anchor], steps[-1], color="#eee", label="Future: hidden from predictor")
    for method, label in (("table_gated_local", "Matched local correction (no mirror)"), ("table_gated_virtual_mirror", "Soft mirror-node correction")):
        ax.plot(steps[targets[ai]], np.exp(forecasts[method][0, ai]), "o--", label=label)
    source = interpolate_observed(test[:1], steps, mirrored, anchor)[0]
    ax.plot(mirrored, np.exp(source), "o", color="#a83685", ms=7, label="Observed mirror source for step 44")
    ax.annotate("past source", (mirrored, np.exp(source)), xytext=(mirrored+4, np.exp(source)*1.5), arrowprops={"arrowstyle": "->"}, fontsize=8)
    ax.axvline(steps[anchor], color="gray", ls=":")
    ax.set(yscale="log", title="First held-out example; anchor at step 36", xlabel="Transition end step")
    ax.legend(fontsize=7)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Mirror-node correction | 350 train / 75 validation / 75 test | SCALAR REPLAY", fontsize=13)
    fig.savefig(args.output/"mirror_diagnostic.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"axis": selected_axis, "selected": {m: choices.get(m, prior_specs.get(m)) for m in forecasts},
                      "test_mape": {m: v["mape_percent"] for m, v in summary["test_metrics"].items()},
                      "best_baseline": best_baseline, "best_mirror": best_mirror,
                      "comparison": summary["comparison_best_mirror_vs_best_nonmirror"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
