"""Evaluate mirror correction on ALL 500 held-out trajectories, at fixed horizons.

Preserves the previous mirror experiment's 36 anchors, horizons, dense observed
history, hyperparameter grids and phase gate. Each partition gives every sample
one out-of-fold prediction. Repeated partitions are sensitivity checks, not new
samples. Full-curve descriptors are diagnostic labels only, never model inputs.
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
    from .analyze_polynomial_prior import ROOT, SEED, HORIZONS, load_data, save_csv
    from .analyze_mirror_prior import symmetry_scan
    from .analyze_mirror_range import correction_operator, make_splits, raw_mirror, scalar_metrics
except ImportError:
    from analyze_polynomial_prior import ROOT, SEED, HORIZONS, load_data, save_csv
    from analyze_mirror_prior import symmetry_scan
    from analyze_mirror_range import correction_operator, make_splits, raw_mirror, scalar_metrics


METHODS = ("anchor", "local_global", "local", "geometry", "mirror", "direct_mirror")


def original_conditions():
    return np.array([(1, a, int(h)) for a in range(4, 40) for h in HORIZONS])


def select_models(train, validation, query, steps, center):
    """Same candidate families and selection loss as analyze_mirror_prior.py."""
    prior = train.mean(axis=0)
    targets = query[:, 1]+query[:, 2]
    residual = validation-prior
    truth = validation[:, targets]
    ready = steps[query[:, 1]] >= center
    best = {}
    for mode in ("local", "geometry", "mirror"):
        for degree in (1, 2, 3):
            for window in (3, 5):
                for ridge in (.01, .1, 1., 10., 100.):
                    for weight in ((0.,) if mode == "local" else (.01, .1, 1.)):
                        cfg = dict(degree=degree, window=window, ridge=ridge, weight=weight, mode=mode)
                        op = correction_operator(query, steps, center, **cfg)
                        loss = np.abs(prior[targets]+residual @ op.T-truth).mean(axis=0)
                        candidates = {mode: float(loss[ready].mean())}
                        if mode == "local":
                            candidates["local_global"] = float(loss.mean())
                        for name, score in candidates.items():
                            if name not in best or score < best[name][0]:
                                best[name] = score, cfg, op
    operators = {name: np.where(ready[:, None], result[2], best["local_global"][2])
                 for name, result in best.items()}
    operators["local_global"] = best["local_global"][2]
    selected = {name: {**result[1], "validation_male_on_selection_region": result[0],
                      "selection_region": "all" if name == "local_global" else "anchor_after_axis"}
                for name, result in best.items()}
    return prior, operators, selected


def paired_summary(candidate, baseline):
    """Cluster bootstrap of paired trajectory-level MAPE, in percentage units.

    Conditional on fitted OOF models; this does not re-fit CV within bootstrap.
    """
    delta = baseline-candidate
    draws = np.random.default_rng(SEED+818).integers(0, len(delta), (5000, len(delta)))
    rel = 100*(1-candidate[draws].mean(axis=1)/baseline[draws].mean(axis=1))
    gap = delta[draws].mean(axis=1)
    return {"n_trajectories": len(delta), "baseline_mape_percent": float(baseline.mean()),
            "mirror_mape_percent": float(candidate.mean()),
            "absolute_improvement_percentage_points": float(delta.mean()),
            "absolute_improvement_95_interval": np.quantile(gap, [.025, .975]).tolist(),
            "relative_mape_reduction_percent": float(100*(1-candidate.mean()/baseline.mean())),
            "relative_reduction_95_interval": np.quantile(rel, [.025, .975]).tolist(),
            "improved_samples": int((delta > 1e-10).sum()),
            "worsened_samples": int((delta < -1e-10).sum()),
            "tied_samples": int((np.abs(delta) <= 1e-10).sum()),
            "per_sample_improvement_quantiles_pp": dict(zip(
                ("min", "p05", "p25", "p50", "p75", "p95", "max"),
                map(float, np.quantile(delta, [0, .05, .25, .5, .75, .95, 1]))))}


def descriptors(logs, priors, centers, steps):
    right = steps >= 31
    raw, residual, best_axis, best_loss = [], [], [], []
    for y, prior, center in zip(logs, priors, centers):
        e = y-prior
        raw.append(np.abs(np.interp(2*center-steps[right], steps, y)-y[right]).mean())
        residual.append(np.abs(np.interp(2*center-steps[right], steps, e)-e[right]).mean())
        # Full-trajectory axis is a retrospective descriptive statistic ONLY.
        grid = np.round(np.arange(24.5, 30.001, .1), 8)
        losses = [np.abs(np.interp(2*c-steps[right], steps, y)-y[right]).mean() for c in grid]
        idx = int(np.argmin(losses))
        best_axis.append(grid[idx])
        best_loss.append(losses[idx])
    return {"log_amplitude": logs[:, steps >= 5].mean(axis=1),
            "raw_asymmetry": np.array(raw), "residual_asymmetry": np.array(residual),
            "training_axis": np.asarray(centers),
            "descriptive_best_axis": np.array(best_axis),
            "descriptive_best_symmetry_loss": np.array(best_loss),
            "minimum_step": steps[logs.argmin(axis=1)]}


def quartiles(x):
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=int)
    for q, indices in enumerate(np.array_split(order, 4)):
        result[indices] = q
    return result


def make_figures(output, values, steps, desc, errors, comparisons, strata, files):
    logs = np.log(values)
    diff = errors["local"]-errors["mirror"]
    late_diff = errors["late_local"]-errors["late_mirror"]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(steps, values.T, color="#367cb0", alpha=.035, lw=.55, rasterized=True)
    q05, q25, median, q75, q95 = np.quantile(values, [.05, .25, .5, .75, .95], axis=0)
    ax.fill_between(steps, q05, q95, color="#8bbbe0", alpha=.22, label="5-95% of 500 samples")
    ax.fill_between(steps, q25, q75, color="#367cb0", alpha=.30, label="25-75%")
    ax.plot(steps, median, color="black", lw=1.8, label="Median")
    ax.set(title="All 500 actual trajectories", xlabel="Transition end step", ylabel="Output-change ratio q", yscale="log")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    mask = steps >= 5
    normalized = np.exp(logs-desc["log_amplitude"][:, None])
    ax.plot(steps[mask], normalized[:, mask].T, color="#438b67", alpha=.035, lw=.55, rasterized=True)
    lo, med, hi = np.quantile(normalized, [.05, .5, .95], axis=0)
    ax.fill_between(steps[mask], lo[mask], hi[mask], color="#438b67", alpha=.2)
    ax.plot(steps[mask], med[mask], color="black", lw=1.8)
    ax.set(title="After removing each sample's overall scale", xlabel="Transition end step", ylabel="q / sample geometric mean (steps 5-48)", yscale="log")
    ax.text(.03, .04, "Full-curve normalization: visualization only", transform=ax.transAxes, fontsize=8)
    ax = axs[0, 2]
    ax.scatter(desc["residual_asymmetry"], diff, c=np.where(diff >= 0, "#298460", "#c46143"), s=15, alpha=.65)
    ax.axhline(0, color="black", lw=.8)
    ax.set(title="Each dot = one held-out trajectory (n=500)", xlabel="Residual asymmetry (full-curve diagnostic)", ylabel="Local MAPE - mirror MAPE (percentage points)")
    ax.text(.03, .97, "Positive = mirror improves\nNegative = mirror worsens", va="top", transform=ax.transAxes, fontsize=8)
    ax = axs[1, 0]
    for delta, label, color in ((diff, "All original queries", "#367cb0"), (late_diff, "Target steps 36-48", "#dc8b38")):
        ax.plot(np.sort(delta), np.arange(1, len(delta)+1)/len(delta), label=label, color=color)
    ax.axvline(0, color="black", lw=.8)
    ax.set(title="Distribution of per-sample mirror benefit", xlabel="Local MAPE - mirror MAPE (percentage points)", ylabel="Fraction of 500 trajectories")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    rows = [r for r in strata if r["descriptor"] == "residual_asymmetry"]
    for offset, phase, label, color in ((-.16, "all", "All queries", "#367cb0"), (.16, "late_36_48", "Late targets", "#dc8b38")):
        selected = [r for r in rows if r["phase"] == phase]
        ys = np.array([r["absolute_improvement_percentage_points"] for r in selected])
        cis = np.array([r["absolute_improvement_95_interval"] for r in selected])
        ax.errorbar(np.arange(4)+offset, ys, yerr=np.maximum(0, np.array([ys-cis[:, 0], cis[:, 1]-ys])), fmt="o", capsize=4, label=label, color=color)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(4), ["Q1\nMost symmetric", "Q2", "Q3", "Q4\nLeast symmetric"])
    ax.set(title="All samples stratified: 125 per quartile", ylabel="Mean MAPE benefit (pp), 95% paired CI")
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    for offset, phase, label, color in ((-.1, "all", "All queries", "#367cb0"), (.1, "late_36_48", "Late targets", "#dc8b38")):
        rows = [comparisons[p][phase]["vs_geometry"] for p in ("random_1", "random_2", "prompt_order")]
        ys = np.array([r["absolute_improvement_percentage_points"] for r in rows])
        cis = np.array([r["absolute_improvement_95_interval"] for r in rows])
        ax.errorbar(np.arange(3)+offset, ys, yerr=np.maximum(0, np.array([ys-cis[:, 0], cis[:, 1]-ys])), fmt="o", capsize=4, label=label, color=color)
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(3), ["Random A", "Random B", "Prompt-order"])
    ax.set(title="Mirror vs geometry-only control (same 500)", ylabel="Geometry MAPE - mirror MAPE (pp), 95% CI")
    ax.legend(fontsize=8)
    fig.suptitle("Sample coverage expanded from 75 to 500 | Same 1/2/4/8-step horizons and anchors 5-40", fontsize=15)
    fig.savefig(output/"all_500_samples.png", dpi=180)
    fig.savefig(output/"all_500_samples.pdf")
    plt.close(fig)

    # Representative examples are selected by descriptor rank, not by benefit.
    fig, axs = plt.subplots(4, 4, figsize=(15, 12), constrained_layout=True)
    groups = quartiles(desc["residual_asymmetry"])
    selections = []
    for group in range(4):
        members = np.flatnonzero(groups == group)
        members = members[np.argsort(desc["log_amplitude"][members], kind="stable")]
        for col, rank in enumerate((.1, .35, .65, .9)):
            idx = members[int(round(rank*(len(members)-1)))]
            selections.append({"sample": files[idx].name, "asymmetry_quartile": group+1,
                               "amplitude_rank_in_quartile": rank, "index": int(idx)})
            ax = axs[group, col]
            ax.plot(steps, values[idx], color="#367cb0", lw=1.5)
            ax.plot(steps, np.median(values, axis=0), color="gray", ls=":", lw=1.)
            right = steps >= 31
            reflected = np.exp(np.interp(2*desc["training_axis"][idx]-steps[right], steps, logs[idx]))
            ax.plot(steps[right], reflected, "--", color="#dc8b38", lw=1.3)
            sample_id = files[idx].stem.split("_")[0]
            ax.set(yscale="log", ylim=(values.min()*.8, values.max()*1.15),
                   title=f"Q{group+1}, amplitude rank {rank:.0%} | Sample {sample_id}\nSoft-mirror benefit: {diff[idx]:+.3f} pp")
            ax.tick_params(labelsize=7)
            ax.title.set_fontsize(8)
            if group == 3:
                ax.set_xlabel("Transition end step", fontsize=8)
    fig.suptitle("16 descriptor-selected examples from all 500 (not selected by prediction success)\nRows: increasing residual asymmetry | Columns: increasing amplitude\nBlue: actual curve | Orange: direct reflection of left arm | Gray: median of all 500", fontsize=13)
    fig.savefig(output/"stratified_examples.png", dpi=170)
    plt.close(fig)
    (output/"example_selection.json").write_text(json.dumps(selections, indent=2)+"\n")


def previous_sample_coverage(output, files, desc):
    """Compare old test sample coverage with all 500, without selecting models."""
    path = ROOT/"outputs/polynomial_prior/manifest.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text())
    if [p.name for p in files] != [item["name"] for item in old["files"]]:
        raise ValueError("Previous sample manifest does not match current data")
    ids = np.array(old["splits"]["test"], dtype=int)
    result = {"previous_test_count": len(ids), "all_unique_count": len(files), "descriptors": {}}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, name, label in zip(axs, ("log_amplitude", "residual_asymmetry"),
                               ("Overall log amplitude (steps 5-48)", "Residual asymmetry")):
        x = desc[name]
        bins = np.linspace(x.min(), x.max(), 24)
        ax.hist(x, bins=bins, density=True, color="#367cb0", alpha=.25, label="All 500")
        ax.hist(x[ids], bins=bins, density=True, histtype="step", color="#dc8b38", lw=1.8, label=f"Previous {len(ids)} test samples")
        ax.set(xlabel=label, ylabel="Density")
        ax.legend(fontsize=9)
        result["descriptors"][name] = {"quantile_levels": [0,.05,.5,.95,1],
            "all_500": np.quantile(x, [0,.05,.5,.95,1]).tolist(),
            "previous_test": np.quantile(x[ids], [0,.05,.5,.95,1]).tolist(),
            "previous_test_counts_in_population_quartiles": np.bincount(quartiles(x)[ids], minlength=4).tolist()}
    fig.suptitle("Sample distribution: previous 75 vs all 500 | Descriptive comparison only")
    fig.savefig(output/"sample_coverage.png", dpi=180)
    plt.close(fig)
    (output/"previous_sample_coverage.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/videos500_experiments/run_train/lazy_dataset_500")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/mirror_samples")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files, objects, data = load_data(args.data)
    values, steps, _ = data["output_change"]
    assert len(files) == 500
    logs = np.log(values)
    query = original_conditions()
    targets = query[:, 1]+query[:, 2]
    truth = logs[:, targets]
    phases = {"all": np.ones(len(query), dtype=bool), "late_36_48": steps[targets] >= 36}
    protocols = ("random_1", "random_2", "prompt_order")
    forecasts = {p: {m: np.full(truth.shape, np.nan) for m in METHODS} for p in protocols}
    priors = {p: np.full(logs.shape, np.nan) for p in protocols}
    centers = {p: np.full(len(files), np.nan) for p in protocols}
    folds = {p: np.full(len(files), -1) for p in protocols}
    records, fold_metrics = [], []
    for split in make_splits(len(files)):
        protocol, fold = split["protocol"], split["fold"]
        train, val, test = (logs[split[k]] for k in ("train", "validation", "test"))
        axis, _ = symmetry_scan(train, steps)
        center = axis["center_step"]
        prior, operators, choices = select_models(train, val, query, steps, center)
        pred = {"anchor": prior[targets]+(test-prior)[:, query[:, 1]]}
        pred.update({m: prior[targets]+(test-prior) @ op.T for m, op in operators.items()})
        sources = 2*center-steps[targets]
        eligible = (steps[targets] > center) & (sources >= steps[0]) & (sources <= steps[query[:, 1]])
        pred["direct_mirror"] = raw_mirror(test, pred["anchor"], query, steps, center, eligible)
        ids = split["test"]
        assert (folds[protocol][ids] == -1).all()
        folds[protocol][ids] = fold
        priors[protocol][ids] = prior
        centers[protocol][ids] = center
        for method in METHODS:
            forecasts[protocol][method][ids] = pred[method]
            for phase, mask in phases.items():
                fold_metrics.append({"protocol": protocol, "fold": fold, "method": method,
                                     "phase": phase, **scalar_metrics(pred[method][:, mask], test[:, targets[mask]])})
        records.append({"protocol": protocol, "fold": fold, "axis": axis, "models": choices,
                        **{k: split[k].tolist() for k in ("train", "validation", "test")}})
        print(f"{protocol} fold {fold+1}/5: {len(ids)} held-out trajectories; "
              f"local={scalar_metrics(pred['local'],test[:,targets])['mape_percent']:.4f}%, "
              f"mirror={scalar_metrics(pred['mirror'],test[:,targets])['mape_percent']:.4f}%", flush=True)
    errors, comparisons, aggregate = {}, {}, []
    for protocol in protocols:
        assert (folds[protocol] >= 0).all()
        errors[protocol] = {}
        comparisons[protocol] = {}
        for phase, mask in phases.items():
            errors[protocol][phase] = {}
            for method in METHODS:
                pred = forecasts[protocol][method]
                assert np.isfinite(pred).all()
                errors[protocol][phase][method] = 100*np.abs(np.expm1(pred[:, mask]-truth[:, mask])).mean(axis=1)
                aggregate.append({"protocol": protocol, "phase": phase, "method": method,
                                  "unique_test_samples": len(files), **scalar_metrics(pred[:, mask], truth[:, mask])})
            comparisons[protocol][phase] = {"vs_"+m: paired_summary(errors[protocol][phase]["mirror"], errors[protocol][phase][m])
                                            for m in ("anchor", "local", "geometry")}
        np.savez_compressed(args.output/(protocol+"_forecasts.npz"), truth_log=truth,
                            query=query, steps=steps, fold=folds[protocol], prior=priors[protocol],
                            center=centers[protocol], **{"log_"+m: p for m, p in forecasts[protocol].items()})
    desc = descriptors(logs, priors["random_1"], centers["random_1"], steps)
    primary = errors["random_1"]
    strata = []
    for name in ("log_amplitude", "raw_asymmetry", "residual_asymmetry"):
        groups = quartiles(desc[name])
        for group in range(4):
            mask = groups == group
            for phase in phases:
                strata.append({"descriptor": name, "quartile": group+1, "phase": phase,
                               "descriptor_min": float(desc[name][mask].min()),
                               "descriptor_max": float(desc[name][mask].max()),
                               **paired_summary(primary[phase]["mirror"][mask], primary[phase]["local"][mask])})
    rows = []
    for i, (file, obj) in enumerate(zip(files, objects)):
        row = {"file": file.name, "prompt_index": obj["metadata"]["prompt_index"],
               "seed": obj["metadata"]["seed"], "prompt": obj["metadata"]["prompt"],
               **{name: float(value[i]) for name, value in desc.items()}}
        for protocol in protocols:
            row[protocol+"_test_fold"] = int(folds[protocol][i])
            for phase in phases:
                for method in METHODS:
                    row[f"{protocol}__{phase}__{method}_mape_percent"] = float(errors[protocol][phase][method][i])
                row[f"{protocol}__{phase}__mirror_gain_pp"] = float(errors[protocol][phase]["local"][i]-errors[protocol][phase]["mirror"][i])
        rows.append(row)
    save_csv(args.output/"per_trajectory.csv", rows)
    save_csv(args.output/"aggregate_metrics.csv", aggregate)
    save_csv(args.output/"fold_metrics.csv", fold_metrics)
    save_csv(args.output/"strata.csv", strata)
    # Quantify spread in raw curves and after per-sample amplitude removal.
    shape = logs[:, steps >= 5]
    grand = shape.mean()
    sample_offset = shape.mean(axis=1)-grand
    time_effect = shape.mean(axis=0)
    total_ss = np.square(shape-time_effect).sum()
    amplitude_ss = shape.shape[1]*np.square(sample_offset).sum()
    normalized = np.exp(logs-desc["log_amplitude"][:, None])
    spread = []
    for i, step in enumerate(steps):
        quant = np.quantile(values[:, i], [.05, .5, .95])
        nq = np.quantile(normalized[:, i], [.05, .5, .95])
        spread.append({"step": int(step), "q_p05": float(quant[0]), "q_p50": float(quant[1]),
                       "q_p95": float(quant[2]), "p95_over_p05": float(quant[2]/quant[0]),
                       "normalized_p95_over_p05": float(nq[2]/nq[0])})
    save_csv(args.output/"curve_spread.csv", spread)
    summary = {"unique_samples": len(files), "distinct_prompts": len(set(o["metadata"]["prompt"] for o in objects)),
               "distinct_seeds": len(set(o["metadata"]["seed"] for o in objects)),
               "train_validation_test_per_fold": [300, 100, 100], "folds_per_partition": 5,
               "primary_partition": "random_1", "sensitivity_partitions": ["random_2", "prompt_order"],
               "forecast_horizons_unchanged": HORIZONS.tolist(), "anchor_steps_unchanged": [5, 40],
               "queries_per_sample": len(query), "observed_history": "all scalar observations through anchor",
               "matched_phase_gate": "anchor_step >= training-selected mirror axis",
               "amplitude_share_of_between_sample_log_variation_steps_5_48": float(amplitude_ss/total_ss),
               "descriptive_axis_quantiles": np.quantile(desc["descriptive_best_axis"], [0,.05,.5,.95,1]).tolist(),
               "comparisons": comparisons,
               "limitations": ["All partitions reuse the same 500 unique trajectories, not 1500 independent samples.",
                 "Compared with previous 350/75/75 split, fold sizes are 300/100/100; methods, conditions and search grids are preserved.",
                 "All share one model, resolution, frame count, sampler and guidance setting; prompt and seed vary together.",
                 "Only scalar output-change statistics are stored; this is not tensor prediction or cached video generation.",
                 "The dataset has been explored before. Out-of-fold prediction is leakage-controlled internal validation, not a new external dataset.",
                 "Trajectory bootstrap is conditional on fitted OOF models; training-set dependence and selection uncertainty are not fully captured.",
                 "Full-curve amplitude and symmetry quartiles are retrospective diagnostics, not deployable routing features.",
                 "No multiplicity adjustment for subgroup comparisons; do not use them as confirmatory evidence."]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps({"seed": SEED, "splits_and_selected_models": records,
        "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}, indent=2)+"\n")
    make_figures(args.output, values, steps, desc,
                 {**primary["all"], **{"late_"+m: v for m,v in primary["late_36_48"].items()}},
                 comparisons, strata, files)
    previous_sample_coverage(args.output, files, desc)
    print(json.dumps({"primary_all": comparisons["random_1"]["all"]["vs_local"],
                      "primary_late": comparisons["random_1"]["late_36_48"]["vs_local"],
                      "amplitude_share": amplitude_ss/total_ss}, indent=2), flush=True)


if __name__ == "__main__":
    main()
