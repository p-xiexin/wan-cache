"""Fit/export scalar piecewise polynomial coefficients; no GPU or neural net.

Five-fold nested holdout: 300 train / 100 validation / 100 test per fold.
Final artifact is refitted on all 500 only after OOF predictions are saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from polynomial_prior import (OnlineFitConfig, OnlinePolynomial, PiecewisePolynomialPrior,
                              correction_weights, fit_piecewise)
from analyze.analyze_polynomial_prior import SEED, HORIZONS, load_data, metrics, save_csv
from analyze.analyze_mirror_prior import symmetry_scan


KNOT_SETS = (
    (1, 4, 8, 16, 24, 32, 40, 44, 48),
    (1, 4, 8, 16, 24, 26, 28, 32, 40, 44, 48),
    (1, 3, 5, 8, 12, 16, 20, 24, 26, 28, 32, 36, 40, 44, 46, 48),
)
OPTIONS = {
    "prior_anchor": dict(use_online_fit=False, use_mirror_node=False),
    "online_fit": dict(use_online_fit=True, use_mirror_node=False),
    "online_mirror": dict(use_online_fit=True, use_mirror_node=True),
    "geometry_control": dict(use_online_fit=True, use_mirror_node=True, mirror_node_mode="anchor"),
}


def forecast(logs, prior, steps, anchors, center, config, options):
    result = np.empty((len(logs), len(anchors), len(HORIZONS)))
    for ai, anchor in enumerate(anchors):
        targets = anchor+HORIZONS
        w = correction_weights(steps[:anchor+1], steps[targets], center, config,
                               domain_end=steps[-1], **options)
        result[:, ai] = prior[targets] + (logs[:, :anchor+1]-prior[:anchor+1]) @ w.T
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data/videos500_experiments/run_train/lazy_dataset_500")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/piecewise_polynomial")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files, objects, data = load_data(args.data)
    values, steps, times = data["output_change"]
    logs = np.log(values)
    if len(files) != 500:
        raise ValueError("this experiment expects the supplied 500 trajectories")
    anchors = np.arange(4, 40)
    targets = anchors[:, None]+HORIZONS
    truth = logs[:, targets]
    splits = np.array_split(np.random.default_rng(SEED+101).permutation(len(files)), 5)
    predictions = {name: np.full(truth.shape, np.nan) for name in OPTIONS}
    selections, fold_metrics, search = [], [], []
    for fold in range(5):
        test_ids, val_ids = splits[fold], splits[(fold+1) % 5]
        train_ids = np.concatenate([s for j,s in enumerate(splits) if j not in (fold,(fold+1) % 5)])
        train, val, test = logs[train_ids], logs[val_ids], logs[test_ids]
        center = symmetry_scan(train, steps)[0]["center_step"]
        best = None
        for breaks in KNOT_SETS:
            for roughness in (0., .001, .01, .1, 1.):
                prior, spec = fit_piecewise(train, steps, breaks, roughness)
                pred = forecast(val, prior, steps, anchors, center, OnlineFitConfig(), OPTIONS["prior_anchor"])
                score = metrics(pred, val[:, targets])["male"]
                search.append({"fold": fold, "family": "offline", "break_steps": breaks,
                               "roughness": roughness, "validation_male": score})
                if best is None or score < best[0]:
                    best = score, prior, spec
        _, prior, spec = best
        best = None
        for before in (.1, 1., 10.):
            for after in (.01, .1, 1.):
                config = OnlineFitConfig(ridge_before_axis=before, ridge_after_axis=after)
                pred = forecast(val, prior, steps, anchors, center, config, OPTIONS["online_fit"])
                score = metrics(pred, val[:, targets])["male"]
                search.append({"fold": fold, "family": "online", **asdict(config), "validation_male": score})
                if best is None or score < best[0]:
                    best = score, config
        _, config = best
        best = None
        for weight in (.001, .01, .1, 1.):
            candidate = replace(config, mirror_weight=weight)
            pred = forecast(val, prior, steps, anchors, center, candidate, OPTIONS["online_mirror"])
            score = metrics(pred, val[:, targets])["male"]
            search.append({"fold": fold, "family": "mirror", **asdict(candidate), "validation_male": score})
            if best is None or score < best[0]:
                best = score, candidate
        _, config = best
        for name, options in OPTIONS.items():
            pred = forecast(test, prior, steps, anchors, center, config, options)
            predictions[name][test_ids] = pred
            fold_metrics.append({"fold": fold, "method": name, **metrics(pred, test[:, targets])})
        selections.append({"fold": fold, "train": train_ids.tolist(), "validation": val_ids.tolist(),
                           "test": test_ids.tolist(), "center": center, "polynomial": spec, "online": asdict(config)})
        print(f"fold {fold+1}/5: 100 held-out trajectories; "
              f"prior={metrics(predictions['prior_anchor'][test_ids],test[:,targets])['mape_percent']:.4f}%, "
              f"online={metrics(predictions['online_fit'][test_ids],test[:,targets])['mape_percent']:.4f}%, "
              f"mirror={metrics(predictions['online_mirror'][test_ids],test[:,targets])['mape_percent']:.4f}%", flush=True)
    for pred in predictions.values():
        assert np.isfinite(pred).all()
    # Select final settings by the mode of validation-selected configurations.
    # OOF test losses never participate in this choice.
    chosen = Counter((tuple(s["polynomial"]["break_steps"]),s["polynomial"]["roughness"]) for s in selections).most_common(1)[0][0]
    cfg_string = Counter(json.dumps(s["online"], sort_keys=True) for s in selections).most_common(1)[0][0]
    final_config = OnlineFitConfig(**json.loads(cfg_string))
    full_prior, spec = fit_piecewise(logs, steps, *chosen)
    axis = symmetry_scan(logs, steps)[0]["center_step"]
    generation_keys = ("task", "size", "frame_num", "sampling_steps", "sample_solver", "sample_shift", "guide_scale")
    artifact = {"kind": "piecewise_log_change_prior", "version": 1,
                "target": "q_s = mean(abs(v_s-v_(s-1))) / (mean(abs(v_(s-1))) + 1e-8)",
                "target_scope": "scalar conditional output-change amplitude, not a DiT tensor",
                "coordinate": "0-based denoising transition end step; supported steps 1..48",
                "generation": {key: objects[0]["metadata"][key] for key in generation_keys},
                "step_grid": steps.tolist(), "model_timestep_grid": times.tolist(),
                "polynomial": spec, "mirror_axis_step": axis, "online_fit": asdict(final_config),
                "training": {"trajectories": len(files), "refit_on_all_data": True,
                    "hyperparameter_selection": "mode of five validation-selected configurations; test losses not used",
                    "evaluation": "separate OOF models, not the final all-data fit", "seed": SEED},
                "observation_contract": "observe(s,q_s) requires actual outputs at BOTH s-1 and s; a gap-spanning difference is not q_s",
                "mirror_contract": "reflect log deviations from the fixed prior; enable only after last real node crosses the train-selected axis",
                "anchor_contract": "latest real value is exact with either switch setting; online_fit adds slope/curvature correction",
                "runtime_contract": "does not schedule DiT calls, update tensor outputs, or update a neural network"}
    write_json(output/"prior.json", artifact)
    runtime_prior = PiecewisePolynomialPrior.load(output/"prior.json")
    np.testing.assert_allclose(runtime_prior.log_value(steps), full_prior, atol=1e-10)
    # The exported ordinary coefficients, not a cached training array, must
    # reproduce all four ablations in the runtime interface.
    for name, options in OPTIONS.items():
        runtime = OnlinePolynomial(runtime_prior, **options)
        for step, q in zip(steps[:30], values[0,:30]):
            runtime.observe(step, q)
        actual = runtime.predict_log(steps[29+HORIZONS])
        expected = forecast(logs[:1], full_prior, steps, [29], axis, final_config, options)[0,0]
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    all_metrics = {name: metrics(pred, truth) for name,pred in predictions.items()}
    late = steps[targets] >= 36
    late_metrics = {name: metrics(pred[:,late],truth[:,late]) for name,pred in predictions.items()}
    summary = {"unique_test_trajectories": 500, "queries_per_trajectory": int(np.prod(targets.shape)),
               "fold_sizes": {"train":300,"validation":100,"test":100},
               "offline_training": "linear least squares for continuous piecewise cubic coefficients",
               "same_prior_and_online_parameters_for_all_ablations": True,
               "oof_metrics": all_metrics, "late_36_48_oof_metrics": late_metrics,
               "final_refit_segments": len(spec["coefficients_ascending"]),
               "stored_polynomial_coefficients": int(np.size(spec["coefficients_ascending"])),
               "selected_offline": {"break_steps": spec["break_steps"], "roughness": spec["roughness"]},
               "selected_online": asdict(final_config), "mirror_axis_step": axis,
               "limitations": ["This is a scalar shape prior, not a learned output-tensor predictor.",
                   "OOF evaluation observes a dense true prefix; server sparse-node and closed-loop behavior are untested.",
                   "Data were previously explored; this is internal validation, not an untouched external test set.",
                   "No video quality, tensor direction, wall-clock speedup or new sample generation is claimed.",
                   "Prior is calibrated to the saved 50-step generation configuration; no silent cross-schedule extrapolation.",
                   "Mirror node values are soft hypotheses and never enter the true observation history."]}
    write_json(output/"summary.json", summary)
    write_json(output/"manifest.json", {"folds": selections, "files": [
        {"name": p.name, "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
         "prompt_index":o["metadata"]["prompt_index"],"seed":o["metadata"]["seed"]}
        for p,o in zip(files,objects)]})
    write_json(output/"validation_search.json", search)
    write_json(output/"ablations.json", {name: {"prior_path":"prior.json",**options} for name,options in OPTIONS.items()})
    save_csv(output/"fold_metrics.csv", fold_metrics)
    save_csv(output/"per_trajectory.csv", [{"file":p.name, **{name+"_mape_percent":float(
        100*np.abs(np.expm1(pred[i]-truth[i])).mean()) for name,pred in predictions.items()}} for i,p in enumerate(files)])
    np.savez_compressed(output/"oof_forecasts.npz",truth_log=truth,anchors=anchors,steps=steps,
                        horizons=HORIZONS,**{"log_"+name:pred for name,pred in predictions.items()})
    np.savez_compressed(output/"coefficients.npz",break_steps=spec["break_steps"],
                        coefficients_ascending=spec["coefficients_ascending"])
    fig, axs = plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    axs[0].plot(steps,values.T,color="#367cb0",alpha=.03,lw=.5)
    axs[0].plot(steps,np.exp(logs.mean(axis=0)),"k--",lw=1.5,label="Mean of 500 log trajectories")
    dense=np.linspace(steps[0],steps[-1],600)
    axs[0].plot(dense,runtime_prior.value(dense),color="#c77c2b",lw=2,label="Exported piecewise cubic")
    for b in spec["break_steps"][1:-1]: axs[0].axvline(b,color="gray",alpha=.2,lw=.6)
    axs[0].set(yscale="log",xlabel="Transition end step",ylabel="Output-change ratio q",title="Final prior refitted on all 500")
    axs[0].legend(fontsize=8)
    labels={"prior_anchor":"Prior + anchor","online_fit":"+ online fit","online_mirror":"+ soft mirror nodes","geometry_control":"Geometry-only control"}
    for name,pred in predictions.items():
        ys=[metrics(pred[:,:,hi],truth[:,:,hi])["mape_percent"] for hi in range(len(HORIZONS))]
        axs[1].plot(HORIZONS,ys,"o-",label=labels[name])
    axs[1].set(xlabel="Forecast horizon",ylabel="MAPE (%)",title="500 held-out trajectories across five folds",xticks=HORIZONS)
    axs[1].legend(fontsize=8)
    fig.savefig(output/"fit_and_ablation.png",dpi=180)
    plt.close(fig)
    instructions = '''分段多项式先验实验包

本地拟合对象：log(q)，q 是相邻完整 DiT 输出的相对变化幅度。
训练是最小二乘求分段三次多项式系数，不训练神经网络，不需要原始张量。
prior.json 是在模型选择结束后用全部 500 条曲线重拟合的部署先验。
summary.json / oof_forecasts.npz 是五折留出模型的评估结果，不是部署先验的独立测试。

推理端只需要 numpy：
    from polynomial_prior import PiecewisePolynomialPrior, OnlinePolynomial
    prior = PiecewisePolynomialPrior.load("prior.json")
    predictor = OnlinePolynomial(prior, use_online_fit=True, use_mirror_node=True)
    predictor.observe(step, measured_q)  # 只接受真实观测
    predicted_q = predictor.predict(future_step)

ablations.json 保存四组开关配置。关闭 online_fit 时仍用最新真实节点作幅值对齐。
mirror_node_mode="anchor" 只增加同位置约束，虚拟节点不读取镜像历史值。
镜像只在真实锚点越过训练得到的对称轴后启用，预测点不会写入真实历史。
全部组共享同一先验、拟合阶数、窗口和正则参数；mirror 仅增加弱镜像约束。

坐标是当前 50 步配置的推理 step，支持 1..48。分段局部 z=(step-left)/(right-left)。
每段 log(q)=c0+c1*z+c2*z^2+c3*z^3，q=exp(log(q))；段间 C2 连续。
不会在训练范围之外自动外推；换步数、模型或 scheduler 应重新校准。
q_s 的真实观测需要完整 v_(s-1) 和 v_s。跨多个步骤的差值不能当作 q_s。

这份包实现系数先验和在线标量校正，不包含刷新调度或完整张量重建。
服务器中的张量预测仍需使用真实 DiT 节点提供方向和幅值，并单独验证。
重新拟合：python analyze/train_piecewise_polynomial.py --data /path/to/lazy_dataset_500 --output /path/to/output
训练依赖 numpy、torch、matplotlib；部署拟合器仅依赖 numpy。
'''
    (output/"README.txt").write_text(instructions)
    archive=output/"piecewise_polynomial_bundle.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.is_file() and p != archive: z.write(p,p.name)
        z.write(ROOT/"polynomial_prior.py","polynomial_prior.py")
        for name in ("train_piecewise_polynomial.py","analyze_polynomial_prior.py","analyze_mirror_prior.py"):
            z.write(ROOT/"analyze"/name,"analyze/"+name)
        z.writestr("requirements-runtime.txt","numpy>=1.26\n")
        z.writestr("requirements-training.txt","numpy>=1.26\ntorch>=2.1\nmatplotlib>=3.8\n")
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Saved: {archive}",flush=True)


if __name__ == "__main__":
    main()
