"""Full held-out trajectories: latent drift, paired videos and video metrics."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from eval.predictor_data import iter_raw_pairs, read_metadata, read_schedule
from eval.predictor_rollout import WanTeacher, rollout
from eval.predictor_schedule import make_wan_scheduler


class TrajectoryDrift:
    """Compare every student state with the full-DiT raw reference.

    Replay saved outputs in a separate solver to recover the terminal latent,
    which raw DiT input records do not include. Never modify the student solver.
    """

    def __init__(self, path, scheduler, guide_scale):
        self.pairs = iter(iter_raw_pairs(path))
        self.scheduler = scheduler
        self.guide_scale = guide_scale
        self.trace = []
        self.final_reference = None

    @torch.no_grad()
    def __call__(self, step, x):
        if step != len(self.trace):
            raise ValueError("drift observer must receive every state in order")
        if step < len(self.scheduler.timesteps):
            pair = next(self.pairs, None)
            if pair is None:
                raise ValueError("incomplete raw reference trajectory")
            for record in pair:
                if float(record["timestep"]) != float(self.scheduler.timesteps[step]):
                    raise ValueError("raw reference timestep differs from solver grid")
            reference = pair[0]["model_input"][0].to(x).float()
            v_c, v_u = [record["model_output"][0].to(x).float() for record in pair]
            guided = v_u + self.guide_scale * (v_c - v_u)
            self.final_reference = self.scheduler.step(
                guided[None], self.scheduler.timesteps[step], reference[None], return_dict=False,
            )[0][0]
        else:
            if step != len(self.scheduler.timesteps) or next(self.pairs, None) is not None:
                raise ValueError("raw reference length differs from solver grid")
            reference = self.final_reference
        if reference is None or reference.shape != x.shape:
            raise ValueError("student/reference latent shapes differ")
        error = x.float() - reference
        if not torch.isfinite(error).all():
            raise FloatingPointError("non-finite trajectory drift")
        self.trace.append({
            "step": step, "sigma": float(self.scheduler.sigmas[step]),
            "mae": float(error.abs().mean()), "rmse": float(error.square().mean().sqrt()),
        })

    def metrics(self):
        if len(self.trace) != len(self.scheduler.timesteps) + 1:
            raise ValueError("drift metrics require a complete trajectory")
        # Exclude the shared initial state from the trajectory average.
        values = [row["mae"] for row in self.trace[1:]]
        return {
            "latent_mae_mean": sum(values) / len(values),
            "latent_mae_max": max(values),
            "latent_mae_final": self.trace[-1]["mae"],
            "latent_rmse_final": self.trace[-1]["rmse"],
        }


def run_trajectory(cfg, generation, runtime, path, model, criterion, optimizer=None, measure_drift=False):
    device = next(model.parameters()).device
    initial = next(iter_raw_pairs(path))[0]["model_input"][0].to(device)
    scheduler = make_wan_scheduler(generation, device, runtime.pipeline.num_train_timesteps)
    sigmas, times = read_schedule(path, generation, int(cfg.data.num_train_timesteps))
    if not torch.equal(scheduler.timesteps.cpu(), times) or not torch.allclose(
        scheduler.sigmas.cpu(), sigmas, rtol=0, atol=1e-7,
    ):
        raise ValueError("installed Wan scheduler differs from the raw training schedule")
    teacher = WanTeacher(
        runtime.pipeline, read_metadata(path)["prompt"], initial.shape,
        offload_text=bool(generation.get("offload_model", True)),
    )
    drift = None
    if measure_drift:
        reference_solver = make_wan_scheduler(generation, device, runtime.pipeline.num_train_timesteps)
        drift = TrajectoryDrift(path, reference_solver, float(generation.sample_guide_scale))
    values, cache, final = rollout(
        model, criterion, teacher, scheduler, initial,
        cache_threshold=float(cfg.cache.threshold), warmup_steps=int(cfg.cache.warmup_steps),
        final_full_steps=int(cfg.cache.final_full_steps), guide_scale=float(generation.sample_guide_scale),
        window_steps=int(cfg.train.rollout_steps), optimizer=optimizer,
        grad_clip=float(cfg.train.grad_clip), on_state=drift,
    )
    return values, cache, final, drift


def validate_predictor(cfg, generation, runtime, paths, model, criterion, output_dir):
    """Explicit validation stage; reuse the existing video quality evaluator."""
    from eval.evaluate_quality import run as evaluate_quality

    model.eval()
    output_dir = Path(output_dir).resolve()
    quality = OmegaConf.load(Path(__file__).parent / "conf" / "quality_predictor.yaml")
    quality.project_root = str(Path(str(cfg.project_root)).expanduser().resolve())
    quality.experiment_root = str(output_dir)
    quality.device = str(cfg.device)
    for key in ("alexnet_path", "i3d_path"):
        path = Path(str(cfg.validation[key])).expanduser()
        if not path.is_absolute():
            path = Path(quality.project_root) / path
        if not path.is_file():
            raise FileNotFoundError(f"validation.{key} must point to local metric weights: {path}")
        quality[key] = str(path)
    runtime.ensure_loaded()
    quality.origins = []
    targets, records = [], []
    for path in paths:
        values, cache, final, drift = run_trajectory(
            cfg, generation, runtime, path, model, criterion, measure_drift=True,
        )
        origin = output_dir / "origin" / f"{path.name}.mp4"
        target = output_dir / "polynomial" / f"{path.name}.mp4"
        if bool(generation.get("offload_model", True)):
            runtime.pipeline.model.cpu()
            torch.cuda.empty_cache()
        with torch.no_grad():
            for destination, latent in [(origin, drift.final_reference), (target, final)]:
                destination.parent.mkdir(parents=True, exist_ok=True)
                video = runtime.pipeline.vae.decode([latent])[0]
                runtime.save_video(
                    tensor=video[None], save_file=str(destination), fps=runtime.wan_config.sample_fps,
                    nrow=1, normalize=True, value_range=(-1, 1),
                )
                del video
        quality.origins.append(str(origin))
        targets.append(str(target))
        metadata = read_metadata(path)
        records.append({
            "trajectory": str(path), "prompt": metadata["prompt"], "seed": metadata["seed"],
            "cache": cache, "prediction_samples": values["samples"],
            "prediction_mae": values["mae"] / values["samples"] if values["samples"] else None,
            "drift": drift.metrics(), "trace": drift.trace,
        })
    quality.targets = [{
        "method": "polynomial", "cache_threshold": float(cfg.cache.threshold), "videos": targets,
    }]
    quality_path = output_dir / "quality.yaml"
    OmegaConf.save(quality, quality_path)
    report = {
        "artifact": str(cfg.init_artifact), "trajectories": records,
        "drift": {key: sum(r["drift"][key] for r in records) / len(records) for key in records[0]["drift"]},
    }
    # Keep drift results even if video evaluation fails; no completed quality claim.
    report_path = output_dir / "validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    runtime.pipeline.model.cpu()
    torch.cuda.empty_cache()
    _, summaries = evaluate_quality(quality_path)
    report["quality"] = summaries
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
