from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eval.artifacts import build_predictor_artifact, load_artifact
from eval.model.easycache import EasyCacheMethod
from eval.model.polynomial import (
    PolynomialLoss, PolynomialMethod, PolynomialPredictor, ResidualHistory,
)
from eval.predictor_data import RawPredictorDataset, find_raw_trajectories, read_schedule, split_by_prompt
from eval.predictor_rollout import rollout
from eval.predictor_schedule import reconstruct_wan_schedule
from eval.predictor_training import fit_predictor
from eval.predictor_validation import TrajectoryDrift, validate_predictor
from eval.pipeline import _generate_video, build_tasks


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def write_raw(root: Path, count=3, steps=9, replayable=False):
    root.mkdir(parents=True, exist_ok=True)
    generation = dict(
        sample_solver="unipc", sample_steps=steps, sample_shift=5.0,
        sample_guide_scale=2.0, image=None,
    )
    OmegaConf.save(OmegaConf.create({"generation": generation}), root / "dataset.yaml")
    sigmas, times = reconstruct_wan_schedule("unipc", steps, 5.0)
    if replayable:
        sigmas, times = EulerScheduler(steps).sigmas, EulerScheduler(steps).timesteps
    for trajectory in range(count):
        path = root / f"trajectory_{trajectory:08d}"
        path.mkdir()
        shards = []
        records = []
        solver = EulerScheduler(steps)
        x = torch.ones(2, 2, 3, 3) * (1 + trajectory * 0.01)
        for step in range(steps):
            if not replayable:
                x = torch.ones(2, 2, 3, 3) * (1 + step * 0.1 + trajectory * 0.01)
            outputs = []
            for branch, scale in [("conditional", 1.0), ("unconditional", 2.0)]:
                residual = scale + 0.2 * sigmas[step] + 0.3 * sigmas[step] ** 3
                v = synthetic_teacher(x, times[step], branch == "conditional") if replayable else x + residual
                outputs.append(v)
                records.append({
                    "step_index": step, "branch": branch, "timestep": float(times[step]),
                    "model_input": (x,), "model_output": (v,),
                })
            if replayable:
                guided = outputs[1] + generation["sample_guide_scale"] * (outputs[0] - outputs[1])
                x = solver.step(guided[None], times[step], x[None])[0][0]
            if step % 2 == 1 or step == steps - 1:
                name = f"shard_{len(shards):04d}.pt"
                torch.save(records, path / name)
                shards.append(name)
                records = []
        metadata = {
            "prompt": f"synthetic {trajectory}", "seed": trajectory, "shards": shards,
        }
        if replayable:
            metadata["scheduler"] = {"sigmas": sigmas.tolist(), "timesteps": times.tolist()}
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "_SUCCESS").touch()
    return OmegaConf.create(generation), sigmas, times


def synthetic_teacher(x, timestep, conditional):
    return 0.2 * x.square() + (1.0 if conditional else 0.5)


class EulerScheduler:
    def __init__(self, steps=9):
        self.sigmas = torch.linspace(1, 0, steps + 1)
        self.timesteps = self.sigmas[:-1] * 1000
        self.model_outputs = [None, None]
        self.last_sample = None
        self.index = 0

    def step(self, v, timestep, x, return_dict=False):
        del timestep, return_dict
        self.model_outputs = [self.model_outputs[-1], v]
        self.last_sample = x
        result = x + (self.sigmas[self.index + 1] - self.sigmas[self.index]) * v
        self.index += 1
        return (result,)


class PolynomialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nonuniform_nonadjacent_history_recovers_quadratic_and_reuse(self):
        history = ResidualHistory()
        x = torch.ones(1, 2, 2, 3, 3)
        for step, sigma in [(0, 0.95), (2, 0.75), (5, 0.40)]:
            history.push(step, sigma, x, x + 1 + 2 * sigma + 3 * sigma**2)
        bases, q = history.features(7, 0.15)
        expected = x + 1 + 2 * 0.15 + 3 * 0.15**2
        self.assertTrue(torch.allclose(x + bases.sum(1), expected, atol=3e-6))
        self.assertTrue(torch.allclose(q, torch.tensor([[0.15, -0.25, -0.35, -0.20, 2]])))
        self.assertTrue(torch.allclose(x + bases[:, 0], x + 1 + 2 * 0.4 + 3 * 0.4**2))
        self.assertEqual([n.step for n in history.nodes], [5, 2, 0])

    def test_model_dimensions_initialization_bounds_and_loss_gradients(self):
        model = PolynomialPredictor(channels=2, hidden_dim=4)
        x = torch.randn(2, 2, 2, 3, 3)
        bases = torch.randn(2, 3, 2, 2, 3, 3)
        q = torch.randn(2, 5)
        prediction, coefficients = model.predict(x, bases, q)
        self.assertEqual(tuple(coefficients.shape), (2, 3))
        self.assertTrue(torch.allclose(coefficients, torch.ones_like(coefficients)))
        self.assertTrue(torch.allclose(prediction, x + bases.sum(1)))
        target = torch.randn_like(x, requires_grad=True)
        PolynomialLoss()(prediction, target, coefficients).backward()
        self.assertIsNone(target.grad)
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.head[-1].bias.copy_(torch.tensor([-100.0, 0.0, 100.0]))
        values = model(x, bases, q)
        self.assertTrue((values.abs() <= model.a_max).all())
        self.assertTrue((values[:, 0] < 0).all())

    def test_cache_branches_and_history_only_advance_on_full_refresh(self):
        model = PolynomialPredictor(channels=2, hidden_dim=4)
        method = PolynomialMethod(model=model, cache_threshold=100)
        method.reset(8, 3, 1)
        sigmas = torch.tensor([1., .94, .83, .72, .60, .43, .21, .1, 0.])
        times = sigmas[:-1] * 1000
        method.set_schedule(sigmas, times)
        for step in range(8):
            method.cache_threshold = 0 if step == 5 else 100
            x = torch.ones(2, 2, 3, 3) * (1 + step * .1)
            for branch, offset in [(True, 1.0), (False, 2.0)]:
                cached = method.try_skip([x], times[step])
                truth = x + offset + sigmas[step] + sigmas[step] ** 2
                if cached is None:
                    method.update([x], [truth])
                else:
                    self.assertTrue(torch.allclose(cached[0], truth, atol=2e-5))
                    self.assertNotIn(step, [n.step for n in method.histories[branch].nodes])
            if step == 6:
                self.assertEqual([n.step for n in method.histories[True].nodes], [5, 2, 1])
        self.assertEqual(method.summary()["skipped_pair_indices"], [3, 4, 6])
        self.assertEqual(method.summary()["calculated_pair_indices"], [0, 1, 2, 5, 7])
        method.reset(8, 3, 1)
        self.assertFalse(method.histories[True].nodes)
        self.assertIsNone(method.sigmas)

    def test_missing_or_mismatched_sigma_grid_fails(self):
        method = PolynomialMethod(model=PolynomialPredictor(2, 4))
        with self.assertRaises(ValueError):
            method.reset(8, 2, 1)
        method.reset(8, 3, 1)
        with self.assertRaises(RuntimeError):
            method.try_skip([torch.ones(2, 2, 3, 3)], torch.tensor([1000]))
        method.set_schedule(torch.linspace(1, 0, 9), torch.arange(8, 0, -1))
        with self.assertRaises(ValueError):
            method.try_skip([torch.ones(2, 2, 3, 3)], torch.tensor([1000]))

    def test_raw_dataset_samples_history_without_gate_and_checks_recorded_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, sigmas, times = write_raw(root, count=2)
            paths, loaded_generation = find_raw_trajectories(root)
            data = RawPredictorDataset(paths[:1], loaded_generation, 2)
            with patch.object(EasyCacheMethod, "try_skip", side_effect=AssertionError("no gate in pretraining")):
                samples = list(data)
            self.assertEqual([s["step"] for s in samples], [3,3,4,4,5,5,6,6,7,7,8,8])
            for sample in samples:
                i, j, k = sample["history_steps"]
                t = sample["step"]
                self.assertTrue(k < j < i < t)
                self.assertEqual(tuple(sample["x"].shape), (2, 2, 3, 3))
                expected_q = torch.tensor([sigmas[t], sigmas[t]-sigmas[i], sigmas[i]-sigmas[j], sigmas[j]-sigmas[k], t-i])
                self.assertTrue(torch.equal(sample["q"], expected_q))
                scale = 1 + sample["branch"]
                residuals = {r: scale + .2*sigmas[r] + .3*sigmas[r]**3 for r in (i,j,k)}
                first = (residuals[i]-residuals[j])/(sigmas[i]-sigmas[j])
                second = (first-(residuals[j]-residuals[k])/(sigmas[j]-sigmas[k]))/(sigmas[i]-sigmas[k])
                expected = torch.stack([residuals[i], (sigmas[t]-sigmas[i])*first,
                                        (sigmas[t]-sigmas[i])*(sigmas[t]-sigmas[j])*second])
                self.assertTrue(torch.allclose(sample["bases"], expected[:,None,None,None,None].expand_as(sample["bases"]), atol=1e-5))
            validation_nodes = [s["history_steps"] for s in samples]
            data.epoch = 3
            self.assertEqual(validation_nodes, [s["history_steps"] for s in data])
            train = RawPredictorDataset(paths[:1], loaded_generation, 2, shuffle=True)
            nodes_0 = {(s["step"], s["history_steps"]) for s in train}
            train.epoch = 1
            self.assertNotEqual(nodes_0, {(s["step"], s["history_steps"]) for s in train})
            # Float sigmas are not the rounded model times / 1000.
            self.assertFalse(torch.allclose(sigmas[:-1], times.float() / 1000))
            metadata = json.loads((paths[0] / "metadata.json").read_text())
            metadata["scheduler"] = {"sigmas": sigmas.tolist(), "timesteps": times.tolist()}
            (paths[0] / "metadata.json").write_text(json.dumps(metadata))
            self.assertTrue(torch.equal(read_schedule(paths[0], generation)[0], sigmas))
            shard = paths[0] / metadata["shards"][0]
            records = torch.load(shard, weights_only=True)
            records[0]["timestep"] += 1
            torch.save(records, shard)
            with self.assertRaisesRegex(ValueError, "timestep differs"):
                list(data)

    def test_split_groups_same_prompt_across_ids_and_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_raw(root, count=6)
            paths, _ = find_raw_trajectories(root)
            for index, path in enumerate(paths):
                metadata = json.loads((path / "metadata.json").read_text())
                metadata.update(prompt=f"prompt {index % 3}", prompt_id=index)
                (path / "metadata.json").write_text(json.dumps(metadata))
            train, validation = split_by_prompt(paths, 0.34, 7)
            self.assertEqual((len(train), len(validation)), (4, 2))
            groups = lambda selected: {json.loads((p / "metadata.json").read_text())["prompt"] for p in selected}
            self.assertFalse(groups(train) & groups(validation))
            self.assertEqual((train, validation), split_by_prompt(list(reversed(paths)), 0.34, 7))
            with self.assertRaisesRegex(ValueError, "two distinct prompts"):
                split_by_prompt([paths[0], paths[3]], 0.2, 0)

    def test_refresh_rate_updates_even_when_full_inputs_are_identical(self):
        method = EasyCacheMethod(cache_threshold=.05)
        method.reset(5, 3, 0)
        for x, v in [(1., 10.), (2., 10.), (2., 20.)]:
            for _ in range(2):
                inputs = [torch.tensor([x])]
                self.assertIsNone(method.try_skip(inputs, torch.tensor([1.])))
                method.update(inputs, [torch.tensor([v])])
        self.assertAlmostEqual(method.transformation_rate, 10 / method.epsilon)
        self.assertIsNone(method.try_skip([torch.tensor([2.01])], torch.tensor([1.])))

    def test_offline_training_artifact_and_sweep_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_raw(root / "raw")
            with initialize_config_dir(config_dir=str(PACKAGE_ROOT / "conf"), version_base="1.3"):
                cfg = compose(config_name="train_predictor", overrides=[
                    f"data.data_dir={root / 'raw'}", f"output_dir={root / 'out'}",
                    "model.channels=2", "model.hidden_dim=4", "device=cpu",
                    "cache.warmup_steps=9", "cache.threshold=0", "train.epochs=2",
                ])
                sweep = compose(config_name="sweep_predictor")
            fit_predictor(cfg, root / "out")
            artifact = root / "out/model.pth"
            saved = load_artifact(artifact)
            self.assertEqual(saved["kind"], "residual_polynomial")
            self.assertFalse(torch.equal(saved["model_state_dict"]["head.2.weight"], torch.zeros(3, 4)))
            loaded = PolynomialMethod(artifact_path=str(artifact), device="cpu")
            self.assertEqual(loaded.model.channels, 2)
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertFalse(set(summary["train_trajectories"]) & set(summary["val_trajectories"]))
            self.assertFalse(set(summary["prompt_split"]["train"]) & set(summary["prompt_split"]["validation"]))
            self.assertEqual(summary["val_samples"], 12)
            self.assertEqual([m.name for m in sweep.methods], ["origin", "easycache", "d2cache_output", "polynomial"])
            self.assertIn("val_quadratic_mae", (root / "out/history.csv").read_text())

    def test_rollout_gradients_cross_skips_but_teacher_probes_do_not_refresh(self):
        model = PolynomialPredictor(2, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = EulerScheduler(9)
        seen = []
        def teacher(x, timestep, conditional):
            seen.append((float(timestep), conditional, x.detach().clone(), x.requires_grad))
            return 0.2 * x.square() + (1.0 if conditional else 0.5)
        initial_weights = model.head[-1].weight.detach().clone()
        totals, summary, final = rollout(
            model, PolynomialLoss(), teacher, scheduler, torch.ones(2,2,3,3),
            cache_threshold=100, warmup_steps=3, final_full_steps=1,
            guide_scale=2, window_steps=2, optimizer=optimizer,
        )
        self.assertEqual(summary["calculated_pair_indices"], [0, 1, 2, 8])
        self.assertEqual(summary["skipped_pair_indices"], [3, 4, 5, 6, 7])
        self.assertEqual(totals["samples"], 10)
        self.assertEqual(len(seen), 18)
        self.assertFalse(any(entry[3] for entry in seen))
        self.assertFalse(torch.equal(initial_weights, model.head[-1].weight))
        self.assertTrue(torch.isfinite(final).all())
        self.assertFalse(final.requires_grad)
        self.assertFalse(scheduler.last_sample.requires_grad)
        self.assertTrue(all(not v.requires_grad for v in scheduler.model_outputs))

    def test_complete_trajectory_drift_includes_terminal_state_and_survives_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_raw(root, count=1, replayable=True)
            path = find_raw_trajectories(root)[0][0]
            model = PolynomialPredictor(2, 4)
            with torch.no_grad():
                model.head[-1].bias.copy_(torch.atanh(torch.tensor([1., 0., 0.]) / model.a_max))
            for threshold in (0, 100):
                observer = TrajectoryDrift(path, EulerScheduler(), guide_scale=2)
                totals, summary, final = rollout(
                    model, PolynomialLoss(), synthetic_teacher, EulerScheduler(), torch.ones(2,2,3,3),
                    cache_threshold=threshold, warmup_steps=3, final_full_steps=1,
                    guide_scale=2, on_state=observer,
                )
                self.assertEqual([row["step"] for row in observer.trace], list(range(10)))
                self.assertEqual(observer.trace[-1]["sigma"], 0)
                self.assertAlmostEqual(observer.metrics()["latent_mae_final"],
                                       float((final - observer.final_reference).abs().mean()))
                if threshold == 0:
                    self.assertEqual(observer.metrics()["latent_mae_max"], 0)
                    self.assertEqual(totals["samples"], 0)
                else:
                    self.assertEqual(summary["calculated_pair_indices"][-1], 8)
                    self.assertGreater(observer.metrics()["latent_mae_final"], 0)
            with self.assertRaisesRegex(ValueError, "between 1 and 4"):
                rollout(model, PolynomialLoss(), synthetic_teacher, EulerScheduler(), torch.ones(2,2,3,3),
                        cache_threshold=100, warmup_steps=3, final_full_steps=1, guide_scale=2, window_steps=5)

    def test_full_validation_saves_paired_videos_drift_and_runs_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, _, _ = write_raw(root / "raw", count=2, replayable=True)
            paths, _ = find_raw_trajectories(root / "raw")
            metric_weights = root / "metric.pth"
            metric_weights.touch()
            output = root / "validation"
            with initialize_config_dir(config_dir=str(PACKAGE_ROOT / "conf"), version_base="1.3"):
                cfg = compose(config_name="train_predictor", overrides=[
                    "device=cpu", "cache.warmup_steps=3", "cache.threshold=100",
                    f"validation.alexnet_path={metric_weights}", f"validation.i3d_path={metric_weights}",
                ])
            model = PolynomialPredictor(2, 4)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            runtime = SimpleNamespace(
                ensure_loaded=Mock(), wan_config=SimpleNamespace(sample_fps=16),
                pipeline=SimpleNamespace(
                    model=torch.nn.Linear(1,1), num_train_timesteps=1000,
                    vae=SimpleNamespace(decode=lambda latents: [latents[0]]),
                ),
                save_video=lambda **kwargs: Path(kwargs["save_file"]).write_bytes(b"video"),
            )
            with (
                patch("eval.predictor_validation.make_wan_scheduler", side_effect=lambda *args: EulerScheduler()),
                patch("eval.predictor_validation.WanTeacher", return_value=synthetic_teacher),
                patch("eval.evaluate_quality.run", return_value=([], [{"psnr_mean": 30.}])) as quality,
            ):
                report = validate_predictor(cfg, generation, runtime, paths, model, PolynomialLoss(), output)
            quality.assert_called_once_with((output / "quality.yaml").resolve())
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
            self.assertEqual(len(report["trajectories"]), 2)
            self.assertEqual(report["quality"][0]["psnr_mean"], 30.)
            self.assertEqual(len(report["trajectories"][0]["trace"]), 10)
            written = OmegaConf.load(output / "quality.yaml")
            self.assertEqual(len(written.origins), 2)
            self.assertEqual(len(written.targets[0].videos), 2)
            self.assertTrue(all(Path(p).is_file() for p in list(written.origins) + list(written.targets[0].videos)))
            self.assertIn("latent_mae_final", json.loads((output / "validation.json").read_text())["drift"])

    def test_predictor_quality_config_matches_every_sweep_target(self):
        from eval.evaluate_quality import load_pairs

        with initialize_config_dir(config_dir=str(PACKAGE_ROOT / "conf"), version_base="1.3"):
            sweep = compose(config_name="sweep_predictor", overrides=[f"prompts.file={PACKAGE_ROOT / 'prompts.txt'}"])
        tasks = build_tasks(sweep, PACKAGE_ROOT.parent)
        _, pairs, _ = load_pairs(PACKAGE_ROOT / "conf/quality_predictor.yaml", preview=True)
        output_root = (PACKAGE_ROOT.parent / str(sweep.paths.output_root)).resolve()
        generated = {(t.method, float(t.cache_threshold), str(output_root / t.output_group / f"{t.prompt_id}.mp4")) for t in tasks if t.method != "origin"}
        evaluated = {(p.method, float(p.cache_threshold), str(p.target_video)) for p in pairs}
        self.assertEqual(generated, evaluated)

    def test_generation_pipeline_loads_predictor_and_supplies_solver_sigmas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "predictor.pth"
            model_config = {
                "_target_": "eval.model.PolynomialPredictor", "channels": 2,
                "hidden_dim": 4, "a_max": 2.0,
            }
            torch.save(build_predictor_artifact(PolynomialPredictor(2,4), model_config, 100), artifact)
            with initialize_config_dir(config_dir=str(PACKAGE_ROOT / "conf"), version_base="1.3"):
                cfg = compose(config_name="sweep_predictor", overrides=[
                    f"prompts.file={PACKAGE_ROOT / 'prompts.txt'}",
                    f"paths.predictor_artifact={artifact}",
                    "generation.sample_steps=9", "generation.warmup_steps=3",
                ])
            task = next(t for t in build_tasks(cfg, root) if t.method == "polynomial")
            task = replace(task, cache_threshold=100)
            scheduler = EulerScheduler(9)

            class FakeDiT(torch.nn.Module):
                def forward(self, x, t, context, seq_len):
                    del context, seq_len
                    sigma = float(t.flatten()[0]) / 1000
                    return [value + 1 + sigma + sigma**2 for value in x]

            dit = FakeDiT()
            original_forward = dit.forward
            def generate(*args, **kwargs):
                del args, kwargs
                with torch.no_grad():
                    for step, t in enumerate(scheduler.timesteps):
                        x = torch.ones(2,2,3,3) * (1 + .1 * step)
                        for _ in range(2):
                            value = dit([x], t[None], [], 1)[0]
                            self.assertTrue(torch.isfinite(value).all())
                return torch.zeros(1)
            runtime = SimpleNamespace(
                pipeline=SimpleNamespace(model=dit, generate=generate, num_train_timesteps=1000),
                size_configs={str(cfg.generation.size): (1,1)},
                max_area_configs={str(cfg.generation.size): 1},
                wan_config=SimpleNamespace(sample_fps=16),
                save_video=lambda **kwargs: Path(kwargs["save_file"]).write_bytes(b"video"),
            )
            with (
                patch("eval.predictor_schedule.make_wan_scheduler", return_value=scheduler) as make,
                patch("eval.pipeline.torch.cuda.synchronize"),
            ):
                result = _generate_video(task, cfg, runtime, root, root/"out.mp4", root/"out.log")
            make.assert_called_once()
            self.assertEqual(dit.forward, original_forward)
            self.assertEqual(result["cache"]["calculated_pair_indices"], [0,1,2,8])
            self.assertEqual(result["cache"]["skipped_pair_indices"], [3,4,5,6,7])
            self.assertEqual(result["timing"]["dit_forward_calls"], 8)
            self.assertEqual(result["timing"]["cache_forward_calls"], 10)


if __name__ == "__main__":
    unittest.main()
