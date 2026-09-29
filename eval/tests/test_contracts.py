from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eval.artifacts import build_artifact
from eval.cache_hook import patch_forward
from eval.evaluate import run as run_evaluation
from eval.evaluate_quality import (
    SUMMARY_FIELDS,
    VIDEO_FIELDS,
    FVDComputer,
    MetricComputer,
    VideoPair,
    _load_alexnet_weights,
    _read_fvd_clip,
    evaluate_fvd,
    evaluate_pair,
    frechet_distance,
    load_pairs,
    summarize as summarize_quality,
)
from eval.pipeline import (
    ForwardTiming,
    WanWorkerRuntime,
    _generate_video,
    _pending_tasks,
    _resolve_num_processes,
    _run_worker,
    build_tasks,
    run_task,
)
from eval.model import (
    CacheMethod,
    CacheModel,
    CumulativeMethod,
    D2CacheMethod,
    EasyCacheMethod,
    MagCacheMethod,
    TemporalMethod,
    TemporalModel,
)
from eval.summarize import run as summarize_performance


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_performance_result(
    output_root: Path,
    output_group: str,
    run_id: str,
    prompt_id: str,
    method: str,
    cache_threshold: float | None,
    generation_seconds: float,
    skipped_pair_indices: list[int],
) -> None:
    output_dir = output_root / output_group
    output_dir.mkdir(parents=True, exist_ok=True)
    dit_forward_seconds = generation_seconds * 0.8
    dit_forward_calls = 100 - 2 * len(skipped_pair_indices)
    cache_forward_seconds = generation_seconds * 0.02 * len(
        skipped_pair_indices
    )
    cache_forward_calls = 2 * len(skipped_pair_indices)
    result = {
        "status": "complete",
        "run_id": run_id,
        "prompt_id": prompt_id,
        "method": method,
        "cache_threshold": cache_threshold,
        "timing": {
            "generation_seconds": generation_seconds,
            "dit_forward_seconds": dit_forward_seconds,
            "dit_forward_calls": dit_forward_calls,
            "cache_forward_seconds": cache_forward_seconds,
            "cache_forward_calls": cache_forward_calls,
        },
        "cache": {
            "skipped_pairs": len(skipped_pair_indices),
            "skipped_pair_indices": skipped_pair_indices,
            "skip_ratio_all_pairs": len(skipped_pair_indices) / 50,
        },
    }
    (output_dir / f"{prompt_id}.result.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )


def write_direct_quality_config(
    root: Path,
    origins: list[str],
    targets: list[object],
) -> Path:
    cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "quality.yaml")
    cfg.project_root = str(root)
    cfg.output_dir = str(root / "quality")
    cfg.origins = origins
    cfg.targets = targets
    path = root / "quality.yaml"
    OmegaConf.save(cfg, path)
    return path


def labeled_target(
    method: str,
    videos: list[str],
    cache_threshold: float | None = None,
) -> dict:
    return {
        "method": method,
        "cache_threshold": cache_threshold,
        "videos": videos,
    }


class ScriptedMethod(CacheMethod):
    name = "scripted"

    def __init__(self) -> None:
        self.decisions = iter([False, True, False])

    def decide_conditional(self, raw_input, timestep):
        del raw_input, timestep
        return next(self.decisions)


class HistoryMethod(CacheMethod):
    name = "history"

    def __init__(self) -> None:
        self.decisions = iter([False, True, True, False])
        self.seen_input_history: list[tuple[float | None, float | None]] = []

    def decide_conditional(self, raw_input, timestep):
        del raw_input, timestep
        previous = (
            None
            if self.previous_raw_input_even is None
            else float(self.previous_raw_input_even[0].item())
        )
        previous_previous = (
            None
            if self.prev_previous_raw_input_even is None
            else float(self.prev_previous_raw_input_even[0].item())
        )
        self.seen_input_history.append((previous, previous_previous))
        return next(self.decisions)


class AlwaysSkipMethod(CacheMethod):
    name = "always_skip"

    def decide_conditional(self, raw_input, timestep):
        return True


class FakeModel(torch.nn.Module):
    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        return [value + 1.0 for value in x]


class DistinctResidualModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_index = 0

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        residual = 10.0 if self.forward_index % 2 == 0 else 20.0
        self.forward_index += 1
        return [value + residual for value in x]


class ScalingModel(torch.nn.Module):
    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        return [value * 2.0 for value in x]


class ChangingResidualModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residuals = iter([10.0, 20.0, 12.0, 23.0])

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        residual = next(self.residuals)
        return [value + residual for value in x]


class ObserveHistoryMethod(CacheMethod):
    name = "observe_history"

    def _reset_method(self) -> None:
        self.observed_history = []

    def decide_conditional(self, raw_input, timestep):
        return False

    def observe_conditional(self, raw_input, output):
        previous_input = (
            None
            if self.previous_raw_input_even is None
            else float(self.previous_raw_input_even[0].item())
        )
        previous_output = (
            None
            if self.previous_raw_output_even is None
            else float(self.previous_raw_output_even[0].item())
        )
        self.observed_history.append((previous_input, previous_output))


def run_with_cache_method(
    model: torch.nn.Module,
    method: CacheMethod,
    pairs: int,
) -> list[tuple[float, float]]:
    original_forward = model.forward
    observed = []

    def cached_forward(_model, x, t, context, seq_len, clip_fea=None, y=None):
        raw_input = [value.clone() for value in x]
        cached = method.try_skip(raw_input, t)
        if cached is not None:
            return cached
        kwargs = {}
        if clip_fea is not None:
            kwargs["clip_fea"] = clip_fea
        if y is not None:
            kwargs["y"] = y
        output = original_forward(x, t, context, seq_len, **kwargs)
        method.update(raw_input, output)
        return output

    with patch_forward(model, cached_forward):
        for pair in range(pairs):
            for cfg_call in range(2):
                value = torch.tensor([float(pair * 2 + cfg_call)])
                output = model([value], torch.tensor([1.0]), [], 1)
                observed.append((value.item(), output[0].item()))
    return observed


class FakeMetricComputer:
    def __call__(self, reference, candidate):
        difference = (
            reference.astype(np.float32) - candidate.astype(np.float32)
        ) / 255
        squared = np.square(difference)
        return (
            float(squared.sum()),
            squared.size,
            float(len(reference)),
            0.0,
        )


class EvalContractTests(unittest.TestCase):
    def test_forward_timing_accumulates_cuda_event_durations(self) -> None:
        class FakeEvent:
            def __init__(self, milliseconds: float) -> None:
                self.milliseconds = milliseconds

            def record(self) -> None:
                return None

            def elapsed_time(self, ended) -> float:
                return ended.milliseconds - self.milliseconds

        events = iter(
            [FakeEvent(0.0), FakeEvent(12.0), FakeEvent(20.0), FakeEvent(23.0)]
        )
        with (
            patch("eval.pipeline.torch.cuda.is_available", return_value=True),
            patch("eval.pipeline.torch.cuda.Event", side_effect=lambda **_: next(events)),
        ):
            timing = ForwardTiming()
            started = timing.start()
            timing.stop("dit", started)
            started = timing.start()
            timing.stop("cache", started)
            summary = timing.summary()

        self.assertAlmostEqual(summary["dit_forward_seconds"], 0.012)
        self.assertEqual(summary["dit_forward_calls"], 1)
        self.assertAlmostEqual(summary["cache_forward_seconds"], 0.003)
        self.assertEqual(summary["cache_forward_calls"], 1)
        self.assertAlmostEqual(summary["dit_to_cache_speedup"], 4.0)

    def test_num_processes_uses_visible_cuda_devices(self) -> None:
        with patch("eval.pipeline.torch.cuda.device_count", return_value=8):
            self.assertEqual(_resolve_num_processes("auto"), 8)
            self.assertEqual(_resolve_num_processes(4), 4)
            with self.assertRaisesRegex(ValueError, "only 8 CUDA devices"):
                _resolve_num_processes(9)

    def test_python_workers_partition_each_task_once(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        tasks = build_tasks(cfg, PROJECT_ROOT)
        cfg_payload = OmegaConf.to_container(cfg, resolve=True)
        self.assertIsInstance(cfg_payload, dict)
        selected = []

        def fake_run_task(task, *_args):
            selected.append(int(task.run_id))
            return {"status": "complete"}

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with patch("eval.pipeline.run_task", side_effect=fake_run_task):
                for rank in range(4):
                    _run_worker(
                        rank,
                        4,
                        tasks,
                        cfg_payload,
                        PROJECT_ROOT,
                        output_root,
                    )

        self.assertEqual(sorted(selected), list(range(len(tasks))))
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(int(cfg.runtime.device_id), 0)

    def test_worker_loads_wan_once_for_multiple_tasks(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        tasks = build_tasks(cfg, PROJECT_ROOT)[:2]
        constructor_calls = []
        runtimes = []

        class FakeWan:
            @staticmethod
            def WanTI2V(**kwargs):
                constructor_calls.append(kwargs)
                return SimpleNamespace(model=FakeModel())

        wan_module = ModuleType("wan")
        wan_module.WanTI2V = FakeWan.WanTI2V
        configs_module = ModuleType("wan.configs")
        configs_module.WAN_CONFIGS = {
            "ti2v-5B": SimpleNamespace(sample_fps=16)
        }
        configs_module.SIZE_CONFIGS = {str(cfg.generation.size): (1, 1)}
        configs_module.MAX_AREA_CONFIGS = {str(cfg.generation.size): 1}
        utils_package = ModuleType("wan.utils")
        utils_module = ModuleType("wan.utils.utils")
        utils_module.save_video = lambda **kwargs: None

        def fake_run_task(task, task_cfg, runtime, *_args):
            del task_cfg
            runtimes.append(runtime)
            runtime.ensure_loaded()
            return {
                "status": "complete",
                "run_id": task.run_id,
            }

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            checkpoint = output_root / "checkpoint"
            checkpoint.mkdir()
            cfg.paths.checkpoint_dir = str(checkpoint)
            cfg_payload = OmegaConf.to_container(cfg, resolve=True)
            with (
                patch.dict(
                    sys.modules,
                    {
                        "wan": wan_module,
                        "wan.configs": configs_module,
                        "wan.utils": utils_package,
                        "wan.utils.utils": utils_module,
                    },
                ),
                patch("eval.pipeline.run_task", side_effect=fake_run_task),
                patch("eval.pipeline.torch.cuda.set_device"),
                patch("eval.pipeline.torch.cuda.synchronize"),
            ):
                _run_worker(
                    0,
                    1,
                    tasks,
                    cfg_payload,
                    PROJECT_ROOT,
                    output_root,
                )

        self.assertEqual(len(constructor_calls), 1)
        self.assertIsInstance(runtimes[0], WanWorkerRuntime)
        self.assertEqual(len({id(runtime) for runtime in runtimes}), 1)

    def test_manifest_skips_complete_tasks(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        task = build_tasks(cfg, PROJECT_ROOT)[0]

        def fake_generate(
            selected_task,
            selected_cfg,
            runtime,
            project_root,
            video_path,
            raw_log_path,
        ):
            del runtime, project_root, raw_log_path
            video_path.write_bytes(b"video")
            return {
                "status": "complete",
                "run_id": selected_task.run_id,
                "prompt_id": selected_task.prompt_id,
                "prompt": selected_task.prompt,
                "seed": selected_task.seed,
                "method": selected_task.method,
                "cache_threshold": selected_task.cache_threshold,
                "generation": OmegaConf.to_container(
                    selected_cfg.generation,
                    resolve=True,
                ),
                "timing": {
                    "generation_seconds": 1.0,
                    "dit_forward_seconds": 0.8,
                    "dit_forward_calls": 100,
                    "cache_forward_seconds": 0.0,
                    "cache_forward_calls": 0,
                },
                "cache": {
                    "skipped_pairs": 0,
                    "skip_ratio_all_pairs": 0.0,
                    "skipped_pair_indices": [],
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with patch("eval.pipeline._generate_video", side_effect=fake_generate):
                run_task(
                    task,
                    cfg,
                    Mock(),
                    PROJECT_ROOT,
                    output_root,
                )
            output_dir = output_root / task.output_group
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["0.mp4", "0.raw.log", "0.result.json"],
            )
            manifest_path = output_root / "tasks.jsonl"
            manifest_path.write_text(
                json.dumps(asdict(task), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _pending_tasks([task], output_root, manifest_path),
                [],
            )

            result_path = output_dir / "0.result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["timing"] = {"generation_seconds": 1.0}
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(
                _pending_tasks([task], output_root, manifest_path),
                [task],
            )

            (output_dir / "0.result.json").unlink()
            self.assertEqual(
                _pending_tasks([task], output_root, manifest_path),
                [task],
            )

    def test_default_grid_has_thirty_six_tasks(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        tasks = build_tasks(cfg, PROJECT_ROOT)
        self.assertEqual(len(tasks), 36)
        self.assertEqual(sum(task.method == "origin" for task in tasks), 3)
        self.assertEqual(sum(task.method == "easycache" for task in tasks), 9)
        self.assertEqual(sum(task.method == "model" for task in tasks), 9)
        self.assertEqual(sum(task.method == "temporal" for task in tasks), 9)
        self.assertEqual(sum(task.method == "magcache_output" for task in tasks), 3)
        self.assertEqual(sum(task.method == "d2cache_output" for task in tasks), 3)
        self.assertEqual([task.run_id for task in tasks], [str(i) for i in range(36)])
        self.assertEqual(
            [task.prompt_id for task in tasks],
            ["0"] * 12 + ["1"] * 12 + ["2"] * 12,
        )
        self.assertEqual({task.seed for task in tasks}, {123})
        model_tasks = [task for task in tasks if task.method == "model"]
        baseline_tasks = [task for task in tasks if task.method != "model"]
        self.assertTrue(
            all(
                task.warmup_steps == 10 and task.final_full_steps == 7
                for task in model_tasks
            )
        )
        self.assertTrue(
            all(
                task.warmup_steps == 7 and task.final_full_steps == 1
                for task in baseline_tasks
            )
        )
        self.assertTrue(bool(cfg.generation.offload_model))
        self.assertFalse(bool(cfg.generation.convert_model_dtype))
        self.assertFalse(bool(cfg.generation.t5_cpu))
        self.assertEqual(
            [task.cache_threshold for task in tasks if task.method == "model"],
            [0.03, 0.05, 0.07] * 3,
        )
        expected_groups = [
            "origin",
            "easycache_0.03",
            "easycache_0.05",
            "easycache_0.07",
            "model_0.03",
            "model_0.05",
            "model_0.07",
            "temporal_0.03",
            "temporal_0.05",
            "temporal_0.07",
            "magcache_output_0.06",
            "d2cache_output_0.05",
        ]
        self.assertEqual(
            [task.output_group for task in tasks],
            expected_groups * 3,
        )
        self.assertEqual(
            {
                group: [task.prompt_id for task in tasks if task.output_group == group]
                for group in expected_groups
            },
            {group: ["0", "1", "2"] for group in expected_groups},
        )
        self.assertEqual(
            len({(task.output_group, task.prompt_id) for task in tasks}),
            len(tasks),
        )
        self.assertEqual(
            {task.method_config["_target_"] for task in tasks},
            {
                "eval.model.origin.OriginMethod",
                "eval.model.easycache.EasyCacheMethod",
                "eval.model.cumulative.CumulativeMethod",
                "eval.model.temporal.TemporalMethod",
                "eval.model.magcache.MagCacheMethod",
                "eval.model.d2cache.D2CacheMethod",
            },
        )
        self.assertNotIn("origin_root", cfg.paths)
        self.assertNotIn("wan_root", cfg.paths)
        self.assertNotIn("checkpoint_id", cfg.paths)

    def test_grid_uses_one_shared_seed_for_every_prompt(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        cfg.prompts.seed = 42
        tasks = build_tasks(cfg, PROJECT_ROOT)
        self.assertEqual({task.seed for task in tasks}, {42})

    def test_output_adaptations_instantiate_from_the_sweep(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        tasks = build_tasks(cfg, PROJECT_ROOT)
        magcache_task = next(
            task for task in tasks if task.method == "magcache_output"
        )
        d2cache_task = next(
            task for task in tasks if task.method == "d2cache_output"
        )

        magcache = instantiate(
            magcache_task.method_config,
            cache_threshold=magcache_task.cache_threshold,
        )
        d2cache = instantiate(
            d2cache_task.method_config,
            cache_threshold=d2cache_task.cache_threshold,
        )
        self.assertIsInstance(magcache, MagCacheMethod)
        self.assertIsInstance(d2cache, D2CacheMethod)
        self.assertEqual(len(magcache.magnitude_ratios), 50)

    def test_default_quality_config_uses_the_same_experiment(self) -> None:
        cfg, pairs, output_dir = load_pairs(
            PROJECT_ROOT / "eval" / "conf" / "quality.yaml",
            preview=True,
        )
        experiment_root = (PROJECT_ROOT / "eval" / "outputs" / "experiment").resolve()
        self.assertEqual(len(pairs), 33)
        self.assertEqual(
            [pair.origin_video for pair in pairs],
            [
                experiment_root / "origin" / f"{prompt_id}.mp4"
                for prompt_id in range(3)
                for _ in range(11)
            ],
        )
        groups = (
            "easycache_0.03",
            "easycache_0.05",
            "easycache_0.07",
            "model_0.03",
            "model_0.05",
            "model_0.07",
            "temporal_0.03",
            "temporal_0.05",
            "temporal_0.07",
            "magcache_output_0.06",
            "d2cache_output_0.05",
        )
        self.assertEqual(
            [pair.target_video for pair in pairs],
            [
                experiment_root / group / f"{prompt_id}.mp4"
                for prompt_id in range(3)
                for group in groups
            ],
        )
        self.assertEqual(output_dir, experiment_root / "metrics" / "quality")
        self.assertEqual(int(cfg.fvd_num_frames), 16)
        self.assertEqual(int(cfg.fvd_frame_stride), 8)

    def test_quality_rejects_origin_target_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_direct_quality_config(
                root,
                ["origin0.mp4", "origin1.mp4"],
                [labeled_target("easycache", ["target0.mp4"], 0.03)],
            )
            with self.assertRaisesRegex(ValueError, "one video per origin"):
                load_pairs(config, preview=True)

    def test_quality_config_accepts_dotlist_overrides(self) -> None:
        config = PROJECT_ROOT / "eval" / "conf" / "quality.yaml"
        experiment_root = PROJECT_ROOT / "eval" / "outputs" / "override"
        cfg, pairs, output_dir = load_pairs(
            config,
            preview=True,
            overrides=[
                f"experiment_root={experiment_root.as_posix()}",
                "device=cuda:3",
                "fvd_frame_stride=4",
            ],
        )

        self.assertEqual(str(cfg.device), "cuda:3")
        self.assertEqual(int(cfg.fvd_frame_stride), 4)
        self.assertEqual(
            pairs[0].origin_video,
            experiment_root / "origin" / "0.mp4",
        )
        self.assertEqual(output_dir, experiment_root / "metrics" / "quality")

    def test_quality_config_rejects_non_assignment_override(self) -> None:
        config = PROJECT_ROOT / "eval" / "conf" / "quality.yaml"
        with self.assertRaisesRegex(ValueError, "key=value"):
            load_pairs(
                config,
                preview=True,
                overrides=["experiment_root"],
            )

    def test_metric_uses_explicit_alexnet_weights(self) -> None:
        calls = {}

        class FakeLPIPS(torch.nn.Module):
            def __init__(self, **kwargs) -> None:
                super().__init__()
                calls.update(kwargs)
                self.net = torch.nn.Identity()

        lpips_module = ModuleType("lpips")
        lpips_module.LPIPS = FakeLPIPS
        image_module = ModuleType("torchmetrics.functional.image")
        image_module.structural_similarity_index_measure = Mock()
        weights = Path("/models/alexnet-owt-7be5be79.pth")
        modules = {
            "lpips": lpips_module,
            "torchmetrics": ModuleType("torchmetrics"),
            "torchmetrics.functional": ModuleType("torchmetrics.functional"),
            "torchmetrics.functional.image": image_module,
        }
        with (
            patch.dict(sys.modules, modules),
            patch("eval.evaluate_quality._load_alexnet_weights") as load_weights,
        ):
            computer = MetricComputer("cpu", weights)

        self.assertEqual(
            calls,
            {"net": "alex", "spatial": True, "pnet_rand": True},
        )
        load_weights.assert_called_once_with(computer.lpips.net, weights)

    def test_alexnet_checkpoint_is_loaded_from_configured_path(self) -> None:
        state = {"weights": torch.tensor(1.0)}
        backbone = SimpleNamespace(
            features=list(range(12)),
            load_state_dict=Mock(),
        )
        net = SimpleNamespace(requires_grad_=Mock())
        weights = Path("/models/alexnet-owt-7be5be79.pth")
        with (
            patch("torchvision.models.alexnet", return_value=backbone) as build,
            patch(
                "eval.evaluate_quality.torch.load",
                return_value=state,
            ) as torch_load,
        ):
            _load_alexnet_weights(net, weights)

        build.assert_called_once_with(weights=None)
        torch_load.assert_called_once_with(
            weights,
            map_location="cpu",
            weights_only=True,
        )
        backbone.load_state_dict.assert_called_once_with(state)
        self.assertEqual(net.slice1, [0, 1])
        self.assertEqual(net.slice5, [10, 11])
        net.requires_grad_.assert_called_once_with(False)

    def test_quality_summary_uses_method_and_threshold_labels(self) -> None:
        rows = []
        for origin_index in range(2):
            for target_slot in range(2):
                rows.append(
                    {
                        "method": "easycache" if target_slot == 0 else "model",
                        "cache_threshold": 0.03 if target_slot == 0 else 0.05,
                        "psnr": float(20 + target_slot),
                        "ssim": float(0.8 + 0.1 * target_slot),
                        "lpips": float(0.2 - 0.1 * target_slot),
                    }
                )
        fvd_scores = {
            ("easycache", 0.03): 1.5,
            ("model", 0.05): 2.5,
        }
        summaries = summarize_quality(rows, fvd_scores)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(row["videos"] == 2 for row in summaries))
        labels = {
            (row["method"], row["cache_threshold"])
            for row in summaries
        }
        self.assertEqual(
            labels,
            {("easycache", 0.03), ("model", 0.05)},
        )
        self.assertTrue(
            all(set(row) == set(SUMMARY_FIELDS) for row in summaries)
        )
        self.assertEqual(
            {row["fvd"] for row in summaries},
            {1.5, 2.5},
        )

    def test_frechet_distance_uses_distribution_statistics(self) -> None:
        origin = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        self.assertAlmostEqual(frechet_distance(origin, origin), 0.0, places=8)
        shifted = origin + np.array([1.0, 2.0])
        self.assertAlmostEqual(
            frechet_distance(origin, shifted),
            5.0,
            places=7,
        )

    def test_fvd_groups_videos_by_method_and_threshold(self) -> None:
        pairs = []
        for method, threshold, offset in (
            ("easycache", 0.03, 0),
            ("model", 0.05, 2),
        ):
            for index in range(3):
                pairs.append(
                    VideoPair(
                        pair_id=f"{method}-{index}",
                        origin_video=Path("origin") / f"{index}.mp4",
                        target_video=Path(method) / f"{index + offset}.mp4",
                        origin_index=index,
                        method=method,
                        cache_threshold=threshold,
                    )
                )

        class FakeFVDComputer:
            def __call__(self, paths):
                return np.array([[float(path.stem)] for path in paths])

        scores = evaluate_fvd(pairs, FakeFVDComputer())
        self.assertAlmostEqual(scores[("easycache", 0.03)], 0.0)
        self.assertAlmostEqual(scores[("model", 0.05)], 4.0)

    def test_same_experiment_performance_summary_pairs_origin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment_root = Path(directory) / "experiment"
            write_performance_result(
                experiment_root,
                output_group="origin",
                run_id="0",
                prompt_id="0",
                method="origin",
                cache_threshold=None,
                generation_seconds=10.0,
                skipped_pair_indices=[],
            )
            write_performance_result(
                experiment_root,
                output_group="easycache_0.05",
                run_id="1",
                prompt_id="0",
                method="easycache",
                cache_threshold=0.05,
                generation_seconds=5.0,
                skipped_pair_indices=[7],
            )

            summarize_performance(experiment_root)
            metrics_dir = experiment_root / "metrics"
            per_run_path = metrics_dir / "performance_per_run.csv"
            summary_path = metrics_dir / "performance_summary.csv"
            self.assertTrue(per_run_path.is_file())
            self.assertTrue(summary_path.is_file())

            with per_run_path.open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            candidate = next(row for row in rows if row["method"] == "easycache")
            self.assertEqual(float(candidate["speedup"]), 2.0)
            self.assertEqual(candidate["skipped_pair_indices"], "7")
            self.assertEqual(int(candidate["dit_forward_calls"]), 98)
            self.assertEqual(int(candidate["cache_forward_calls"]), 2)
            self.assertAlmostEqual(
                float(candidate["dit_forward_mean_milliseconds"]),
                4000.0 / 98,
            )
            self.assertAlmostEqual(
                float(candidate["cache_forward_mean_milliseconds"]),
                50.0,
            )

            with summary_path.open(newline="", encoding="utf-8") as handle:
                summaries = list(csv.DictReader(handle))
            candidate_summary = next(
                row for row in summaries if row["method"] == "easycache"
            )
            self.assertEqual(candidate_summary["cache_threshold"], "0.05")
            self.assertEqual(float(candidate_summary["speedup_mean"]), 2.0)
            self.assertAlmostEqual(
                float(candidate_summary["dit_forward_mean_milliseconds"]),
                4000.0 / 98,
            )
            self.assertAlmostEqual(
                float(candidate_summary["cache_forward_mean_milliseconds"]),
                50.0,
            )
            self.assertAlmostEqual(
                float(candidate_summary["dit_forward_seconds_total"]),
                4.0,
            )
            self.assertAlmostEqual(
                float(candidate_summary["dit_forward_seconds_mean"]),
                4.0,
            )
            self.assertAlmostEqual(
                float(candidate_summary["dit_forward_calls_mean"]),
                98.0,
            )
            self.assertAlmostEqual(
                float(candidate_summary["cache_forward_seconds_mean"]),
                0.1,
            )
            self.assertAlmostEqual(
                float(candidate_summary["cache_forward_calls_mean"]),
                2.0,
            )

    def test_combined_evaluation_runs_performance_then_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "quality.yaml"
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "project_root": str(root),
                        "experiment_root": "experiment",
                    }
                ),
                config,
            )
            calls = []
            performance = [
                {
                    "method": "origin",
                    "cache_threshold": None,
                    "latency_mean_seconds": 10.0,
                    "skipped_pairs_mean": 0.0,
                    "speedup_mean": 1.0,
                    "dit_forward_seconds_total": 1950.0,
                    "dit_forward_seconds_mean": 650.0,
                    "dit_forward_calls_mean": 100.0,
                    "cache_forward_seconds_mean": 0.0,
                    "cache_forward_calls_mean": 0.0,
                    "dit_forward_mean_milliseconds": 80.0,
                    "cache_forward_mean_milliseconds": None,
                    "dit_to_cache_speedup": None,
                },
                {
                    "method": "easycache",
                    "cache_threshold": 0.03,
                    "latency_mean_seconds": 6.0,
                    "skipped_pairs_mean": 8.0,
                    "speedup_mean": 1.5,
                    "dit_forward_seconds_total": 1500.0,
                    "dit_forward_seconds_mean": 500.0,
                    "dit_forward_calls_mean": 84.0,
                    "cache_forward_seconds_mean": 0.01,
                    "cache_forward_calls_mean": 16.0,
                    "dit_forward_mean_milliseconds": 75.0,
                    "cache_forward_mean_milliseconds": 5.0,
                    "dit_to_cache_speedup": 15.0,
                },
                {
                    "method": "easycache",
                    "cache_threshold": 0.05,
                    "latency_mean_seconds": 5.0,
                    "skipped_pairs_mean": 10.0,
                    "speedup_mean": 2.0,
                    "dit_forward_seconds_total": 1350.0,
                    "dit_forward_seconds_mean": 450.0,
                    "dit_forward_calls_mean": 80.0,
                    "cache_forward_seconds_mean": 0.02,
                    "cache_forward_calls_mean": 20.0,
                    "dit_forward_mean_milliseconds": 70.0,
                    "cache_forward_mean_milliseconds": 4.0,
                    "dit_to_cache_speedup": 17.5,
                },
            ]
            quality = [
                {
                    "method": "easycache",
                    "cache_threshold": 0.05,
                    "psnr_mean": 30.0,
                    "ssim_mean": 0.9,
                    "lpips_mean": 0.1,
                    "fvd": 10.0,
                },
                {
                    "method": "easycache",
                    "cache_threshold": 0.03,
                    "psnr_mean": 25.0,
                    "ssim_mean": 0.8,
                    "lpips_mean": 0.2,
                    "fvd": 20.0,
                }
            ]

            def fake_performance(path):
                calls.append(("performance", path))
                return [], performance

            overrides = ["experiment_root=experiment"]

            def fake_quality(path, selected_overrides):
                calls.append(("quality", path, selected_overrides))
                return [], quality

            with (
                patch(
                    "eval.evaluate.summarize.run",
                    side_effect=fake_performance,
                ),
                patch(
                    "eval.evaluate.evaluate_quality.run",
                    side_effect=fake_quality,
                ),
            ):
                table_path = run_evaluation(config, overrides)

            self.assertEqual(
                calls,
                [
                    ("performance", (root / "experiment").resolve()),
                    ("quality", config.resolve(), overrides),
                ],
            )
            self.assertEqual(
                table_path,
                (root / "experiment" / "metrics" / "evaluation_summary.md").resolve(),
            )
            table = table_path.read_text(encoding="utf-8")
            self.assertIn('colspan="4">Efficiency', table)
            self.assertIn('colspan="4">Visual Quality Retention', table)
            self.assertEqual(table.count("<td>-</td>"), 4)
            row_003 = next(
                row for row in table.split("<tr>") if "EasyCache (0.03)" in row
            )
            row_005 = next(
                row for row in table.split("<tr>") if "EasyCache (0.05)" in row
            )
            self.assertIn("8.00", row_003)
            self.assertIn("500.0000", row_003)
            self.assertNotIn("1500.0000", row_003)
            self.assertIn("25.0000", row_003)
            self.assertIn("20.0000", row_003)
            self.assertIn("10.00", row_005)
            self.assertIn("450.0000", row_005)
            self.assertNotIn("1350.0000", row_005)
            self.assertIn("30.0000", row_005)
            self.assertIn("10.0000", row_005)

    def test_combined_evaluation_removes_stale_table_before_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "quality.yaml"
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "project_root": str(root),
                        "experiment_root": "experiment",
                    }
                ),
                config,
            )
            table_path = root / "experiment" / "metrics" / "evaluation_summary.md"
            table_path.parent.mkdir(parents=True)
            table_path.write_text("stale table\n", encoding="utf-8")

            with (
                patch("eval.evaluate.summarize.run", return_value=([], [])),
                patch(
                    "eval.evaluate.evaluate_quality.run",
                    side_effect=RuntimeError("quality failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "quality failed"),
            ):
                run_evaluation(config)

            self.assertFalse(table_path.exists())

    def test_cache_method_shares_one_decision_across_each_cfg_pair(self) -> None:
        method = ScriptedMethod()
        method.reset(sample_steps=3, warmup_steps=0, final_full_steps=0)
        model = FakeModel()
        observed = run_with_cache_method(model, method, pairs=3)
        summary = method.summary()
        self.assertEqual(observed, [(float(i), float(i + 1)) for i in range(6)])
        self.assertEqual(summary["calculated_pair_indices"], [0, 2])
        self.assertEqual(summary["skipped_pair_indices"], [1])

    def test_patch_forward_restores_the_model_after_an_exception(self) -> None:
        model = FakeModel()

        def failing_forward(model_self, x, t, context, seq_len, **kwargs):
            del model_self, x, t, context, seq_len, kwargs
            raise RuntimeError("generation failed")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with patch_forward(model, failing_forward):
                model([torch.tensor([1.0])], torch.tensor([1.0]), [], 1)

        restored = model([torch.tensor([1.0])], torch.tensor([1.0]), [], 1)
        self.assertEqual(restored[0].item(), 2.0)

    def test_origin_generation_times_each_wan_forward(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        task = build_tasks(cfg, PROJECT_ROOT)[0]
        model = FakeModel()

        def generate(*args, **kwargs):
            del args, kwargs
            for index in range(2):
                value = torch.tensor([float(index)])
                model([value], torch.tensor([1.0]), [], 1)
            return torch.zeros(1)

        runtime = SimpleNamespace(
            pipeline=SimpleNamespace(
                model=model,
                generate=generate,
            ),
            size_configs={str(cfg.generation.size): (1, 1)},
            max_area_configs={str(cfg.generation.size): 1},
            wan_config=SimpleNamespace(sample_fps=16),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "0.mp4"
            runtime.save_video = lambda **kwargs: Path(
                kwargs["save_file"]
            ).write_bytes(b"video")
            with patch("eval.pipeline.torch.cuda.synchronize"):
                result = _generate_video(
                    task,
                    cfg,
                    runtime,
                    PROJECT_ROOT,
                    video_path,
                    root / "0.raw.log",
                )

        self.assertEqual(result["cache"]["calculated_pairs"], 50)
        self.assertEqual(result["cache"]["skipped_pairs"], 0)
        self.assertEqual(result["timing"]["dit_forward_calls"], 2)
        self.assertEqual(result["timing"]["cache_forward_calls"], 0)
        self.assertGreater(result["timing"]["dit_forward_seconds"], 0)
        self.assertIsNone(
            result["timing"]["cache_forward_mean_milliseconds"]
        )

    def test_accelerated_generation_times_dit_and_cache_paths(self) -> None:
        cfg = OmegaConf.load(PROJECT_ROOT / "eval" / "conf" / "sweep.yaml")
        task = next(
            task
            for task in build_tasks(cfg, PROJECT_ROOT)
            if task.method == "easycache"
        )
        task = replace(task, warmup_steps=0, final_full_steps=0)
        model = FakeModel()

        def generate(*args, **kwargs):
            del args, kwargs
            for pair in range(3):
                for cfg_call in range(2):
                    value = torch.tensor([float(pair * 2 + cfg_call)])
                    model([value], torch.tensor([1.0]), [], 1)
            return torch.zeros(1)

        runtime = SimpleNamespace(
            pipeline=SimpleNamespace(model=model, generate=generate),
            size_configs={str(cfg.generation.size): (1, 1)},
            max_area_configs={str(cfg.generation.size): 1},
            wan_config=SimpleNamespace(sample_fps=16),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime.save_video = lambda **kwargs: Path(
                kwargs["save_file"]
            ).write_bytes(b"video")
            with (
                patch("eval.pipeline.instantiate", return_value=ScriptedMethod()),
                patch("eval.pipeline.torch.cuda.synchronize"),
            ):
                result = _generate_video(
                    task,
                    cfg,
                    runtime,
                    PROJECT_ROOT,
                    root / "0.mp4",
                    root / "0.raw.log",
                )

        self.assertEqual(result["cache"]["calculated_pairs"], 2)
        self.assertEqual(result["cache"]["skipped_pairs"], 1)
        self.assertEqual(result["timing"]["dit_forward_calls"], 4)
        self.assertEqual(result["timing"]["cache_forward_calls"], 2)
        self.assertGreater(result["timing"]["dit_forward_seconds"], 0)
        self.assertGreater(result["timing"]["cache_forward_seconds"], 0)
        self.assertGreater(result["timing"]["dit_to_cache_speedup"], 0)

    def test_cache_method_advances_conditional_input_history_during_skips(self) -> None:
        method = HistoryMethod()
        method.reset(sample_steps=4, warmup_steps=0, final_full_steps=0)
        model = FakeModel()
        run_with_cache_method(model, method, pairs=4)
        self.assertEqual(
            method.seen_input_history,
            [(None, None), (0.0, None), (2.0, 0.0), (4.0, 2.0)],
        )

    def test_cache_method_forces_warmup_and_keeps_cfg_residuals_separate(self) -> None:
        method = AlwaysSkipMethod()
        method.reset(sample_steps=3, warmup_steps=2, final_full_steps=0)
        observed = run_with_cache_method(
            DistinctResidualModel(),
            method,
            pairs=3,
        )

        self.assertEqual(
            observed,
            [
                (0.0, 10.0),
                (1.0, 21.0),
                (2.0, 12.0),
                (3.0, 23.0),
                (4.0, 14.0),
                (5.0, 25.0),
            ],
        )
        self.assertEqual(method.summary()["calculated_pair_indices"], [0, 1])
        self.assertEqual(method.summary()["skipped_pair_indices"], [2])

    def test_method_observes_output_before_shared_history_moves(self) -> None:
        method = ObserveHistoryMethod()
        method.reset(sample_steps=2, warmup_steps=0, final_full_steps=0)
        run_with_cache_method(FakeModel(), method, pairs=2)
        self.assertEqual(method.observed_history, [(None, None), (0.0, 1.0)])

    def test_easycache_rate_uses_two_full_pairs(self) -> None:
        method = EasyCacheMethod(cache_threshold=0.05)
        method.reset(sample_steps=2, warmup_steps=0, final_full_steps=0)
        run_with_cache_method(ScalingModel(), method, pairs=2)
        self.assertEqual(method.transformation_rate, 2.0)

    def test_magcache_uses_ratio_error_and_max_cached_pairs(self) -> None:
        method = MagCacheMethod(
            cache_threshold=0.05,
            magnitude_ratios=[1.0, 0.99, 0.99, 0.99],
            max_cached_pairs=2,
            retention_ratio=0.0,
        )
        method.reset(sample_steps=4, warmup_steps=1, final_full_steps=0)
        run_with_cache_method(FakeModel(), method, pairs=4)

        summary = method.summary()
        self.assertEqual(summary["calculated_pair_indices"], [0, 3])
        self.assertEqual(summary["skipped_pair_indices"], [1, 2])
        self.assertEqual(summary["max_cached_pairs"], 2)

    def test_d2cache_corrects_conditional_and_unconditional_residuals(self) -> None:
        method = D2CacheMethod(cache_threshold=100.0)
        method.reset(sample_steps=3, warmup_steps=0, final_full_steps=0)
        observed = run_with_cache_method(
            ChangingResidualModel(),
            method,
            pairs=3,
        )

        self.assertEqual(method.summary()["calculated_pair_indices"], [0, 1])
        self.assertEqual(method.summary()["skipped_pair_indices"], [2])
        self.assertEqual(observed[-2:], [(4.0, 18.0), (5.0, 31.0)])
        self.assertEqual(method.residual_delta_even[0].item(), 2.0)
        self.assertEqual(method.residual_delta_odd[0].item(), 3.0)

    def test_cumulative_method_loads_generic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = CacheModel(
                input_dim=8,
                horizon=2,
                hidden_dim=8,
                num_hidden_layers=1,
            )
            artifact_payload = build_artifact(
                model,
                {
                    "_target_": "eval.model.CacheModel",
                    "input_dim": 8,
                    "horizon": 2,
                    "hidden_dim": 8,
                    "num_hidden_layers": 1,
                },
                feature_mean=torch.zeros(8),
                feature_std=torch.ones(8),
                calibration_offsets=torch.zeros(2),
                cache_threshold=0.05,
            )
            artifact = Path(directory) / "model.pth"
            torch.save(artifact_payload, artifact)

            method = CumulativeMethod(
                artifact_path=str(artifact),
                cache_threshold=0.05,
                device="cpu",
            )
            self.assertEqual(method.name, "model")
            self.assertEqual(method.cache_threshold, 0.05)
            method.reset(sample_steps=4, warmup_steps=0, final_full_steps=0)
            run_with_cache_method(FakeModel(), method, pairs=4)
            self.assertEqual(method.summary()["calculated_pair_indices"], [0, 3])
            self.assertEqual(method.summary()["skipped_pair_indices"], [1, 2])

    def test_temporal_method_uses_full_refresh_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = TemporalModel(
                input_dim=8,
                history_length=2,
                hidden_dim=8,
                head_dim=8,
                horizon=2,
            )
            artifact_payload = build_artifact(
                model,
                {
                    "_target_": "eval.model.temporal.TemporalModel",
                    "input_dim": 8,
                    "history_length": 2,
                    "hidden_dim": 8,
                    "head_dim": 8,
                    "horizon": 2,
                },
                feature_mean=torch.zeros(8),
                feature_std=torch.ones(8),
                calibration_offsets=torch.zeros(2),
                cache_threshold=0.05,
            )
            artifact = Path(directory) / "temporal.pth"
            torch.save(artifact_payload, artifact)

            method = TemporalMethod(
                artifact_path=str(artifact),
                cache_threshold=0.05,
                device="cpu",
            )
            method.model.forward = Mock(
                return_value=torch.tensor([[0.01, 0.02]])
            )
            method.reset(sample_steps=6, warmup_steps=3, final_full_steps=0)
            run_with_cache_method(FakeModel(), method, pairs=6)

            self.assertEqual(method.summary()["calculated_pair_indices"], [0, 1, 2, 5])
            self.assertEqual(method.summary()["skipped_pair_indices"], [3, 4])
            self.assertEqual(len(method.feature_history), 1)
            method.model.forward.assert_called_once()
            self.assertEqual(
                tuple(method.model.forward.call_args.args[0].shape),
                (1, 3, 8),
            )

    def test_strict_quality_decode_on_identical_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = root / "origin.mp4"
            candidate = root / "candidate.mp4"
            for path in (origin, candidate):
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    8.0,
                    (16, 16),
                )
                self.assertTrue(writer.isOpened())
                for value in (0, 64, 128, 255):
                    writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
                writer.release()
            pair = VideoPair(
                pair_id="unit",
                origin_video=origin,
                target_video=candidate,
                origin_index=0,
                method="easycache",
                cache_threshold=0.03,
            )
            row = evaluate_pair(
                pair,
                FakeMetricComputer(),
                batch_size=2,
            )
            self.assertEqual(row["frame_count"], 4)
            self.assertTrue(math.isinf(row["psnr"]))
            self.assertEqual(row["ssim"], 1.0)
            self.assertEqual(row["lpips"], 0.0)
            self.assertEqual(set(row), set(VIDEO_FIELDS))
            self.assertEqual(row["method"], "easycache")
            self.assertEqual(row["cache_threshold"], 0.03)

    def test_fvd_clip_uses_configured_temporal_stride(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                8.0,
                (16, 16),
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 32, 64, 96, 128):
                writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
            writer.release()

            clip = _read_fvd_clip(path, num_frames=3, frame_stride=2)
            self.assertEqual(clip.shape, (3, 16, 16, 3))
            self.assertLess(float(clip[0].mean()), float(clip[1].mean()))
            self.assertLess(float(clip[1].mean()), float(clip[2].mean()))

    def test_fvd_computer_uses_local_i3d_torchscript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "i3d_torchscript.pt"
            weights.touch()
            video_path = root / "clip.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                8.0,
                (16, 16),
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 32, 64):
                writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
            writer.release()

            class FakeI3D:
                def eval(self):
                    return self

                def to(self, device):
                    self.device = device
                    return self

                def __call__(self, video, **kwargs):
                    self.video = video
                    self.kwargs = kwargs
                    return torch.tensor([[1.0, 2.0]])

            model = FakeI3D()
            with patch("eval.evaluate_quality.torch.jit.load", return_value=model) as load:
                features = FVDComputer("cpu", weights, 2, 2)([video_path])

            load.assert_called_once_with(str(weights), map_location=torch.device("cpu"))
            self.assertEqual(tuple(model.video.shape), (1, 3, 2, 16, 16))
            self.assertEqual(model.video.dtype, torch.uint8)
            self.assertTrue(model.video.is_contiguous())
            self.assertEqual(
                model.kwargs,
                {"rescale": True, "resize": True, "return_features": True},
            )
            np.testing.assert_array_equal(features, [[1.0, 2.0]])

    def test_strict_quality_rejects_resolution_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = root / "origin.mp4"
            target = root / "target.mp4"
            for path, size in ((origin, (16, 16)), (target, (20, 16))):
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    8.0,
                    size,
                )
                self.assertTrue(writer.isOpened())
                writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
                writer.release()
            pair = VideoPair(
                pair_id="unit",
                origin_video=origin,
                target_video=target,
                origin_index=0,
                method="target",
                cache_threshold=None,
            )
            with self.assertRaisesRegex(ValueError, "resolution mismatch"):
                evaluate_pair(
                    pair,
                    FakeMetricComputer(),
                    batch_size=1,
                )


if __name__ == "__main__":
    unittest.main()
