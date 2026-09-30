"""Generate Wan videos for every prompt and cache method in the sweep."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from eval.cache_hook import patch_forward


LOGGER = logging.getLogger(__name__)

FORWARD_TIMING_FIELDS = {
    "dit_forward_seconds",
    "dit_forward_calls",
    "cache_forward_seconds",
    "cache_forward_calls",
}


class ForwardTiming:
    """Accumulate per-call CUDA time without synchronizing every forward."""

    def __init__(self) -> None:
        self.use_cuda_events = torch.cuda.is_available()
        self.cuda_events: dict[str, list[tuple[Any, Any]]] = {
            "dit": [],
            "cache": [],
        }
        self.cpu_seconds = {"dit": 0.0, "cache": 0.0}
        self.calls = {"dit": 0, "cache": 0}

    def start(self) -> Any:
        if self.use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, category: str, started: Any) -> None:
        if category not in self.cuda_events:
            raise ValueError(f"unknown forward timing category {category}")
        self.calls[category] += 1
        if self.use_cuda_events:
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self.cuda_events[category].append((started, ended))
        else:
            self.cpu_seconds[category] += time.perf_counter() - started

    def _seconds(self, category: str) -> float:
        if self.use_cuda_events:
            return sum(
                started.elapsed_time(ended) / 1000.0
                for started, ended in self.cuda_events[category]
            )
        return self.cpu_seconds[category]

    def summary(self) -> dict[str, Any]:
        dit_seconds = self._seconds("dit")
        cache_seconds = self._seconds("cache")
        dit_calls = self.calls["dit"]
        cache_calls = self.calls["cache"]
        dit_mean_ms = 1000.0 * dit_seconds / dit_calls if dit_calls else None
        cache_mean_ms = (
            1000.0 * cache_seconds / cache_calls if cache_calls else None
        )
        speedup = (
            dit_mean_ms / cache_mean_ms
            if dit_mean_ms is not None
            and cache_mean_ms is not None
            and cache_mean_ms > 0
            else None
        )
        return {
            "dit_forward_seconds": dit_seconds,
            "dit_forward_calls": dit_calls,
            "dit_forward_mean_milliseconds": dit_mean_ms,
            "cache_forward_seconds": cache_seconds,
            "cache_forward_calls": cache_calls,
            "cache_forward_mean_milliseconds": cache_mean_ms,
            "dit_to_cache_speedup": speedup,
        }


@dataclass(frozen=True)
class EvaluationTask:
    run_id: str
    output_group: str
    prompt_id: str
    prompt: str
    seed: int
    method: str
    cache_threshold: float | None
    method_config: dict[str, Any]
    warmup_steps: int
    final_full_steps: int


@dataclass
class WanWorkerRuntime:
    """One Wan pipeline shared by all videos assigned to one GPU worker."""

    cfg: DictConfig
    project_root: Path
    device_id: int
    pipeline: Any = None
    wan_config: Any = None
    size_configs: Any = None
    max_area_configs: Any = None
    save_video: Any = None

    def ensure_loaded(self) -> None:
        if self.pipeline is not None:
            return

        generation = self.cfg.generation
        checkpoint_dir = _resolve(
            str(self.cfg.paths.checkpoint_dir),
            self.project_root,
        )

        print(f"[GPU {self.device_id}] Import Wan", flush=True)
        torch.cuda.set_device(self.device_id)
        import wan
        from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
        from wan.utils.utils import save_video

        task_name = str(generation.task)
        started = time.perf_counter()
        print(
            f"[GPU {self.device_id}] Load Wan checkpoint from {checkpoint_dir}",
            flush=True,
        )
        self.wan_config = WAN_CONFIGS[task_name]
        self.pipeline = wan.WanTI2V(
            config=self.wan_config,
            checkpoint_dir=str(checkpoint_dir),
            device_id=self.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=bool(generation.t5_cpu),
            convert_model_dtype=bool(generation.convert_model_dtype),
        )
        torch.cuda.synchronize(self.device_id)
        self.size_configs = SIZE_CONFIGS
        self.max_area_configs = MAX_AREA_CONFIGS
        self.save_video = save_video
        print(
            f"[GPU {self.device_id}] Wan ready in "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )

def _resolve(path_value: str, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def build_tasks(cfg: DictConfig, project_root: Path) -> list[EvaluationTask]:
    prompt_path = _resolve(str(cfg.prompts.file), project_root)
    prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seed = int(cfg.prompts.seed)

    sample_steps = int(cfg.generation.sample_steps)
    warmup_steps = int(cfg.generation.warmup_steps)
    final_full_steps = int(cfg.generation.final_full_steps)
    if sample_steps <= 0 or not 0 <= warmup_steps + final_full_steps <= sample_steps:
        raise ValueError("invalid sampling or protected-step configuration")

    tasks: list[EvaluationTask] = []
    for prompt_id, prompt in enumerate(prompts):
        for method_cfg in cfg.methods:
            method_config = dict(
                OmegaConf.to_container(method_cfg.method, resolve=True)
            )
            method_warmup_steps = int(
                method_cfg.get("warmup_steps", warmup_steps)
            )
            method_final_full_steps = int(
                method_cfg.get("final_full_steps", final_full_steps)
            )
            if (
                method_warmup_steps < 0
                or method_final_full_steps < 0
                or method_warmup_steps + method_final_full_steps > sample_steps
            ):
                raise ValueError(
                    f"invalid protected-step configuration for {method_cfg.name}"
                )
            if "artifact_path" in method_config:
                method_config["artifact_path"] = str(
                    _resolve(str(method_config["artifact_path"]), project_root)
                )

            for threshold in method_cfg.get("cache_thresholds", [None]):
                cache_threshold = None if threshold is None else float(threshold)
                method = str(method_cfg.name)
                output_group = (
                    method
                    if cache_threshold is None
                    else f"{method}_{cache_threshold}"
                )
                tasks.append(
                    EvaluationTask(
                        run_id=str(len(tasks)),
                        output_group=output_group,
                        prompt_id=str(prompt_id),
                        prompt=prompt,
                        seed=seed,
                        method=method,
                        cache_threshold=cache_threshold,
                        method_config=method_config,
                        warmup_steps=method_warmup_steps,
                        final_full_steps=method_final_full_steps,
                    )
                )

    paths = [(task.output_group, task.prompt_id) for task in tasks]
    if len(paths) != len(set(paths)):
        raise ValueError("the sweep contains duplicate output paths")
    return tasks


def _pending_tasks(
    tasks: list[EvaluationTask],
    output_root: Path,
    manifest_path: Path,
) -> list[EvaluationTask]:
    if not manifest_path.is_file():
        return tasks

    previous_tasks = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            previous_tasks[(record["output_group"], record["prompt_id"])] = record
    pending = []
    for task in tasks:
        if previous_tasks.get((task.output_group, task.prompt_id)) != asdict(task):
            pending.append(task)
            continue

        result_path = (
            output_root
            / task.output_group
            / f"{task.prompt_id}.result.json"
        )
        if not result_path.is_file():
            pending.append(task)
            continue
        try:
            timing = json.loads(result_path.read_text(encoding="utf-8"))["timing"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pending.append(task)
            continue
        if not FORWARD_TIMING_FIELDS.issubset(timing):
            pending.append(task)
    return pending


def _generate_video(
    task: EvaluationTask,
    cfg: DictConfig,
    runtime: WanWorkerRuntime,
    project_root: Path,
    video_path: Path,
    raw_log_path: Path,
) -> dict[str, Any]:
    """Run one method and save its video and cache summary."""

    generation = cfg.generation
    device_id = int(cfg.runtime.device_id)
    method_kwargs: dict[str, Any] = {}
    if task.cache_threshold is not None:
        method_kwargs["cache_threshold"] = task.cache_threshold
    method = instantiate(task.method_config, **method_kwargs)
    method.reset(
        sample_steps=int(generation.sample_steps),
        warmup_steps=task.warmup_steps,
        final_full_steps=task.final_full_steps,
    )
    if hasattr(method, "validate_generation"):
        method.validate_generation(generation)
    if getattr(method, "requires_sigma_schedule", False):
        from eval.predictor_schedule import make_wan_scheduler

        scheduler = make_wan_scheduler(
            generation, f"cuda:{device_id}", runtime.pipeline.num_train_timesteps,
        )
        method.set_schedule(scheduler.sigmas, scheduler.timesteps)
        del scheduler
    image = None
    if generation.image is not None:
        image = Image.open(
            _resolve(str(generation.image), project_root)
        ).convert("RGB")

    original_forward = runtime.pipeline.model.forward
    forward_timing = ForwardTiming()

    def call_original(x, t, context, seq_len, optional_kwargs):
        started = forward_timing.start()
        output = original_forward(
            x,
            t,
            context,
            seq_len,
            **optional_kwargs,
        )
        forward_timing.stop("dit", started)
        return output

    def cached_forward(_model, x, t, context, seq_len, clip_fea=None, y=None):
        cache_started = forward_timing.start()
        raw_input = [value.clone() for value in x]
        cached_output = method.try_skip(raw_input, t)
        if cached_output is not None:
            forward_timing.stop("cache", cache_started)
            return cached_output

        optional_kwargs = {}
        if clip_fea is not None:
            optional_kwargs["clip_fea"] = clip_fea
        if y is not None:
            optional_kwargs["y"] = y
        output = call_original(x, t, context, seq_len, optional_kwargs)
        method.update(raw_input, output)
        return [value.float() for value in output]

    def timed_forward(_model, x, t, context, seq_len, clip_fea=None, y=None):
        optional_kwargs = {}
        if clip_fea is not None:
            optional_kwargs["clip_fea"] = clip_fea
        if y is not None:
            optional_kwargs["y"] = y
        return call_original(x, t, context, seq_len, optional_kwargs)

    hook = patch_forward(
        runtime.pipeline.model,
        cached_forward if method.accelerated else timed_forward,
    )
    with hook:
        torch.cuda.synchronize(device_id)
        started = time.perf_counter()
        video = runtime.pipeline.generate(
            task.prompt,
            img=image,
            size=runtime.size_configs[str(generation.size)],
            max_area=runtime.max_area_configs[str(generation.size)],
            frame_num=int(generation.frame_num),
            shift=float(generation.sample_shift),
            sample_solver=str(generation.sample_solver),
            sampling_steps=int(generation.sample_steps),
            guide_scale=float(generation.sample_guide_scale),
            seed=task.seed,
            offload_model=bool(generation.offload_model),
        )
        torch.cuda.synchronize(device_id)
        generation_seconds = time.perf_counter() - started

    cache_summary = method.summary()
    timing_summary = forward_timing.summary()

    runtime.save_video(
        tensor=video[None],
        save_file=str(video_path),
        fps=runtime.wan_config.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    generation_config = OmegaConf.to_container(cfg.generation, resolve=True)
    generation_config["warmup_steps"] = task.warmup_steps
    generation_config["final_full_steps"] = task.final_full_steps

    result = {
        "status": "complete",
        "run_id": task.run_id,
        "prompt_id": task.prompt_id,
        "prompt": task.prompt,
        "seed": task.seed,
        "method": task.method,
        "cache_threshold": task.cache_threshold,
        "generation": generation_config,
        "timing": {
            "generation_seconds": generation_seconds,
            **timing_summary,
        },
        "cache": cache_summary,
        "video_path": str(video_path.resolve()),
        "raw_log_path": str(raw_log_path.resolve()),
    }
    return result


def run_task(
    task: EvaluationTask,
    cfg: DictConfig,
    runtime: WanWorkerRuntime,
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_dir = output_root / task.output_group
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{task.prompt_id}.mp4"
    result_path = output_dir / f"{task.prompt_id}.result.json"
    raw_log_path = output_dir / f"{task.prompt_id}.raw.log"

    result_path.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)
    print(
        f"[GPU {runtime.device_id}] Start {task.output_group} "
        f"prompt {task.prompt_id}",
        flush=True,
    )
    runtime.ensure_loaded()
    with raw_log_path.open("w", encoding="utf-8", buffering=1) as raw_log:
        with contextlib.redirect_stdout(raw_log):
            with contextlib.redirect_stderr(raw_log):
                result = _generate_video(
                    task,
                    cfg,
                    runtime,
                    project_root,
                    video_path,
                    raw_log_path,
                )

    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[GPU {runtime.device_id}] Done {task.output_group} "
        f"prompt {task.prompt_id} latency "
        f"{result['timing']['generation_seconds']:.2f}s DiT "
        f"{result['timing']['dit_forward_seconds']:.2f}s cache "
        f"{result['timing']['cache_forward_seconds']:.2f}s skip "
        f"{result['cache']['skipped_pairs']}",
        flush=True,
    )
    return result


def _resolve_num_processes(value: Any) -> int:
    visible_devices = torch.cuda.device_count()
    if isinstance(value, str) and value.lower() == "auto":
        num_processes = visible_devices
    else:
        num_processes = int(value)
    if num_processes <= 0 or num_processes > visible_devices:
        raise ValueError(
            f"parallel.num_processes requests {num_processes} workers but only "
            f"{visible_devices} CUDA devices are visible"
        )
    return num_processes


def _run_worker(
    local_rank: int,
    world_size: int,
    tasks: list[EvaluationTask],
    cfg_payload: dict[str, Any],
    project_root: Path,
    output_root: Path,
) -> None:
    cfg = OmegaConf.create(cfg_payload)
    cfg.runtime.device_id = local_rank
    runtime = WanWorkerRuntime(cfg, project_root, local_rank)
    for task in tasks[local_rank::world_size]:
        run_task(task, cfg, runtime, project_root, output_root)


@hydra.main(version_base=None, config_path="conf", config_name="sweep")
def main(cfg: DictConfig) -> None:
    project_root = _resolve(str(cfg.paths.project_root), Path(get_original_cwd()))
    output_root = _resolve(str(cfg.paths.output_root), project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(cfg, project_root)
    manifest_path = output_root / "tasks.jsonl"
    pending_tasks = _pending_tasks(tasks, output_root, manifest_path)

    if bool(cfg.runtime.dry_run):
        LOGGER.info(
            "Prepared %d tasks, %d already complete, %d pending in %s",
            len(tasks),
            len(tasks) - len(pending_tasks),
            len(pending_tasks),
            manifest_path,
        )
        return

    manifest_path.write_text(
        "".join(
            json.dumps(asdict(task), ensure_ascii=False) + "\n" for task in tasks
        ),
        encoding="utf-8",
    )

    completed_count = len(tasks) - len(pending_tasks)
    if completed_count:
        LOGGER.info(
            "Skip %d completed tasks listed in %s",
            completed_count,
            manifest_path,
        )

    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    task_index = int(cfg.runtime.task_index)
    if task_index >= 0:
        if task_index >= len(tasks):
            raise IndexError(f"task_index {task_index} exceeds {len(tasks)} tasks")
        selected_task = tasks[task_index]
        if selected_task not in pending_tasks:
            LOGGER.info(
                "Skip completed task %s prompt %s",
                selected_task.output_group,
                selected_task.prompt_id,
            )
            return
        runtime = WanWorkerRuntime(cfg, project_root, int(cfg.runtime.device_id))
        run_task(selected_task, cfg, runtime, project_root, output_root)
        return

    if not pending_tasks:
        LOGGER.info("All %d tasks are already complete", len(tasks))
        return

    num_processes = min(
        _resolve_num_processes(cfg.parallel.num_processes),
        len(pending_tasks),
    )
    LOGGER.info(
        "Generate %d pending videos with %d GPU workers",
        len(pending_tasks),
        num_processes,
    )
    worker_args = (
        num_processes,
        pending_tasks,
        cfg_payload,
        project_root,
        output_root,
    )
    if num_processes == 1:
        _run_worker(0, *worker_args)
    else:
        torch.multiprocessing.spawn(
            _run_worker,
            args=worker_args,
            nprocs=num_processes,
            join=True,
        )
    LOGGER.info("Generation finished")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
