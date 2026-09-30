"""Collect raw Wan inputs and outputs for both CFG branches."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from eval.cache_hook import patch_forward
from eval.pipeline import WanWorkerRuntime, _resolve, _resolve_num_processes
from eval.predictor_schedule import make_wan_scheduler


class RawRecorder:
    def __init__(self, output_dir: Path, shard_steps: int) -> None:
        self.output_dir = output_dir
        self.shard_steps = shard_steps
        self.call_index = 0
        self.records: list[dict[str, Any]] = []
        self.shards: list[str] = []

    @staticmethod
    def copy(values) -> tuple[torch.Tensor, ...]:
        return tuple(value.detach().cpu().contiguous() for value in values)

    def record(self, model_input, timestep, model_output) -> None:
        step_index = self.call_index // 2
        branch = "conditional" if self.call_index % 2 == 0 else "unconditional"
        self.records.append(
            {
                "step_index": step_index,
                "branch": branch,
                "timestep": float(timestep.detach().max().cpu()),
                "model_input": model_input,
                "model_output": self.copy(model_output),
            }
        )
        self.call_index += 1
        if branch == "unconditional" and (step_index + 1) % self.shard_steps == 0:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        name = f"shard_{len(self.shards):04d}.pt"
        torch.save(self.records, self.output_dir / name)
        self.shards.append(name)
        self.records = []

    def finish(self, sample_steps: int) -> None:
        if self.call_index != sample_steps * 2:
            raise RuntimeError(
                f"expected {sample_steps * 2} CFG calls, got {self.call_index}"
            )
        self.flush()


def collect_trajectory(
    record: dict[str, Any],
    cfg: DictConfig,
    runtime: WanWorkerRuntime,
    project_root: Path,
    output_root: Path,
) -> None:
    output_dir = output_root / f"trajectory_{record['trajectory_id']:08d}"
    if (output_dir / "_SUCCESS").is_file():
        return
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    runtime.ensure_loaded()
    scheduler = make_wan_scheduler(
        cfg.generation, runtime.pipeline.device, runtime.pipeline.num_train_timesteps,
    )
    recorder = RawRecorder(output_dir, int(cfg.storage.shard_steps))
    original_forward = runtime.pipeline.model.forward

    def collecting_forward(_model, x, t, context, seq_len, clip_fea=None, y=None):
        model_input = recorder.copy(x)
        kwargs = {}
        if clip_fea is not None:
            kwargs["clip_fea"] = clip_fea
        if y is not None:
            kwargs["y"] = y
        model_output = original_forward(x, t, context, seq_len, **kwargs)
        recorder.record(model_input, t, model_output)
        return model_output

    image = None
    if cfg.generation.image is not None:
        image = Image.open(
            _resolve(str(cfg.generation.image), project_root)
        ).convert("RGB")

    with patch_forward(runtime.pipeline.model, collecting_forward):
        video = runtime.pipeline.generate(
            record["prompt"],
            img=image,
            size=runtime.size_configs[str(cfg.generation.size)],
            max_area=runtime.max_area_configs[str(cfg.generation.size)],
            frame_num=int(cfg.generation.frame_num),
            shift=float(cfg.generation.sample_shift),
            sample_solver=str(cfg.generation.sample_solver),
            sampling_steps=int(cfg.generation.sample_steps),
            guide_scale=float(cfg.generation.sample_guide_scale),
            seed=int(record["seed"]),
            offload_model=bool(cfg.generation.offload_model),
        )
    del video

    recorder.finish(int(cfg.generation.sample_steps))
    metadata = {
        **record, "shards": recorder.shards,
        "scheduler": {
            "sigmas": scheduler.sigmas.cpu().tolist(),
            "timesteps": scheduler.timesteps.cpu().tolist(),
            "num_train_timesteps": runtime.pipeline.num_train_timesteps,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    print(f"GPU {runtime.device_id} saved {output_dir.name}", flush=True)


def run_worker(
    local_rank: int,
    world_size: int,
    records: list[dict[str, Any]],
    cfg_payload: dict[str, Any],
    project_root: Path,
    output_root: Path,
) -> None:
    cfg = OmegaConf.create(cfg_payload)
    cfg.runtime.device_id = local_rank
    runtime = WanWorkerRuntime(cfg, project_root, local_rank)
    for record in records[local_rank::world_size]:
        collect_trajectory(record, cfg, runtime, project_root, output_root)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="collect_raw_trajectories",
)
def main(cfg: DictConfig) -> None:
    project_root = _resolve(str(cfg.paths.project_root), Path(get_original_cwd()))
    output_root = _resolve(str(cfg.paths.output_root), project_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_path = _resolve(str(cfg.data.prompt_file), project_root)
    prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [
        {
            "trajectory_id": prompt_id * int(cfg.data.seeds_per_prompt) + seed_index,
            "prompt_id": prompt_id,
            "seed_index": seed_index,
            "prompt": prompt,
            "seed": int(cfg.data.base_seed)
            + prompt_id * int(cfg.data.seeds_per_prompt)
            + seed_index,
        }
        for prompt_id, prompt in enumerate(prompts)
        for seed_index in range(int(cfg.data.seeds_per_prompt))
    ]
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "fields": ["model_input", "model_output"],
                "branches": ["conditional", "unconditional"],
                "generation": OmegaConf.to_container(cfg.generation, resolve=True),
            }
        ),
        output_root / "dataset.yaml",
    )

    num_processes = min(
        _resolve_num_processes(cfg.parallel.num_processes),
        len(records),
    )
    worker_args = (
        num_processes,
        records,
        OmegaConf.to_container(cfg, resolve=True),
        project_root,
        output_root,
    )
    if num_processes == 1:
        run_worker(0, *worker_args)
    else:
        torch.multiprocessing.spawn(
            run_worker,
            args=worker_args,
            nprocs=num_processes,
            join=True,
        )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
