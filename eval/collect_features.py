"""Collect raw Wan transformer features for learned cache methods."""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from eval.cache_hook import patch_forward
from eval.pipeline import WanWorkerRuntime, _resolve, _resolve_num_processes


class FeatureRecorder:
    """Write one raw conditional/unconditional feature pair per diffusion step."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.call_index = 0
        self.feature: torch.Tensor | None = None
        self.timesteps: list[torch.Tensor] = []

    def capture(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        self.feature = inputs[0].detach().cpu().contiguous()

    def record(self, timestep: torch.Tensor) -> None:
        step_index = self.call_index // 2
        branch = "cond" if self.call_index % 2 == 0 else "uncond"
        torch.save(self.feature, self.output_dir / f"step_{step_index:02d}_{branch}.pt")
        if branch == "cond":
            self.timesteps.append(timestep.detach().reshape(-1)[0].cpu())
        self.call_index += 1


def collect_prompt(
    prompt_id: int,
    prompt: str,
    seed: int,
    cfg: DictConfig,
    runtime: WanWorkerRuntime,
    project_root: Path,
    output_root: Path,
) -> None:
    sample_steps = int(cfg.generation.sample_steps)
    prompt_dir = output_root / f"prompt_{prompt_id:04d}"
    timestep_path = prompt_dir / "timesteps.pt"
    if timestep_path.is_file():
        print(f"[GPU {runtime.device_id}] Reuse prompt {prompt_id}", flush=True)
        return

    prompt_dir.mkdir(parents=True, exist_ok=True)
    runtime.ensure_loaded()
    recorder = FeatureRecorder(prompt_dir)
    original_forward = runtime.pipeline.model.forward

    def collecting_forward(_model, x, t, context, seq_len, clip_fea=None, y=None):
        kwargs = {}
        if clip_fea is not None:
            kwargs["clip_fea"] = clip_fea
        if y is not None:
            kwargs["y"] = y
        output = original_forward(x, t, context, seq_len, **kwargs)
        recorder.record(t)
        return output

    image = None
    if cfg.generation.image is not None:
        image = Image.open(
            _resolve(str(cfg.generation.image), project_root)
        ).convert("RGB")

    print(f"[GPU {runtime.device_id}] Collect prompt {prompt_id}", flush=True)
    head_hook = runtime.pipeline.model.head.register_forward_pre_hook(recorder.capture)
    try:
        with patch_forward(runtime.pipeline.model, collecting_forward):
            runtime.pipeline.generate(
                prompt,
                img=image,
                size=runtime.size_configs[str(cfg.generation.size)],
                max_area=runtime.max_area_configs[str(cfg.generation.size)],
                frame_num=int(cfg.generation.frame_num),
                shift=float(cfg.generation.sample_shift),
                sample_solver=str(cfg.generation.sample_solver),
                sampling_steps=sample_steps,
                guide_scale=float(cfg.generation.sample_guide_scale),
                seed=seed,
                offload_model=bool(cfg.generation.offload_model),
            )
    finally:
        head_hook.remove()
    torch.save(torch.stack(recorder.timesteps), timestep_path)
    print(f"[GPU {runtime.device_id}] Saved prompt {prompt_id}", flush=True)


def run_worker(
    local_rank: int,
    world_size: int,
    prompts: list[str],
    seeds: list[int],
    cfg_payload: dict[str, Any],
    project_root: Path,
    output_root: Path,
) -> None:
    cfg = OmegaConf.create(cfg_payload)
    cfg.runtime.device_id = local_rank
    runtime = WanWorkerRuntime(cfg, project_root, local_rank)
    for prompt_id in range(local_rank, len(prompts), world_size):
        collect_prompt(
            prompt_id,
            prompts[prompt_id],
            seeds[prompt_id],
            cfg,
            runtime,
            project_root,
            output_root,
        )


@hydra.main(version_base=None, config_path="conf", config_name="collect_features")
def main(cfg: DictConfig) -> None:
    project_root = _resolve(str(cfg.paths.project_root), Path(get_original_cwd()))
    output_root = _resolve(str(cfg.paths.output_root), project_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_path = _resolve(str(cfg.prompts.file), project_root)
    prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_path = output_root / "manifest.jsonl"
    if manifest_path.is_file():
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompts = [record["prompt"] for record in records]
        seeds = [int(record["seed"]) for record in records]
    else:
        seeds = [secrets.randbits(63) for _ in prompts]
        manifest_path.write_text(
            "".join(
                json.dumps(
                    {
                        "trajectory_id": index,
                        "prompt": prompt,
                        "seed": seeds[index],
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for index, prompt in enumerate(prompts)
            ),
            encoding="utf-8",
        )
    dataset_config = OmegaConf.create(
        {
            "schema_version": 1,
            "feature": {
                "location": "transformer output before model.head",
                "branches": ["conditional", "unconditional"],
                "file_pattern": (
                    "prompt_{trajectory_id:04d}/"
                    "step_{step_index:02d}_{branch}.pt"
                ),
                "timestep_file": "prompt_{trajectory_id:04d}/timesteps.pt",
            },
            "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        }
    )
    OmegaConf.save(dataset_config, output_root / "dataset.yaml", resolve=True)

    num_processes = min(
        _resolve_num_processes(cfg.parallel.num_processes),
        len(prompts),
    )
    worker_args = (
        num_processes,
        prompts,
        seeds,
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
