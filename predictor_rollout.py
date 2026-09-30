"""Short differentiable cache segments with an actual multistep solver."""

from __future__ import annotations

import math

import torch

from eval.model.polynomial import PolynomialMethod


def detach_solver_history(scheduler):
    """Truncate gradients without resetting any numerical solver state."""
    scheduler.model_outputs = [
        value.detach() if torch.is_tensor(value) else value
        for value in scheduler.model_outputs
    ]
    if torch.is_tensor(getattr(scheduler, "last_sample", None)):
        scheduler.last_sample = scheduler.last_sample.detach()


def rollout(
    model, criterion, teacher, scheduler, initial_latent, *,
    cache_threshold, warmup_steps, final_full_steps, guide_scale,
    window_steps=4, optimizer=None, grad_clip=1.0, on_state=None,
):
    """teacher(x, timestep, conditional) evaluates a frozen DiT on current x.

    Teacher probes on skipped nodes are labels only. All solver updates use
    the selected prediction, and full refreshes end a gradient segment.
    """
    if not 1 <= window_steps <= 4:
        raise ValueError("rollout window_steps must be between 1 and 4")
    method = PolynomialMethod(model=model, cache_threshold=cache_threshold)
    method.reset(len(scheduler.timesteps), warmup_steps, final_full_steps)
    method.set_schedule(scheduler.sigmas, scheduler.timesteps)
    x = initial_latent.detach().float()
    pending = []
    totals = dict(loss=0.0, mae=0.0, reuse_mae=0.0, quadratic_mae=0.0, samples=0)

    def finish_segment():
        nonlocal x
        if pending and optimizer is not None:
            loss = torch.stack(pending).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        pending.clear()
        x = x.detach()
        detach_solver_history(scheduler)
        for name in ["previous_raw_input_even", "prev_previous_raw_input_even"]:
            values = getattr(method, name)
            if values is not None:
                setattr(method, name, [v.detach() for v in values])
        method.last_coefficients.clear()

    with torch.set_grad_enabled(optimizer is not None):
        for step, timestep in enumerate(scheduler.timesteps):
            if on_state is not None:
                on_state(step, x.detach())
            conditional = method.try_skip([x], timestep)
            if conditional is None:
                finish_segment()
                with torch.no_grad():
                    v_c = teacher(x, timestep, True).float()
                method.update([x], [v_c])
                assert method.try_skip([x], timestep) is None
                with torch.no_grad():
                    v_u = teacher(x, timestep, False).float()
                method.update([x], [v_u])
            else:
                v_c = conditional[0]
                v_u = method.try_skip([x], timestep)[0]
                branch_losses = []
                for branch, prediction in [(True, v_c), (False, v_u)]:
                    with torch.no_grad():
                        target = teacher(x.detach(), timestep, branch).float()
                    coefficients = method.last_coefficients[branch]
                    loss = criterion(prediction[None], target[None], coefficients)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite predictor rollout loss")
                    branch_losses.append(loss)
                    bases, _ = method.histories[branch].features(
                        step, float(method.sigmas[step]),
                    )
                    totals["loss"] += float(loss.detach())
                    totals["mae"] += float((prediction.detach() - target).abs().mean())
                    totals["reuse_mae"] += float((x.detach() + bases[0, 0] - target).abs().mean())
                    totals["quadratic_mae"] += float((x.detach() + bases[0].sum(0) - target).abs().mean())
                    totals["samples"] += 1
                pending.append(torch.stack(branch_losses).mean())
            guided = v_u + guide_scale * (v_c - v_u)
            x = scheduler.step(
                guided[None], timestep, x[None], return_dict=False,
            )[0][0]
            if len(pending) == window_steps:
                finish_segment()
        finish_segment()
        if on_state is not None:
            on_state(len(scheduler.timesteps), x)
    return totals, method.summary(), x


class WanTeacher:
    """Prepare text once; reuse the frozen Wan DiT without VAE decoding."""

    def __init__(self, pipeline, prompt, latent_shape, offload_text=True):
        self.pipeline = pipeline
        self.model = pipeline.model.eval().requires_grad_(False)
        device = pipeline.device
        text_device = torch.device("cpu") if pipeline.t5_cpu else device
        with torch.no_grad():
            pipeline.text_encoder.model.to(text_device)
            self.context = pipeline.text_encoder([prompt], text_device)
            self.null_context = pipeline.text_encoder([pipeline.sample_neg_prompt], text_device)
            self.context = [value.to(device) for value in self.context]
            self.null_context = [value.to(device) for value in self.null_context]
            if offload_text:
                pipeline.text_encoder.model.cpu()
        self.model.to(device)
        _, frames, height, width = latent_shape
        patch = pipeline.patch_size
        self.seq_len = math.ceil(frames * height * width / (patch[1] * patch[2]))

    @torch.no_grad()
    def __call__(self, x, timestep, conditional):
        with torch.autocast("cuda", dtype=self.pipeline.param_dtype):
            return self.model(
                [x], t=timestep.expand(1, self.seq_len),
                context=self.context if conditional else self.null_context,
                seq_len=self.seq_len,
            )[0]
