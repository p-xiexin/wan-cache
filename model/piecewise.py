"""Frozen scalar polynomial prior + real-node tensor trajectory extrapolation.

The prior supplies relative-change progress rho(t) = sum_{s<=t} q_hat(s).
A degree-1/2 polynomial through recent FULL output tensors is evaluated at
rho(t). Tensor directions are supplied by those real outputs, not by q_hat.
This is an experimental lifting of the scalar prior into a tensor predictor;
scalar validation alone does not validate its closed-loop generation quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from eval.model.base import CacheMethod, TensorList, _mean_abs, _mean_abs_difference
from eval.polynomial_prior import OnlinePolynomial, PiecewisePolynomialPrior


@dataclass
class OutputNode:
    step: int
    outputs: TensorList


def lagrange_weights(nodes, target):
    """Small FP64 interpolation solve in a scaled progress coordinate."""
    nodes = np.asarray(nodes, dtype=float)
    if (nodes.ndim != 1 or len(nodes) < 2 or not np.isfinite(nodes).all()
            or not np.all(np.diff(nodes) > 0) or not math.isfinite(target)):
        raise ValueError("polynomial nodes must be finite and strictly increasing")
    scale = nodes[-1]-nodes[0]
    x, query = (nodes-nodes[-1])/scale, (target-nodes[-1])/scale
    weights = np.ones(len(nodes))
    for j in range(len(nodes)):
        for k in range(len(nodes)):
            if k != j:
                weights[j] *= (query-x[k])/(x[j]-x[k])
    if not np.isfinite(weights).all():
        raise FloatingPointError("nonfinite tensor polynomial weights")
    return weights


class PiecewisePolynomialMethod(CacheMethod):
    """Direct DiT-output prediction; no neural predictor or adaptive threshold.

    All ablations use the same fixed real-node schedule. Two consecutive real
    CFG pairs per refresh block provide valid adjacent conditional q samples.
    Both CFG branches share that fitted scalar coordinate and keep separate
    output tensors. Predicted outputs never update the real-node histories.
    """
    name = "piecewise_polynomial"

    def __init__(self, artifact_path: str, use_online_fit=True, use_mirror_node=False,
                 mirror_node_mode="observed", tensor_degree=1, skip_steps=2,
                 full_block_steps=2):
        self.prior = PiecewisePolynomialPrior.load(artifact_path)
        self.artifact_sha256 = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()
        self.options = dict(use_online_fit=use_online_fit, use_mirror_node=use_mirror_node,
                            mirror_node_mode=mirror_node_mode)
        OnlinePolynomial(self.prior, **self.options)  # validate switch combination
        if type(tensor_degree) is not int or tensor_degree not in (1, 2):
            raise ValueError("tensor_degree must be 1 or 2")
        if type(skip_steps) is not int or skip_steps < 1:
            raise ValueError("skip_steps must be a positive integer")
        if type(full_block_steps) is not int or full_block_steps < 2:
            raise ValueError("full_block_steps must be >=2 for adjacent real q observations")
        self.tensor_degree = tensor_degree
        self.skip_steps = skip_steps
        self.full_block_steps = full_block_steps
        artifact = self.prior.artifact
        self.trained_generation = artifact["generation"]
        grid = np.asarray(artifact["step_grid"])
        times = np.asarray(artifact["model_timestep_grid"], dtype=float)
        expected = np.arange(1, int(self.trained_generation["sampling_steps"])-1)
        if (not np.array_equal(grid, expected) or times.shape != grid.shape
                or not np.isfinite(times).all() or not np.all(np.diff(times) < 0)
                or self.prior.breaks[0] != grid[0] or self.prior.breaks[-1] != grid[-1]):
            raise ValueError("artifact must cover transition steps 1..sample_steps-2")
        self.expected_timesteps = dict(zip(map(int,grid), map(float,times)))

    def reset(self, sample_steps, warmup_steps, final_full_steps):
        if sample_steps != self.trained_generation["sampling_steps"]:
            raise ValueError("sample_steps differs from the trained polynomial prior")
        if (warmup_steps < max(3,self.tensor_degree+1) or final_full_steps < 1
                or warmup_steps+final_full_steps > sample_steps):
            raise ValueError("piecewise method requires >=3 warmup and >=1 final full step")
        super().reset(sample_steps,warmup_steps,final_full_steps)

    def validate_generation(self, generation):
        mapping = {"task":"task", "size":"size", "frame_num":"frame_num",
                   "sample_steps":"sampling_steps", "sample_solver":"sample_solver",
                   "sample_shift":"sample_shift", "sample_guide_scale":"guide_scale"}
        for runtime_key, trained_key in mapping.items():
            if generation[runtime_key] != self.trained_generation[trained_key]:
                raise ValueError(f"generation.{runtime_key} differs from the offline prior")
        if generation.get("image") is not None:
            raise ValueError("this prior supports the trained text-to-video configuration")

    def _reset_method(self):
        self.trajectory = OnlinePolynomial(self.prior, **self.options)
        self.nodes = {True:[], False:[]}
        self.q_observations = []
        self.predictions = []
        self._weight_step = None
        self._weights = None

    @property
    def history_ready(self):
        return (super().history_ready
                and all(len(nodes) >= self.tensor_degree+1 for nodes in self.nodes.values()))

    def is_full_step(self, step):
        if step < self.warmup_steps or step >= self.sample_steps-self.final_full_steps:
            return True
        phase = (step-self.warmup_steps) % (self.skip_steps+self.full_block_steps)
        return phase >= self.skip_steps

    def decide_conditional(self, raw_input, timestep):
        del raw_input, timestep
        return not self.is_full_step(self.pair_index)

    def try_skip(self, raw_input, timestep):
        step = self.pair_index
        if step >= self.sample_steps:
            raise RuntimeError("more CFG pairs than configured sample_steps")
        observed = torch.as_tensor(timestep).detach().flatten().float()
        if not bool(torch.isfinite(observed).all()) or not bool((observed == observed[0]).all()):
            raise ValueError("expected one finite model timestep for the CFG call")
        if step in self.expected_timesteps and not math.isclose(
                float(observed[0]), self.expected_timesteps[step], abs_tol=1e-4):
            raise ValueError("model timestep differs from the trained step grid")
        return super().try_skip(raw_input,timestep)

    def update(self, raw_input, output):
        branch = self.forward_index % 2 == 0
        step = self.pair_index
        if not self.is_full_step(step):
            raise RuntimeError("only scheduled real DiT outputs may enter node history")
        nodes = self.nodes[branch]
        if nodes and step <= nodes[-1].step:
            raise ValueError("real node steps must strictly advance")
        if len(raw_input) != len(output) or not len(output):
            raise ValueError("raw input/output tensor lists differ")
        if any(x.shape != v.shape for x,v in zip(raw_input,output)):
            raise ValueError("DiT input/output shapes differ")
        real = [v.detach().float().clone() for v in output]
        if nodes and (len(nodes[-1].outputs) != len(real) or any(
                v.shape != old.shape for v,old in zip(real,nodes[-1].outputs))):
            raise ValueError("real output layout changed within a trajectory")
        if branch and nodes and nodes[-1].step == step-1 and step in self.expected_timesteps:
            q = _mean_abs_difference(real,nodes[-1].outputs)/(_mean_abs(nodes[-1].outputs)+1e-8)
            self.trajectory.observe(step,q)
            self.q_observations.append({"step":step,"q":q})
        nodes.append(OutputNode(step,real))
        del nodes[:max(0,len(nodes)-self.tensor_degree-1)]
        self._weight_step = None
        super().update(raw_input,output)

    def _tensor_weights(self, step):
        if self._weight_step == step:
            return self._weights
        node_steps = np.array([n.step for n in self.nodes[True]],dtype=int)
        if [n.step for n in self.nodes[False]] != node_steps.tolist():
            raise RuntimeError("CFG real-node histories do not match")
        query_steps = np.arange(node_steps[0]+1,step+1)
        log_q = self.trajectory.fitted_log_curve(query_steps)
        if not np.isfinite(log_q).all():
            raise FloatingPointError("nonfinite fitted change curve")
        # Overall scale cancels in polynomial interpolation. Subtracting the
        # maximum prevents exp overflow without changing the reconstruction.
        increments = np.exp(log_q-float(np.max(log_q)))
        progress = np.r_[0.,np.cumsum(increments)]
        node_coordinates = progress[node_steps-node_steps[0]]
        self._weights = lagrange_weights(node_coordinates,float(progress[-1]))
        self._weight_step = step
        self.predictions.append({"step":step,"real_node_steps":node_steps.tolist(),
            "tensor_weights":self._weights.tolist(),
            "log_predicted_q":float(log_q[-1]),
            "latest_scalar_node":self.trajectory.observed_steps[-1] if self.trajectory.observed_steps else None})
        return self._weights

    def predict_cached_output(self, raw_input, timestep, is_conditional):
        del timestep
        weights = self._tensor_weights(self.pair_index)
        nodes = self.nodes[is_conditional]
        latest = nodes[-1].outputs
        if len(raw_input) != len(latest) or any(x.shape != v.shape for x,v in zip(raw_input,latest)):
            raise ValueError("cached output layout differs from the current DiT input")
        outputs = []
        for item, value in enumerate(latest):
            predicted = value.clone()
            # Anchor form avoids cancellation of the constant component.
            for weight,node in zip(weights[:-1],nodes[:-1]):
                predicted.add_(node.outputs[item]-value,alpha=float(weight))
            outputs.append(predicted)
        return outputs

    def summary(self):
        return {**super().summary(), **self.options,
            "artifact_sha256":self.artifact_sha256,"tensor_degree":self.tensor_degree,
            "reconstruction":"polynomial in cumulative predicted relative-change coordinate",
            "schedule":{"kind":"fixed_blocks","skip_steps":self.skip_steps,
                        "full_block_steps":self.full_block_steps},
            "planned_full_pairs":[s for s in range(self.sample_steps) if self.is_full_step(s)],
            "q_source":"adjacent real conditional outputs; coordinate shared by CFG branches",
            "mirror_axis_step":self.prior.center,"online_fit_parameters":self.prior.artifact["online_fit"],
            "real_scalar_nodes":self.q_observations,"predictions":self.predictions}
