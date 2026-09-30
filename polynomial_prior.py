"""Portable piecewise-polynomial prior and causal node correction (NumPy only).

The offline coefficients describe log(output-change ratio), not tensor values.
This module does not decide which DiT steps to execute or train a neural net.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


def cubic_bspline_basis(steps, breaks):
    """Open, clamped cubic B-spline basis; internal joins are C2 continuous."""
    x = np.asarray(steps, dtype=float).reshape(-1)
    breaks = np.asarray(breaks, dtype=float)
    if len(breaks) < 2 or not np.all(np.diff(breaks) > 0):
        raise ValueError("breaks must strictly increase")
    if not np.isfinite(x).all() or np.any(x < breaks[0]) or np.any(x > breaks[-1]):
        raise ValueError("evaluation outside the trained step range")
    knots = np.r_[np.repeat(breaks[0], 4), breaks[1:-1], np.repeat(breaks[-1], 4)]
    b = ((x[:, None] >= knots[:-1]) & (x[:, None] < knots[1:])).astype(float)
    for degree in range(1, 4):
        next_b = np.zeros((len(x), len(knots)-degree-1))
        for j in range(next_b.shape[1]):
            left = knots[j+degree]-knots[j]
            right = knots[j+degree+1]-knots[j+1]
            if left > 0:
                next_b[:, j] += (x-knots[j])/left*b[:, j]
            if right > 0:
                next_b[:, j] += (knots[j+degree+1]-x)/right*b[:, j+1]
        b = next_b
    b[x == breaks[-1], -1] = 1.
    return b


def fit_piecewise(train_log, steps, breaks, roughness):
    """Least-squares coefficients on train trajectories, with curvature penalty."""
    train_log = np.asarray(train_log, dtype=float)
    if train_log.ndim != 2 or train_log.shape[1] != len(steps) or not np.isfinite(train_log).all():
        raise ValueError("train_log must be a finite [trajectory, step] array")
    if not np.isfinite(roughness) or roughness < 0:
        raise ValueError("roughness must be finite and nonnegative")
    design = cubic_bspline_basis(steps, breaks)
    penalty = np.diff(design, n=2, axis=0)
    weights = np.linalg.lstsq(np.vstack([design, np.sqrt(roughness)*penalty]),
                            np.r_[train_log.mean(axis=0), np.zeros(len(penalty))], rcond=1e-12)[0]
    # Export four ordinary power coefficients per segment, in local z=[0,1].
    z = np.linspace(0., 1., 4)
    vandermonde = np.polynomial.polynomial.polyvander(z, 3)
    coefficients = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        y = cubic_bspline_basis(left+(right-left)*z, breaks) @ weights
        coefficients.append(np.linalg.solve(vandermonde, y).tolist())
    spec = {"degree": 3, "break_steps": list(map(float, breaks)),
            "coefficients_ascending": coefficients, "roughness": float(roughness),
            "coefficient_coordinate": "z=(step-left)/(right-left)",
            "value_transform": "q=exp(polynomial(z))", "continuity": "C2"}
    return design @ weights, spec


@dataclass(frozen=True)
class OnlineFitConfig:
    degree: int = 1
    window: int = 5
    ridge_before_axis: float = 1.
    ridge_after_axis: float = .1
    mirror_weight: float = .01
    coordinate_scale: float = 8.

    def __post_init__(self):
        if self.degree not in (1, 2, 3) or self.window < 2:
            raise ValueError("invalid online degree or window")
        for key in ("ridge_before_axis", "ridge_after_axis", "mirror_weight", "coordinate_scale"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")


def correction_weights(observed_steps, targets, center, config, *,
                       use_online_fit, use_mirror_node, mirror_node_mode="observed",
                       domain_end=48., allow_past_queries=False):
    """Return weights on REAL observed deviations, never on predicted history."""
    if type(use_online_fit) is not bool or type(use_mirror_node) is not bool:
        raise ValueError("switches must be bool")
    if use_mirror_node and not use_online_fit:
        raise ValueError("use_mirror_node requires use_online_fit")
    if mirror_node_mode not in ("observed", "anchor"):
        raise ValueError("mirror_node_mode must be observed or anchor")
    observed = np.asarray(observed_steps, dtype=float)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if (observed.ndim != 1 or not len(observed) or not np.isfinite(observed).all()
            or not np.all(np.diff(observed) > 0) or not np.isfinite(targets).all()
            or (not allow_past_queries and not np.all(targets >= observed[-1]))
            or not np.all(targets <= domain_end)):
        raise ValueError("invalid real-node history or future query")
    result = np.zeros((len(targets), len(observed)))
    result[:, -1] = 1.
    if not use_online_fit or len(observed) < 2:
        return result
    local = np.arange(max(0, len(observed)-config.window), len(observed))
    x = (observed[local]-observed[-1])/config.coordinate_scale
    design = np.polynomial.polynomial.polyvander(x, config.degree)[:, 1:]/np.sqrt(len(local))
    response = np.zeros((len(local), len(observed)))
    response[np.arange(len(local)), local] += 1.
    response[:, -1] -= 1.
    response /= np.sqrt(len(local))
    if use_mirror_node and observed[-1] >= center:
        mirrored = 2*center-observed
        mask = (mirrored > observed[-1]) & (mirrored <= domain_end)
        sources = np.flatnonzero(mask)
        if len(sources):
            factor = np.sqrt(config.mirror_weight/len(sources))
            x_mirror = (mirrored[mask]-observed[-1])/config.coordinate_scale
            design = np.vstack([design, factor*np.polynomial.polynomial.polyvander(x_mirror, config.degree)[:, 1:]])
            virtual = np.zeros((len(sources), len(observed)))
            if mirror_node_mode == "observed":
                virtual[np.arange(len(sources)), sources] += 1.
                virtual[:, -1] -= 1.
            response = np.vstack([response, factor*virtual])
    ridge = config.ridge_after_axis if observed[-1] >= center else config.ridge_before_axis
    design = np.vstack([design, np.sqrt(ridge)*np.eye(config.degree)])
    response = np.vstack([response, np.zeros((config.degree, len(observed)))])
    coef = np.linalg.lstsq(design, response, rcond=1e-12)[0]
    future = np.polynomial.polynomial.polyvander((targets-observed[-1])/config.coordinate_scale, config.degree)[:, 1:]
    return result+future @ coef


class PiecewisePolynomialPrior:
    def __init__(self, artifact):
        if artifact.get("kind") != "piecewise_log_change_prior" or artifact.get("version") != 1:
            raise ValueError("unsupported piecewise polynomial artifact")
        self.artifact = artifact
        self.breaks = np.array(artifact["polynomial"]["break_steps"], dtype=float)
        self.coefficients = np.array(artifact["polynomial"]["coefficients_ascending"], dtype=float)
        if (self.coefficients.shape != (len(self.breaks)-1, 4)
                or not np.all(np.diff(self.breaks) > 0)
                or not np.isfinite(self.breaks).all() or not np.isfinite(self.coefficients).all()):
            raise ValueError("invalid piecewise cubic coefficients")
        self.center = float(artifact["mirror_axis_step"])
        if not self.breaks[0] < self.center < self.breaks[-1]:
            raise ValueError("invalid mirror axis")
        self.online_config = OnlineFitConfig(**artifact["online_fit"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def log_value(self, steps):
        x = np.asarray(steps, dtype=float)
        if not np.isfinite(x).all() or np.any(x < self.breaks[0]) or np.any(x > self.breaks[-1]):
            raise ValueError("outside the trained step domain; do not silently extrapolate")
        segments = np.clip(np.searchsorted(self.breaks, x, side="right")-1, 0, len(self.breaks)-2)
        z = (x-self.breaks[segments])/(self.breaks[segments+1]-self.breaks[segments])
        c = self.coefficients[segments]
        return ((c[..., 3]*z+c[..., 2])*z+c[..., 1])*z+c[..., 0]

    def value(self, steps):
        return np.exp(self.log_value(steps))


class OnlinePolynomial:
    """Anchored prior, optional online ridge fit, optional soft mirror nodes.

    observe() accepts exact adjacent-step q measurements. observe_interval()
    uses a separately identified chord-change proxy for sparse real nodes;
    its fit coordinate is the prior-weighted midpoint of that interval.
    """
    def __init__(self, prior, *, use_online_fit=True, use_mirror_node=False,
                 mirror_node_mode="observed"):
        self.prior = prior
        self.options = dict(use_online_fit=use_online_fit, use_mirror_node=use_mirror_node,
                            mirror_node_mode=mirror_node_mode)
        correction_weights([prior.breaks[0]], [prior.breaks[0]], prior.center,
                           prior.online_config, domain_end=prior.breaks[-1], **self.options)
        self.observed_steps, self.observed_log_values = [], []
        self.latest_observation_end = None

    def observe(self, step, q):
        self.prior.log_value(step)  # domain check before updating history
        if self.latest_observation_end is not None and step <= self.latest_observation_end:
            raise ValueError("observed nodes must strictly advance")
        if not math.isfinite(q) or q < 0:
            raise ValueError("observed change must be finite and nonnegative")
        self.observed_steps.append(float(step))
        self.observed_log_values.append(math.log(max(float(q), 1e-8)))
        self.latest_observation_end = float(step)

    def observe_interval(self, start_step, end_step, relative_change):
        """Calibrate from two nonadjacent REAL outputs, not a measured q_s.

        The endpoint distance / start-output norm approximates sum(q_s) only
        when output directions and norms vary slowly across the interval.
        A chord can underestimate variation when the trajectory turns back.
        Match its log ratio to the prior's interval mass, placing this derived
        calibration at the prior-weighted step (a first-order moment fit).
        No individual skipped q or output is claimed to have been observed.
        """
        if (not math.isfinite(start_step) or not math.isfinite(end_step)
                or not float(start_step).is_integer() or not float(end_step).is_integer()
                or end_step-start_step <= 1):
            raise ValueError("interval calibration requires nonadjacent integer steps")
        if self.latest_observation_end is not None and start_step < self.latest_observation_end:
            raise ValueError("real observation intervals must not overlap")
        if not math.isfinite(relative_change) or relative_change < 0:
            raise ValueError("interval change must be finite and nonnegative")
        steps = np.arange(int(start_step)+1, int(end_step)+1, dtype=float)
        log_prior = self.prior.log_value(steps)
        largest = float(log_prior.max())
        weights = np.exp(log_prior-largest)
        log_mass = largest+math.log(float(weights.sum()))
        fit_step = float(steps @ weights / weights.sum())
        log_correction = math.log(max(float(relative_change), 1e-8))-log_mass
        self.observed_steps.append(fit_step)
        self.observed_log_values.append(float(self.prior.log_value(fit_step))+log_correction)
        self.latest_observation_end = float(end_step)
        return {"kind":"interval_change_proxy", "start_step":int(start_step),
                "end_step":int(end_step), "fit_step":fit_step,
                "relative_change":float(relative_change),
                "log_prior_interval_sum":log_mass, "log_correction":log_correction}

    def predict_log(self, steps):
        base = self.prior.log_value(steps)
        if not self.observed_steps:
            return base  # offline curve before any real node arrives
        if np.any(np.asarray(steps) < self.latest_observation_end):
            raise ValueError("prediction must not precede the latest real observation")
        weights = correction_weights(self.observed_steps, steps, self.prior.center,
                                     self.prior.online_config, domain_end=self.prior.breaks[-1], **self.options)
        residual = np.array(self.observed_log_values)-self.prior.log_value(self.observed_steps)
        correction = weights @ residual
        return base+correction.reshape(np.shape(base))

    def predict(self, steps):
        with np.errstate(over="raise", invalid="raise"):
            return np.exp(self.predict_log(steps))

    def fitted_log_curve(self, steps):
        """Evaluate the CURRENT fitted curve on past/future coordinates.

        Tensor reconstruction needs a common coordinate for its historical
        real nodes and future query. This is a retrospective fit using only
        the real observations already supplied, never future observations.
        predict_log still rejects queries before the latest observation.
        """
        base = self.prior.log_value(steps)
        if not self.observed_steps:
            return base
        weights = correction_weights(self.observed_steps, steps, self.prior.center,
            self.prior.online_config, domain_end=self.prior.breaks[-1],
            allow_past_queries=True, **self.options)
        residual = np.array(self.observed_log_values)-self.prior.log_value(self.observed_steps)
        return base+(weights @ residual).reshape(np.shape(base))
