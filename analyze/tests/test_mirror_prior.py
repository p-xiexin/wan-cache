"""Causality and known-symmetry checks for mirror-node experiments."""
import unittest

import numpy as np

from analyze.analyze_polynomial_prior import HORIZONS
from analyze.analyze_mirror_prior import (
    interpolate_observed, mirror_forecasts, virtual_node_forecast,
)


class MirrorPriorTests(unittest.TestCase):
    def test_fractional_mirror_cannot_use_unobserved_bracket(self):
        steps = np.arange(1, 25)
        values = steps[None, :].astype(float)
        self.assertIsNone(interpolate_observed(values, steps, 9.1, 8))
        self.assertIsNone(interpolate_observed(values, steps, .9, 8))
        np.testing.assert_array_equal(interpolate_observed(values, steps, 8.5, 8), [8.5])

    def test_no_future_truth_in_mirrors_or_virtual_fits(self):
        rng = np.random.default_rng(12)
        steps = np.arange(1, 49)
        values = rng.normal(size=(3, 48))
        changed = values.copy()
        changed[:, 30:] += rng.normal(size=(3, 18))*100
        prior = rng.normal(size=48)
        a = mirror_forecasts(values, prior, steps, [29], 26.4)
        b = mirror_forecasts(changed, prior, steps, [29], 26.4)
        for kind in a[1]:
            np.testing.assert_array_equal(a[1][kind], b[1][kind])
        a = virtual_node_forecast(values, prior, steps, [29], 26.4)
        b = virtual_node_forecast(changed, prior, steps, [29], 26.4)
        np.testing.assert_array_equal(a, b)

    def test_asymmetry_compensated_mirror_recovers_known_residual(self):
        steps = np.arange(1, 25)
        prior = -.05*steps
        deviation = .002*(steps-13)**2
        values = prior[None, :] + deviation[None, :]
        _, result, mask = mirror_forecasts(values, prior, steps, [14], 13.)
        self.assertTrue(mask["level"].all())
        np.testing.assert_allclose(result["level"][0, 0], values[0, 14+HORIZONS], atol=1e-12)

    def test_virtual_nodes_preserve_a_truly_symmetric_quadratic(self):
        steps = np.arange(1, 49)
        prior = -.04*steps
        residual = .001*(steps-26.4)**2
        values = prior[None, :] + residual[None, :]
        pred = virtual_node_forecast(values, prior, steps, [23], 26.4,
                                     degree=2, window=3, ridge=0., mirror_weight=1.)
        np.testing.assert_allclose(pred[0, 0], values[0, 23+HORIZONS], atol=1e-11)

    def test_geometry_control_does_not_read_old_mirror_values(self):
        rng = np.random.default_rng(19)
        steps = np.arange(1, 49)
        values = rng.normal(size=(2, 48))
        changed = values.copy()
        changed[:, 9] += 2  # mirror source, outside the five-point local history
        prior = np.zeros(48)
        a = virtual_node_forecast(values, prior, steps, [29], 26.4, virtual_target="repeat_anchor")
        b = virtual_node_forecast(changed, prior, steps, [29], 26.4, virtual_target="repeat_anchor")
        np.testing.assert_array_equal(a, b)
        a = virtual_node_forecast(values, prior, steps, [29], 26.4)
        b = virtual_node_forecast(changed, prior, steps, [29], 26.4)
        self.assertGreater(float(np.max(np.abs(a-b))), 1e-5)


if __name__ == "__main__":
    unittest.main()
