"""Check causal information access and nonuniform-time forecasting."""
import unittest

import numpy as np

from analyze.analyze_polynomial_prior import HORIZONS, forecast


class PolynomialPriorTests(unittest.TestCase):
    def test_future_truth_cannot_change_predictions(self):
        rng = np.random.default_rng(32)
        values = rng.normal(size=(4, 24))
        changed = values.copy()
        changed[:, 9:] += rng.normal(size=(4, 15)) * 100
        x = np.linspace(0, 1, 24)**2
        prior = np.sin(x)
        for config in ({"kind": "persistence"}, {"kind": "prior"},
                       {"kind": "local", "degree": 2},
                       {"kind": "correction", "degree": 0},
                       {"kind": "correction", "degree": 2, "ridge": 1.}):
            with self.subTest(config=config):
                before = forecast(values, x, [8], prior, **config)
                after = forecast(changed, x, [8], prior, **config)
                np.testing.assert_array_equal(before, after)

    def test_nonuniform_quadratic_continuation(self):
        x = np.linspace(0, 1, 24)**3
        values = (2+3*x-4*x*x)[None, :]
        pred = forecast(values, x, [8], np.zeros(24), "local", degree=2)
        np.testing.assert_allclose(pred[0, 0], values[0, 8+HORIZONS], atol=1e-12)

    def test_observed_sample_scale_calibrates_shared_prior(self):
        x = np.linspace(0, 1, 24)**2
        prior = np.sin(x)*.4 - 4
        scales = np.log([.7, 1.4, 2.])
        values = prior[None, :] + scales[:, None]
        pred = forecast(values, x, [8], prior, "correction", degree=0)
        np.testing.assert_allclose(pred[:, 0], values[:, 8+HORIZONS], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
