"""Export fidelity, causal node fitting, and ablation contracts."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from polynomial_prior import (OnlineFitConfig, OnlinePolynomial, PiecewisePolynomialPrior,
                              correction_weights, cubic_bspline_basis, fit_piecewise)


def example_prior():
    steps = np.arange(1,49)
    y = -4.6+.001*(steps-27)**2
    _, spec = fit_piecewise(y[None],steps,[1,8,20,28,40,48],0.)
    return PiecewisePolynomialPrior({"kind":"piecewise_log_change_prior","version":1,
        "polynomial":spec,"mirror_axis_step":26.4,"online_fit":asdict(OnlineFitConfig())})


class PiecewiseExportTests(unittest.TestCase):
    def test_piecewise_cubic_recovers_known_polynomial_and_continuous_joins(self):
        steps=np.arange(1,49)
        u=(steps-1)/47
        y=-3.+.6*u-.3*u*u+.9*u*u*u
        fitted,spec=fit_piecewise(np.stack([y-.1,y+.1]),steps,[1,8,20,28,40,48],0.)
        np.testing.assert_allclose(fitted,y,atol=2e-12)
        c=np.array(spec['coefficients_ascending']); widths=np.diff(spec['break_steps'])
        np.testing.assert_allclose(c[:-1].sum(axis=1),c[1:,0],atol=2e-12)
        np.testing.assert_allclose((c[:-1,1]+2*c[:-1,2]+3*c[:-1,3])/widths[:-1],
                                   c[1:,1]/widths[1:],atol=2e-12)
        np.testing.assert_allclose((2*c[:-1,2]+6*c[:-1,3])/widths[:-1]**2,
                                   2*c[1:,2]/widths[1:]**2,atol=2e-12)
        b=cubic_bspline_basis(np.linspace(1,48,333),[1,8,20,28,40,48])
        np.testing.assert_allclose(b.sum(axis=1),1.,atol=1e-14)
        self.assertGreaterEqual(float(b.min()),0.)

    def test_json_roundtrip_and_closed_domain(self):
        prior=example_prior()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prior.json'; path.write_text(json.dumps(prior.artifact))
            loaded=PiecewisePolynomialPrior.load(path)
            np.testing.assert_array_equal(loaded.value([1,26.4,48]),prior.value([1,26.4,48]))
        for step in (0,49,float('nan')):
            with self.assertRaises(ValueError): prior.value(step)

    def test_every_ablation_preserves_last_real_anchor(self):
        prior=example_prior()
        for fit,mirror,mode in ((False,False,'observed'),(True,False,'observed'),
                                (True,True,'observed'),(True,True,'anchor')):
            runtime=OnlinePolynomial(prior,use_online_fit=fit,use_mirror_node=mirror,mirror_node_mode=mode)
            for step,q in ((5,.04),(10,.028),(20,.021),(27,.025),(30,.024)):
                runtime.observe(step,q)
            self.assertAlmostEqual(float(runtime.predict(30)),.024,places=12)
            before=list(runtime.observed_steps)
            runtime.predict([31,34,38])
            self.assertEqual(runtime.observed_steps,before)
            with self.assertRaises(ValueError): runtime.predict(29)
            with self.assertRaises(ValueError): runtime.observe(29,.01)

    def test_switch_dependency_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            OnlinePolynomial(example_prior(),use_online_fit=False,use_mirror_node=True)

    def test_mirror_is_causal_and_geometry_control_ignores_old_values(self):
        # Source step 10 is a future virtual node after reflection, but is
        # outside the local five-node fit. Geometry control must not read it.
        observations=np.arange(1,31)
        opts=dict(use_online_fit=True,use_mirror_node=True)
        mirror=correction_weights(observations,[31,34,38],26.4,OnlineFitConfig(),**opts)
        geometry=correction_weights(observations,[31,34,38],26.4,OnlineFitConfig(),
                                    mirror_node_mode='anchor',**opts)
        np.testing.assert_array_equal(geometry[:,9],np.zeros(3))
        self.assertGreater(float(np.abs(mirror[:,9]).max()),1e-6)
        self.assertEqual(mirror.shape,(3,30)) # No future observation columns.


if __name__=='__main__':
    unittest.main()
