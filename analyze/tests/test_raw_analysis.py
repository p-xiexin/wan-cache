"""Portable numerical and I/O checks for offline analysis (no Wan or real data)."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_analysis import SCRIPT_DIR, find_trajectories, load_trajectory, resolve_path
from analyze_raw_icc import channel_statistics
from analyze_raw_svd import fit_basis, forecast_errors


def make_raw(root):
    trajectory = root / "trajectories" / "trajectory_00000000"
    trajectory.mkdir(parents=True)
    records = []
    generator = torch.Generator().manual_seed(21)
    base = torch.randn(4, 2, 3, 4, generator=generator)
    direction = torch.randn(4, 2, 3, 4, generator=generator)
    for step, timestep in enumerate([1000., 920., 820., 710., 600., 470., 310., 120.]):
        for branch in ("conditional", "unconditional"):
            x = base * (timestep / 1000)
            v = x + direction * (1 + 0.1 * step) + (0.02 if branch == "conditional" else -0.02)
            records.append(dict(step_index=step, branch=branch, timestep=timestep,
                                model_input=(x,), model_output=(v,)))
    shards = ["shard_0000.pt", "shard_0001.pt"]
    torch.save(records[:8], trajectory / shards[0])
    torch.save(records[8:], trajectory / shards[1])
    (trajectory / "metadata.json").write_text(json.dumps({"shards": shards}))
    (trajectory / "_SUCCESS").write_text("complete\n")
    return trajectory


class RawAnalysisTest(unittest.TestCase):
    def test_paths_layouts_sampling_and_residual(self):
        with tempfile.TemporaryDirectory(prefix="raw analysis ") as temp:
            root = Path(temp)
            trajectory = make_raw(root)
            for entry in (root, root / "trajectories", trajectory):
                self.assertEqual(find_trajectories(entry), [trajectory.resolve()])
            with patch.dict(os.environ, {"RAW_TEST_DIRECTORY": str(root)}):
                self.assertEqual(resolve_path("${RAW_TEST_DIRECTORY}"), root.resolve())
            self.assertEqual(resolve_path("outputs/test"), SCRIPT_DIR / "outputs/test")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    resolve_path("$UNSET_RAW_TEST_DIRECTORY/data")
            values = load_trajectory(trajectory, "residual", 11, 0)
            self.assertEqual(values[0].values.shape, (8, 11, 4))
            np.testing.assert_array_equal(values[0].token_indices, values[1].token_indices)
            original = torch.load(trajectory / "shard_0000.pt", weights_only=True)[0]
            expected = (original["model_output"][0] - original["model_input"][0]).reshape(4, -1).T.numpy()
            np.testing.assert_allclose(values[0].values[0], expected[values[0].token_indices])
            (trajectory / "shard_0001.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Shard files"):
                load_trajectory(trajectory, "residual", 11, 0)

    def test_known_channel_relationships(self):
        pattern = np.arange(1, 7, dtype=float)
        values = np.arange(1, 5)[:, None, None] * pattern[None, :, None] * np.array([1., 2., -3.])
        stats = channel_statistics(values, np.array([10., 8., 5., 1.]))
        np.testing.assert_allclose(stats["delta_energy_share"], np.array([1, 4, 9]) / 14)
        np.testing.assert_allclose(stats["delta_correlation"], [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
        np.testing.assert_allclose(stats["delta_direction_cosine"], 1)
        np.testing.assert_allclose(stats["slope_rms"], stats["delta_rms"] / np.array([2, 3, 4])[:, None])

    def test_causal_basis_and_nonuniform_forecasts(self):
        times = np.array([10., 9., 7., 4., 1., 0.])
        pattern = np.arange(1, 6, dtype=float)[:, None] * np.array([[1., 2., -1.]])
        values = (times[:, None, None] + 2) * pattern
        basis, _, _ = fit_basis(values, 3, 0.9)
        self.assertEqual(basis.shape[1], 1)
        changed = values.copy()
        changed[3:] = np.random.default_rng(0).normal(size=changed[3:].shape) * 100
        future_basis, _, _ = fit_basis(changed, 3, 0.9)
        np.testing.assert_allclose(basis @ basis.T, future_basis @ future_basis.T)
        principal = (values @ basis) @ basis.T
        rows = forecast_errors(values, principal, times, 3, 0.9, (1, 2))
        for row in rows:
            if row["method"] in ("full_linear", "principal_linear_reuse_tail"):
                self.assertLess(row["mse"], 1e-20)
        # Changing an unobserved intermediate step cannot affect a horizon-2 prediction.
        changed = values.copy()
        changed[3] += 500
        other = forecast_errors(changed, (changed @ basis) @ basis.T, times, 3, 0.9, (2,))
        original = [r for r in rows if r["anchor_step"] == 2 and r["horizon"] == 2]
        altered = [r for r in other if r["anchor_step"] == 2]
        for a, b in zip(original, altered):
            self.assertAlmostEqual(a["mse"], b["mse"])
        with self.assertRaisesRegex(ValueError, "zero energy"):
            fit_basis(np.zeros_like(values), 3, 0.9)

    def test_scripts_save_figures_and_results_from_another_cwd(self):
        with tempfile.TemporaryDirectory(prefix="raw end to end ") as temp:
            root = Path(temp)
            make_raw(root)
            output = root / "analysis output"
            env = dict(os.environ, RAW_TRAJ_ROOT=str(root), RAW_ANALYSIS_OUTPUT=str(output))
            for script, method, figure in (("analyze_raw_icc.py", "icc", "channels.png"),
                                          ("analyze_raw_svd.py", "svd", "subspaces.png")):
                subprocess.run([sys.executable, str(SCRIPT_DIR / script)], cwd=root,
                               env=env, check=True, capture_output=True, text=True)
                index = json.loads((output / method / "residual" / "index.json").read_text())
                self.assertEqual(len(index["outputs"]), 2)
                for directory in index["outputs"]:
                    folder = Path(directory)
                    self.assertGreater((folder / figure).stat().st_size, 1000)
                    self.assertTrue((folder / "statistics.npz").is_file())
                    info = json.loads((folder / "summary.json").read_text())
                    self.assertEqual(info["source_shape_CFHW"], [4, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
