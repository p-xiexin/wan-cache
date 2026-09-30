from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from eval.artifacts import load_artifact
from eval.predictor_data import RawPredictorDataset, find_raw_trajectories, split_by_prompt
from eval.predictor_parallel import Progress, RankSampler, process_count, search_batch_size
from eval.predictor_training import fit_predictor, offline_epoch
from eval.tests.test_polynomial import write_raw


ROOT = Path(__file__).resolve().parents[1]


class PredictorParallelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self, *overrides):
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
            return compose(config_name="train_predictor", overrides=list(overrides))

    def test_rank_partition_equal_training_steps_and_exact_validation_coverage(self):
        for size, world in [(13, 4), (2, 4), (96, 8)]:
            train = [list(RankSampler(size, rank, world, True)) for rank in range(world)]
            self.assertEqual(len({len(indices) for indices in train}), 1)
            flattened = sum(train, [])
            self.assertEqual(set(flattened), set(range(size)))
            self.assertLess(len(flattened) - size, world)
            validation = sum([list(RankSampler(size, rank, world)) for rank in range(world)], [])
            self.assertEqual(validation, list(range(size)))

    def test_auto_batch_probes_until_budget_or_oom_and_never_swallows_other_errors(self):
        tried = []
        def probe(size):
            tried.append(size)
            if size == 8:
                raise torch.cuda.OutOfMemoryError("simulated GPU memory exhaustion")
            return True
        self.assertEqual(search_batch_size(probe, 32), 4)
        self.assertEqual(tried, [1, 2, 4, 8])
        self.assertEqual(search_batch_size(lambda size: size <= 2, 32), 2)
        self.assertEqual(search_batch_size(lambda size: True, 7), 4)
        with self.assertRaisesRegex(RuntimeError, "batch_size=1"):
            search_batch_size(lambda size: False, 32)
        def broken(size):
            raise ValueError("invalid shape")
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            search_batch_size(broken, 32)

    def test_gpu_process_discovery_respects_visible_devices_and_explicit_device(self):
        cfg = self.config("device=cuda")
        with patch("eval.predictor_parallel.torch.cuda.device_count", return_value=8):
            self.assertEqual(process_count(cfg), 8)
            cfg.device = "cuda:2"
            self.assertEqual(process_count(cfg), 1)
            cfg.parallel.num_processes = 4
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                process_count(cfg)
            cfg.device = "cuda"
            self.assertEqual(process_count(cfg), 4)
            cfg.parallel.num_processes = 9
            with self.assertRaisesRegex(ValueError, "visible CUDA"):
                process_count(cfg)

    def test_slow_loading_heartbeat_reports_current_phase(self):
        with self.assertLogs("eval.predictor_parallel", level="INFO") as logs:
            with Progress("reading first raw batch", interval=.01) as progress:
                progress.update("batch 1/5: reading raw tensors")
                time.sleep(.04)
        self.assertTrue(any("reading first raw batch" in message for message in logs.output))
        self.assertTrue(any("batch 1/5: reading raw tensors" in message for message in logs.output))

    def test_multiple_loader_workers_preserve_complete_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, _, _ = write_raw(root, count=2)
            paths, _ = find_raw_trajectories(root)
            data = RawPredictorDataset(paths, generation, 2)
            samples = []
            for batch in DataLoader(data, batch_size=2, num_workers=2, multiprocessing_context="spawn", prefetch_factor=1):
                samples.extend(zip(batch["step"].tolist(), batch["branch"].tolist(), batch["x"][:,0,0,0,0].tolist()))
            expected = [(s["step"], s["branch"], float(s["x"][0,0,0,0])) for s in data]
            self.assertEqual(samples, expected)

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "requires the CPU DDP backend")
    def test_two_process_training_matches_equivalent_serial_global_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation, _, _ = write_raw(root / "raw", count=4)
            cfg = self.config(
                f"data.data_dir={root / 'raw'}", "data.val_ratio=0.25", "device=cpu",
                "model.channels=2", "model.hidden_dim=4", "parallel.num_processes=2",
                "data.num_workers=0", "train.epochs=1", "train.log_every=20",
            )
            fit_predictor(cfg, root / "out")
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["runtime"]["world_size"], 2)
            self.assertEqual(summary["runtime"]["batch_size_per_device"], 1)
            self.assertEqual(summary["runtime"]["global_batch_size"], 2)
            self.assertEqual(summary["train_samples"], 36)
            self.assertEqual(summary["val_samples"], 12)
            self.assertEqual(len((root / "out/history.csv").read_text().splitlines()), 2)
            saved = load_artifact(root / "out/model.pth")["model_state_dict"]
            self.assertFalse(any(name.startswith("module.") for name in saved))

            # Match the actual global batches formed by the two contiguous ranks.
            paths, _ = find_raw_trajectories(root / "raw")
            train_paths, _ = split_by_prompt(paths, .25, int(cfg.seed))
            data = RawPredictorDataset(train_paths, generation, 2, seed=int(cfg.seed), shuffle=True)
            data.epoch = 1
            indices = sum([list(pair) for pair in zip(RankSampler(len(data),0,2,True),
                                                      RankSampler(len(data),1,2,True))], [])
            torch.manual_seed(int(cfg.seed))
            model, criterion = instantiate(cfg.model), instantiate(cfg.loss)
            optimizer = instantiate(cfg.optimizer, params=model.parameters())
            offline_epoch(model, criterion, DataLoader(data, sampler=indices, batch_size=2),
                          torch.device("cpu"), optimizer, float(cfg.train.grad_clip), primary=False)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.allclose(value, saved[name], atol=2e-6, rtol=1e-5), name)


if __name__ == "__main__":
    unittest.main()
