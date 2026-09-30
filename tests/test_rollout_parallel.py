from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eval.model.polynomial import PolynomialLoss, PolynomialPredictor
from eval.predictor_data import find_raw_trajectories
from eval.predictor_parallel import Process, RolloutGradients, process_count
from eval.predictor_rollout import empty_rollout, rollout
from eval.predictor_training import closed_loop_epoch
from eval.tests.test_polynomial import EulerScheduler, synthetic_teacher, write_raw


ROOT = Path(__file__).resolve().parents[1]


def initialized_model():
    torch.manual_seed(17)
    model = PolynomialPredictor(2, 4)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.atanh(torch.tensor([1., 0., 0.]) / model.a_max))
    return model


def distributed_rollout_worker(rank, directory, config):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2)
    try:
        process = Process(rank, 2, torch.device("cpu"))
        model = initialized_model()
        process.broadcast_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        gradients = RolloutGradients(model, optimizer, process)
        # Rank 0 predicts, rank 1 refreshes every node. Both must update weights.
        values, cache, final = rollout(
            model, PolynomialLoss(), synthetic_teacher, EulerScheduler(), torch.ones(2,2,3,3),
            cache_threshold=100 if rank == 0 else 0, warmup_steps=3, final_full_steps=1,
            guide_scale=2, window_steps=4, gradients=gradients,
        )
        # An incomplete last global batch must not duplicate a trajectory.
        if rank == 0:
            rollout(model, PolynomialLoss(), synthetic_teacher, EulerScheduler(), torch.ones(2,2,3,3),
                    cache_threshold=100, warmup_steps=3, final_full_steps=1, guide_scale=2,
                    window_steps=4, gradients=gradients)
        else:
            empty_rollout(9, 3, 4, gradients)
        torch.save({
            "state": model.state_dict(), "optimizer": optimizer.state_dict(),
            "samples": values["samples"], "cache": cache, "optimizer_steps": gradients.optimizer_steps,
            "final_requires_grad": final.requires_grad,
        }, root / f"rank{rank}.pt")

        # Exercise the actual epoch coordinator: uneven training trajectories
        # and one validation trajectory, leaving rank 1 empty in validation.
        cfg = OmegaConf.create(config)
        paths, generation = find_raw_trajectories(root / "raw")
        runtime = SimpleNamespace(pipeline=SimpleNamespace(num_train_timesteps=1000))
        model = initialized_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        seen_train, seen_validation = [], []
        from eval.predictor_validation import run_trajectory as original_runner

        def record_runner(*args, **kwargs):
            selected = seen_validation if kwargs.get("measure_drift") else seen_train
            selected.append(args[3].name)
            return original_runner(*args, **kwargs)

        with (
            patch("eval.predictor_validation.make_wan_scheduler", side_effect=lambda *args: EulerScheduler()),
            patch("eval.predictor_validation.WanTeacher", return_value=synthetic_teacher),
            patch("eval.predictor_validation.run_trajectory", side_effect=record_runner),
        ):
            train_totals, train_metrics = closed_loop_epoch(
                cfg, generation, runtime, paths, model, PolynomialLoss(), process, optimizer,
            )
            val_totals, val_metrics = closed_loop_epoch(
                cfg, generation, runtime, paths[:1], model, PolynomialLoss(), process,
            )
        torch.save(model.state_dict(), root / f"epoch_rank{rank}.pt")
        (root / f"epoch_rank{rank}.json").write_text(json.dumps({
            "train_totals": train_totals, "train_metrics": train_metrics,
            "val_totals": val_totals, "val_metrics": val_metrics,
            "train_paths": seen_train, "val_paths": seen_validation,
        }))
    finally:
        dist.destroy_process_group()


class RolloutParallelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self):
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
            return compose(config_name="train_predictor", overrides=[
                "train.stage=rollout", "cache.warmup_steps=3", "cache.threshold=100",
                "train.rollout_steps=4", "data.data_dir=unused",
            ])

    def test_rollout_uses_all_visible_gpus(self):
        cfg = self.config()
        with patch("eval.predictor_parallel.torch.cuda.device_count", return_value=4):
            self.assertEqual(process_count(cfg), 4)
            cfg.parallel.num_processes = 2
            self.assertEqual(process_count(cfg), 2)

    def test_gradients_average_prediction_nodes_instead_of_segment_means(self):
        model = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        gradients = RolloutGradients(model, optimizer, Process(0,1,torch.device("cpu")), grad_clip=0)
        gradients.backward([3 * model.weight.sum()])
        gradients.backward([model.weight.sum() for _ in range(3)])
        self.assertEqual(gradients.step(), 4)
        self.assertAlmostEqual(model.weight.item(), .85, places=6)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(gradients.step(), 0)
        self.assertEqual(gradients.optimizer_steps, 1)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_divergent_schedules_empty_ranks_and_uneven_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            write_raw(root / "raw", count=3, replayable=True)
            context = mp.spawn(distributed_rollout_worker,
                               args=(str(root), OmegaConf.to_container(self.config(), resolve=True)),
                               nprocs=2, join=False)
            deadline = time.monotonic() + 45
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    for worker in context.processes:
                        worker.terminate()
                    for worker in context.processes:
                        worker.join()
                    self.fail("distributed rollout did not finish: possible collective deadlock")
            first, second = [torch.load(root / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
            self.assertEqual(first["samples"], 10)
            self.assertEqual(second["samples"], 0)
            self.assertEqual(first["cache"]["calculated_pair_indices"], [0,1,2,8])
            self.assertEqual(second["cache"]["calculated_pair_indices"], list(range(9)))
            self.assertEqual(first["optimizer_steps"], 4)
            self.assertEqual(second["optimizer_steps"], 4)
            self.assertFalse(first["final_requires_grad"])
            for name, value in first["state"].items():
                self.assertTrue(torch.equal(value, second["state"][name]), name)
            for key, state in first["optimizer"]["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, second["optimizer"]["state"][key][name]))
            # The idle/always-full rank contributes zero, so this is exactly the
            # serial update of the two trajectories handled by rank 0.
            model = initialized_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            gradients = RolloutGradients(model, optimizer, Process(0,1,torch.device("cpu")))
            for _ in range(2):
                rollout(model, PolynomialLoss(), synthetic_teacher, EulerScheduler(), torch.ones(2,2,3,3),
                        cache_threshold=100, warmup_steps=3, final_full_steps=1, guide_scale=2,
                        window_steps=4, gradients=gradients)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.allclose(value, first["state"][name], atol=1e-6), name)

            reports = [json.loads((root / f"epoch_rank{rank}.json").read_text()) for rank in range(2)]
            training_paths = reports[0]["train_paths"] + reports[1]["train_paths"]
            self.assertEqual(len(training_paths), 3)
            self.assertEqual(len(set(training_paths)), 3)
            self.assertEqual(len(reports[0]["val_paths"]) + len(reports[1]["val_paths"]), 1)
            for key in ("train_totals", "train_metrics", "val_totals", "val_metrics"):
                self.assertEqual(reports[0][key], reports[1][key])
            self.assertEqual(reports[0]["train_totals"]["samples"], 30)
            self.assertEqual(reports[0]["val_totals"]["samples"], 10)
            states = [torch.load(root / f"epoch_rank{rank}.pt", weights_only=True) for rank in range(2)]
            self.assertTrue(all(torch.equal(states[0][key], states[1][key]) for key in states[0]))


if __name__ == "__main__":
    unittest.main()
