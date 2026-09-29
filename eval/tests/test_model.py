from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from eval.artifacts import instantiate_artifact_model, load_artifact
from eval.dataloader import TrajectoryDataModule
from eval.model import CacheLoss, CacheModel, TemporalLoss, TemporalModel
import eval.train as train_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_DIR = PROJECT_ROOT / "eval" / "conf"


def write_trajectory(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": torch.tensor(
                [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
                    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                ],
                dtype=torch.float32,
            ),
            "targets": torch.tensor(
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                dtype=torch.float32,
            ),
            "step_indices": torch.arange(6),
        },
        data_dir / "unit_lazy_data.pt",
    )


def compose_train_config(data_dir: Path, output_dir: Path):
    overrides = [
        f"data.data_dir={data_dir.as_posix()}",
        f"output_dir={output_dir.as_posix()}",
        "device=cpu",
        "model.horizon=2",
        "+model.hidden_dim=8",
        "+model.num_hidden_layers=1",
        "data.batch_size=2",
        "data.val_ratio=0",
        "train.epochs=1",
        "train.patience=0",
    ]
    with initialize_config_dir(
        config_dir=str(TRAIN_CONFIG_DIR),
        version_base="1.3",
    ):
        return compose(config_name="train", overrides=overrides)


def compose_temporal_train_config(data_dir: Path, output_dir: Path):
    overrides = [
        f"data.data_dir={data_dir.as_posix()}",
        f"output_dir={output_dir.as_posix()}",
        "device=cpu",
        "model._target_=eval.model.TemporalModel",
        "+model.history_length=2",
        "+model.hidden_dim=8",
        "+model.head_dim=8",
        "model.horizon=2",
        "data.batch_size=2",
        "data.val_ratio=0",
        "train.epochs=1",
        "train.patience=0",
    ]
    with initialize_config_dir(
        config_dir=str(TRAIN_CONFIG_DIR),
        version_base="1.3",
    ):
        return compose(config_name="train", overrides=overrides)


class HydraModelPipelineTests(unittest.TestCase):
    def test_hydra_instantiates_model_loss_and_cumulative_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            write_trajectory(data_dir)
            cfg = compose_train_config(data_dir, root / "output")

            model = instantiate(cfg.model)
            criterion = instantiate(cfg.loss)
            data_module = instantiate(cfg.data)
            self.assertIsInstance(model, CacheModel)
            self.assertIsInstance(criterion, CacheLoss)
            self.assertIsInstance(data_module, TrajectoryDataModule)

            data = data_module.setup(model.input_dim, model.horizon)
            torch.testing.assert_close(
                data.train_loader.dataset.tensors[1][0],
                torch.tensor([0.01, 0.03]),
            )
            features, target, mask = next(iter(data.val_loader))
            prediction = model(features)
            loss = criterion(prediction, target, mask)
            self.assertEqual(tuple(prediction.shape), tuple(target.shape))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
            )
            self.assertEqual(float(cfg.scheduler.min_lr), 1e-6)

    def test_one_epoch_training_exports_generic_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            output_dir = root / "output"
            write_trajectory(data_dir)
            cfg = compose_train_config(data_dir, output_dir)

            events: list[str] = []
            original_random_seed = train_module.random.seed
            original_torch_seed = train_module.torch.manual_seed
            original_instantiate = train_module.instantiate

            def tracked_random_seed(*args, **kwargs):
                events.append("random_seed")
                return original_random_seed(*args, **kwargs)

            def tracked_torch_seed(*args, **kwargs):
                events.append("torch_seed")
                return original_torch_seed(*args, **kwargs)

            def tracked_instantiate(*args, **kwargs):
                events.append("instantiate")
                return original_instantiate(*args, **kwargs)

            try:
                with (
                    patch.object(
                        train_module.random,
                        "seed",
                        side_effect=tracked_random_seed,
                    ),
                    patch.object(
                        train_module.torch,
                        "manual_seed",
                        side_effect=tracked_torch_seed,
                    ),
                    patch.object(
                        train_module,
                        "instantiate",
                        side_effect=tracked_instantiate,
                    ),
                ):
                    train_module.main.__wrapped__(cfg)
            finally:
                logging.shutdown()

            self.assertEqual(events.count("random_seed"), 1)
            self.assertEqual(events.count("torch_seed"), 1)
            self.assertLess(events.index("random_seed"), events.index("instantiate"))
            self.assertLess(events.index("torch_seed"), events.index("instantiate"))

            artifact_path = output_dir / "model.pth"
            summary_path = output_dir / "summary.json"
            self.assertTrue(artifact_path.is_file())
            self.assertTrue((output_dir / "history.csv").is_file())
            self.assertTrue(summary_path.is_file())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact"], str(artifact_path))
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["trajectory_files"], 1)

            artifact = load_artifact(artifact_path, "cpu")
            restored = instantiate_artifact_model(artifact, "cpu")
            self.assertEqual(tuple(restored(torch.zeros(1, 8)).shape), (1, 2))
            self.assertEqual(artifact["model_config"]["_target_"], "eval.model.CacheModel")

    def test_temporal_model_trains_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            output_dir = root / "output"
            write_trajectory(data_dir)
            cfg = compose_temporal_train_config(data_dir, output_dir)

            model = instantiate(cfg.model)
            criterion = instantiate(cfg.loss)
            data_module = instantiate(cfg.data)
            self.assertIsInstance(model, TemporalModel)
            self.assertIsInstance(criterion, TemporalLoss)
            self.assertIsInstance(data_module, TrajectoryDataModule)
            self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 20_000)

            self.assertEqual(int(cfg.data.history_length), 2)
            data = data_module.setup(model.input_dim, model.horizon)
            normalized = data.train_loader.dataset.tensors[0]
            self.assertEqual(tuple(normalized.shape), (4, 3, 8))
            torch.testing.assert_close(
                normalized.mean(dim=(0, 1)),
                torch.zeros(8),
                atol=1e-5,
                rtol=0,
            )
            torch.testing.assert_close(
                data.train_loader.dataset.tensors[1][0],
                torch.tensor([0.03, 0.07]),
            )
            torch.testing.assert_close(
                data.train_loader.dataset.tensors[1][-1],
                torch.tensor([0.06, 0.06]),
            )
            self.assertEqual(
                data.train_loader.dataset.tensors[2][-1].tolist(),
                [True, False],
            )
            features, target, mask = next(iter(data.train_loader))
            prediction = model(features)
            self.assertEqual(tuple(prediction.shape), tuple(target.shape))
            self.assertTrue(torch.all(prediction[:, 1:] >= prediction[:, :-1]))
            loss = criterion(prediction, target, mask)
            loss.backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
            )

            train_module.main.__wrapped__(cfg)
            artifact = load_artifact(output_dir / "model.pth", "cpu")
            restored = instantiate_artifact_model(artifact, "cpu")
            self.assertEqual(
                tuple(restored(torch.zeros(1, 3, 8)).shape),
                (1, 2),
            )
            self.assertEqual(
                artifact["model_config"]["_target_"],
                "eval.model.TemporalModel",
            )


if __name__ == "__main__":
    unittest.main()
