"""Contract tests that do not pretend to execute the external RL libraries."""

import contextlib
import csv
import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from phy_rl.results import Metrics, create_run_directory
from phy_rl.train import Config, initialize_jax, parse_args, ppo_settings, run


class ConfigurationTests(unittest.TestCase):
    def test_invalid_training_configuration_fails_before_execution(self):
        for change in (
            {"seed": -1},
            {"seed": 2**32},
            {"total_steps": 0},
            {"num_envs": 0},
            {"num_envs": 3},
            {"num_evals": 1},
            {"learning_rate": math.nan},
            {"learning_rate": 0},
            {"entropy_cost": math.inf},
            {"reward_scaling": 0},
            {"discounting": 1.1},
            {"task": "../outside"},
            {"backend": "tpu"},
            {"checkpoint_interval_steps": 0},
            {"checkpoint_interval_steps": -1},
            {"checkpoint_interval_steps": 1.5},
            {"checkpoint_interval_steps": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(Config(), **change).validate()

    def test_cli_overrides_reach_training_settings(self):
        config = parse_args(["--seed", "22", "--learning-rate", "0.001", "--total-steps", "2048"])
        settings = ppo_settings(config)
        self.assertEqual(settings["seed"], 22)
        self.assertEqual(settings["learning_rate"], 0.001)
        self.assertEqual(settings["num_timesteps"], 2048)
        self.assertEqual(settings["checkpoint_interval_steps"], 1)
        self.assertEqual(
            parse_args(["--checkpoint-interval-steps", "5"]).checkpoint_interval_steps, 5
        )

    def test_invalid_cli_has_actionable_error(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            parse_args(["--num-envs", "3"])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("divisible", stderr.getvalue())

    def test_checkpoint_saving_can_be_disabled_without_changing_training(self):
        from phy_rl.train import training_config

        default = parse_args([])
        disabled = parse_args(["--no-checkpoints"])
        self.assertTrue(default.checkpoints)
        self.assertFalse(disabled.checkpoints)
        self.assertEqual(training_config(default), training_config(disabled))

    def test_cpu_is_default_and_device_flags_are_mutually_exclusive(self):
        self.assertEqual(parse_args([]).backend, "cpu")
        self.assertEqual(parse_args(["--cpu"]).backend, "cpu")
        self.assertEqual(parse_args(["--gpu"]).backend, "gpu")
        for arguments in (["--cpu", "--gpu"], ["--gpu", "--cpu"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments)


class BackendTests(unittest.TestCase):
    def test_selection_overrides_environment_and_previously_imported_jax_configuration(self):
        for backend, expected_platform in (("cpu", "cpu"), ("gpu", "cuda")):
            with self.subTest(backend=backend):
                jax = SimpleNamespace(
                    config=mock.Mock(), default_backend=mock.Mock(return_value=backend)
                )
                with (
                    mock.patch.dict(sys.modules, {"jax": jax}),
                    mock.patch.dict(os.environ, {"JAX_PLATFORMS": "unavailable"}),
                ):
                    self.assertIs(initialize_jax(backend), jax)
                    self.assertEqual(os.environ["JAX_PLATFORMS"], expected_platform)
                    jax.config.update.assert_called_once_with("jax_platforms", expected_platform)

    def test_gpu_failure_and_already_initialized_cpu_do_not_fall_back(self):
        for error in (RuntimeError("CUDA unavailable"), AssertionError()):
            jax = SimpleNamespace(config=mock.Mock(), default_backend=mock.Mock(side_effect=error))
            with (
                mock.patch.dict(sys.modules, {"jax": jax}),
                mock.patch.dict(os.environ),
                self.assertRaisesRegex(RuntimeError, "NVIDIA GPU and CUDA-enabled JAX"),
            ):
                initialize_jax("gpu")
        jax = SimpleNamespace(config=mock.Mock(), default_backend=lambda: "cpu")
        with (
            mock.patch.dict(sys.modules, {"jax": jax}),
            mock.patch.dict(os.environ),
            self.assertRaisesRegex(RuntimeError, "already using cpu"),
        ):
            initialize_jax("gpu")


class ArtifactTests(unittest.TestCase):
    def test_repeated_runs_do_not_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            first = create_run_directory(Path(root), "CartpoleBalance", 11)
            (first / "metrics.jsonl").write_text("existing data\n")
            second = create_run_directory(Path(root), "CartpoleBalance", 11)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "metrics.jsonl").read_text(), "existing data\n")

    def test_metrics_survive_without_a_clean_shutdown_and_export_sparse_columns(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            metrics = Metrics(path)
            metrics.record(0, {"eval/episode_reward": 2})
            metrics.record(100, {"training/total_loss": 0.25})
            # Files are readable immediately, without a finalization/close method.
            records = [
                json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["step"] for row in records], [0, 100])
            metrics.export_csv()
            with (path / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["training/total_loss"], "")
            self.assertEqual(float(rows[1]["training/total_loss"]), 0.25)

    def test_invalid_metrics_do_not_damage_prior_observations(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            metrics = Metrics(path)
            metrics.record(10, {"reward": 1})
            for step, values in [(9, {"reward": 2}), (20, {"loss": math.nan})]:
                with self.assertRaises(ValueError):
                    metrics.record(step, values)
            self.assertEqual(len((path / "metrics.jsonl").read_text().splitlines()), 1)


class RunLifecycleTests(unittest.TestCase):
    def test_failed_and_interrupted_runs_retain_metrics_and_diagnostics(self):
        for error, expected_status in [
            (RuntimeError("test worker failed"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
        ]:
            with self.subTest(status=expected_status), tempfile.TemporaryDirectory() as root:

                def fake_training(config, directory, metrics, tracker=None, error=error):
                    metrics.record(64, {"training/loss": 0.4})
                    raise error

                with (
                    mock.patch("phy_rl.train.train", side_effect=fake_training),
                    self.assertRaises(type(error)),
                ):
                    run(dataclasses.replace(Config(), output=Path(root)))
                (directory,) = Path(root).iterdir()
                self.assertEqual(
                    json.loads((directory / "status.json").read_text())["status"], expected_status
                )
                self.assertTrue((directory / "metrics.csv").exists())
                self.assertTrue((directory / "error.txt").exists())
                self.assertFalse((directory / "summary.json").exists())

    def test_success_records_results_from_training(self):
        with tempfile.TemporaryDirectory() as root:

            def fake_training(config, directory, metrics, tracker=None):
                metrics.record(128, {"eval/episode_reward": 3})
                return {
                    "evaluation_before": {"mean_return": 1},
                    "evaluation_after": {"mean_return": 3},
                }

            with (
                mock.patch("phy_rl.train.train", side_effect=fake_training),
                mock.patch.object(Metrics, "plot"),
            ):
                directory = run(dataclasses.replace(Config(), output=Path(root)))
            self.assertEqual(
                json.loads((directory / "status.json").read_text())["status"], "completed"
            )
            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {
                    "requested_config.json",
                    "status.json",
                    "metrics.jsonl",
                    "metrics.csv",
                    "summary.json",
                },
            )


if __name__ == "__main__":
    unittest.main()
