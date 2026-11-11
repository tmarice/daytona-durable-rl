"""The training work directory is importable after the runner uploads its contents."""

import io
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from daytona_runner import DaytonaRunner


class WorkDirectoryTests(unittest.TestCase):
    def test_uploaded_workdir_contains_entrypoint_and_session_without_parent_sources(self) -> None:
        runner = object.__new__(DaytonaRunner)
        runner._workdir = Path(__file__).resolve().parents[1] / "phy_rl"
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with tarfile.open(fileobj=io.BytesIO(runner._package_source()), mode="r:gz") as archive:
                archive.extractall(directory, filter="data")
            self.assertTrue((directory / "train.py").is_file())
            self.assertTrue((directory / "daytona_experiment_session.py").is_file())
            self.assertTrue((directory / "pyproject.toml").is_file())
            self.assertFalse((directory / "phy_rl").exists())
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-c",
                    (
                        "import inspect; import train; import sweep; import video; "
                        "from daytona_experiment_session import ExperimentSession; "
                        "inspect.signature(train.run).bind({}, experiment_id='job', "
                        "attempt_id='attempt', checkpoint_dir='checkpoints')"
                    ),
                ],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
