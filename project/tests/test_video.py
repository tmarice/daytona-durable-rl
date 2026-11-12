"""Native rendering and video output; opt in with RUN_MUJOCO_RENDER=1."""

import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from phy_rl.video import check_rendering, configure_rendering, write_comparison


class RenderingSetupTests(unittest.TestCase):
    def test_headless_default_preserves_an_explicit_backend(self) -> None:
        with patch("phy_rl.video.sys.platform", "linux"), patch.dict(os.environ, {}, clear=True):
            configure_rendering()
            self.assertEqual(os.environ["MUJOCO_GL"], "osmesa")
            os.environ["MUJOCO_GL"] = "egl"
            configure_rendering()
            self.assertEqual(os.environ["MUJOCO_GL"], "egl")
        with patch("phy_rl.video.sys.platform", "darwin"), patch.dict(os.environ, {}, clear=True):
            configure_rendering()
            self.assertNotIn("MUJOCO_GL", os.environ)


@unittest.skipUnless(os.environ.get("RUN_MUJOCO_RENDER") == "1", "Set RUN_MUJOCO_RENDER=1")
class NativeVideoTests(unittest.TestCase):
    def test_native_video_replays_positions_and_targets_without_simulation_steps(self) -> None:
        configure_rendering()
        import mujoco
        import numpy as np
        from PIL import Image

        model = mujoco.MjModel.from_xml_string("""
            <mujoco>
              <worldbody>
                <light pos="0 -1 3"/>
                <camera name="fixed" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
                <geom type="plane" size="3 3 .1" rgba=".8 .8 .8 1"/>
                <body pos="0 0 .5">
                  <joint type="slide" axis="1 0 0"/>
                  <geom type="sphere" size=".25" rgba=".2 .6 .8 1"/>
                </body>
                <body mocap="true" pos=".5 0 1">
                  <geom type="box" size=".12 .12 .12" rgba=".9 .3 .1 1"/>
                </body>
              </worldbody>
            </mujoco>
        """)
        env = SimpleNamespace(mj_model=model, dt=0.02)
        config = SimpleNamespace(
            task="CartpoleBalance", episode_length=10, eval_episodes=1, seed=11, resume=None
        )
        before = {
            "qpos": np.linspace(-0.5, 0.5, 11).reshape(1, 11, 1),
            "mocap_pos": np.tile([[[[0.5, 0, 1]]]], (1, 11, 1, 1)),
            "mocap_quat": np.tile([[[[1, 0, 0, 0]]]], (1, 11, 1, 1)),
            "returns": np.arange(11).reshape(1, 11),
            "length": np.array([10]),
        }
        after = {key: value.copy() for key, value in before.items()}
        after["qpos"] *= -1
        after["mocap_pos"][..., 0] = -0.5
        poses = []
        original_forward = mujoco.mj_forward

        def forward(model, data):
            poses.append((data.qpos.copy(), data.mocap_pos.copy()))
            original_forward(model, data)

        check_rendering(env, config.task)
        with TemporaryDirectory() as root:
            directory = Path(root)
            with (
                patch.object(mujoco, "mj_forward", side_effect=forward),
                patch.object(mujoco, "mj_step", side_effect=AssertionError("Do not resimulate")),
            ):
                metadata = write_comparison(env, before, after, config, directory)
            self.assertEqual(metadata["renderer"], "mujoco_native")
            self.assertEqual(metadata["camera"], "fixed")
            self.assertEqual(metadata["episode_indices"], [0])
            self.assertEqual(metadata["episode_returns_after"], [10])
            for index, step in enumerate(range(0, 11, 2)):
                for side, rollout in enumerate((before, after)):
                    qpos, mocap = poses[index * 2 + side]
                    np.testing.assert_array_equal(qpos, rollout["qpos"][0, step])
                    np.testing.assert_array_equal(mocap, rollout["mocap_pos"][0, step])
            with Image.open(directory / "policy_comparison.png") as preview:
                self.assertEqual(preview.size, (1280, 600))
                self.assertGreater(np.asarray(preview).std(), 20)
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,nb_frames",
                    "-of",
                    "json",
                    str(directory / "policy_comparison.mp4"),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual((stream["width"], stream["height"]), (1280, 600))
            self.assertEqual(int(stream["nb_frames"]), metadata["frame_count"])
            self.assertEqual(json.loads((directory / "video.json").read_text()), metadata)
