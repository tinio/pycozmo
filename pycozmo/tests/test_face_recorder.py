import importlib.util
import math
import os
import tempfile
import unittest
from datetime import datetime

from PIL import Image


APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "face_recorder.py")


def load_app():
    spec = importlib.util.spec_from_file_location("face_recorder", APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.path.isfile(APP), "face_recorder.py not in this tree.")
class TestFaceRecorderHelpers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = load_app()

    def test_wrap_angle(self):
        self.assertAlmostEqual(self.app.wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(self.app.wrap_angle(math.pi + 0.1), -math.pi + 0.1, places=6)
        self.assertAlmostEqual(self.app.wrap_angle(-math.pi - 0.1), math.pi - 0.1, places=6)

    def test_bearing_to_face(self):
        ahead = self.app.bearing_to_face(0.0, 0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(ahead, 0.0, places=6)
        left = self.app.bearing_to_face(0.0, 0.0, 0.0, 0.0, 100.0)
        self.assertAlmostEqual(left, math.pi / 2.0, places=6)
        # Robot heading wraps: facing behind the face means turning around.
        behind = self.app.bearing_to_face(0.0, 0.0, math.pi, 100.0, 0.0)
        self.assertAlmostEqual(abs(behind), math.pi, places=6)

    def test_should_scan(self):
        self.assertTrue(self.app.should_scan(10.0, None, 5.0))
        self.assertFalse(self.app.should_scan(10.0, 8.0, 5.0))
        self.assertTrue(self.app.should_scan(10.0, 4.0, 5.0))

    def test_build_output_dir(self):
        when = datetime(2026, 11, 4, 10, 11, 12)
        path = self.app.build_output_dir("recordings", when)
        self.assertEqual(path.name, "face_recorder_20261104_101112")

    def test_parse_args_defaults(self):
        args = self.app.parse_args([])
        self.assertEqual(args.fps, 14.0)
        self.assertFalse(args.no_move)
        self.assertFalse(args.color)

    def test_save_and_list_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = self.app.Path(tmp) / "frames"
            img = Image.new("RGB", (32, 24), (10, 20, 30))
            self.app.save_frame(img, frames_dir / "frame_000000.jpg")
            self.app.save_frame(img, frames_dir / "frame_000001.jpg")
            frames = self.app.list_frames(frames_dir)
            self.assertEqual(len(frames), 2)
            self.assertTrue(frames[0].name < frames[1].name)

    def test_append_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self.app.Path(tmp) / "frames.jsonl"
            self.app.append_record(log, {"frame": 0})
            self.app.append_record(log, {"frame": 1})
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

    def test_assemble_video_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(self.app.assemble_video(self.app.Path(tmp), self.app.Path(tmp) / "video.mp4"))

    def test_assemble_video_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = self.app.Path(tmp) / "frames"
            img = Image.new("RGB", (32, 24), (200, 10, 10))
            for i in range(3):
                self.app.save_frame(img, frames_dir / "frame_{:06d}.jpg".format(i))
            out = self.app.assemble_video(frames_dir, self.app.Path(tmp) / "video.mp4", fps=5.0)
            self.assertIsNotNone(out)
            self.assertTrue(self.app.Path(str(out)).is_file())


if __name__ == "__main__":
    unittest.main()
