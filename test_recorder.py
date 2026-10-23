"""Recorder protocol tests with a fake audio driver; never opens a microphone."""
import ast
import select
import subprocess
import types
import os
import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


DRIVER = '''
import os, time
from pathlib import Path
class PortAudioError(Exception):
    pass

def event(text):
    with open(os.environ["RECORDER_EVENTS"], "a") as f:
        f.write(text + "\\n")

def query_devices():
    return [{"name": "Test mic", "max_input_channels": 1}]

attempt = 0
class RawInputStream:
    def __init__(self, **kwargs):
        global attempt
        attempt += 1
        self.callback = kwargs["callback"]
        event("open")
        mode = os.environ.get("RECORDER_MODE", "")
        if mode == "retry" and attempt < 3:
            raise PortAudioError("temporary", -9986)
        if mode == "permanent":
            raise PortAudioError("bad sample rate", -9997)
        if mode == "exhausted":
            raise PortAudioError("temporary", -9986)
        if mode == "hang-open":
            time.sleep(30)
    def start(self):
        event("start")
        if os.environ.get("RECORDER_MODE") == "retry-start" and attempt == 1:
            raise PortAudioError("temporary", -9986)
        self.callback(b"\\x00\\x00\\x00\\x3f", None, None, None)
    def stop(self):
        event("stop")
        if os.environ.get("RECORDER_MODE") == "hang-stop":
            time.sleep(30)
    def close(self):
        event("close")
'''


class RecorderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the actual subprocess code without loading AppKit or speech models.
        tree = ast.parse(Path(__file__).with_name("dictate.py").read_text())
        nodes = [node for node in tree.body
                 if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RECORDER" for t in node.targets))
                 or (isinstance(node, ast.FunctionDef) and node.name in {"spawn_recorder", "open_recorder", "close_recorder"})]
        cls.app = types.SimpleNamespace(os=os, sys=sys, subprocess=subprocess, select=select,
                                        tempfile=tempfile, Path=Path, RATE=16000)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "dictate.py", "exec"), vars(cls.app))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "sounddevice.py").write_text(DRIVER)
        self.events = self.root / "events"
        env = patch.dict(os.environ, PYTHONPATH=str(self.root), RECORDER_EVENTS=str(self.events), RECORDER_MODE="")
        env.start()
        self.addCleanup(env.stop)

    def spawn(self, mode=""):
        os.environ["RECORDER_MODE"] = mode
        process, path = self.app.spawn_recorder()
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        self.addCleanup(self.app.close_recorder, process)
        return process, path

    def test_repeated_recordings_acknowledge_audio_and_close_driver(self):
        for _ in range(5):
            process, path = self.spawn()
            self.app.open_recorder(process, "Test mic")
            self.assertEqual(Path(path).read_bytes(), b"\x00\x00\x00\x3f")
            self.app.close_recorder(process)
            self.assertEqual(process.returncode, 0)
        self.assertEqual(self.events.read_text().splitlines(), ["open", "start", "stop", "close"] * 5)

    def test_transient_open_errors_recover_in_same_attempt(self):
        process, path = self.spawn("retry")
        self.app.open_recorder(process, "Test mic")
        self.assertEqual(Path(path).stat().st_size, 4)
        self.assertEqual(self.events.read_text().splitlines(), ["open", "open", "open", "start"])

    def test_start_error_closes_stream_before_retry(self):
        process, _ = self.spawn("retry-start")
        self.app.open_recorder(process, "Test mic")
        self.assertEqual(self.events.read_text().splitlines(), ["open", "start", "close", "open", "start"])

    def test_failed_open_never_acknowledges_recording(self):
        for mode, attempts in [("permanent", 1), ("exhausted", 3)]:
            with self.subTest(mode=mode):
                self.events.write_text("")
                process, path = self.spawn(mode)
                with self.assertRaisesRegex(RuntimeError, "microphone"):
                    self.app.open_recorder(process, "Test mic")
                self.app.close_recorder(process)
                self.assertEqual(Path(path).stat().st_size, 0)
                self.assertEqual(self.events.read_text().splitlines(), ["open"] * attempts)

    def test_stuck_driver_close_is_killed_and_reaped(self):
        process, _ = self.spawn("hang-stop")
        self.app.open_recorder(process, "Test mic")
        start = time.monotonic()
        self.app.close_recorder(process)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertLess(process.returncode, 0)

    def test_stuck_open_has_bounded_wait(self):
        process, _ = self.spawn("hang-open")
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "microphone"):
            self.app.open_recorder(process, "Test mic", timeout=0.15)
        self.app.close_recorder(process)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertIsNotNone(process.returncode)

    def test_missing_selected_device_uses_default_and_keeps_protocol(self):
        process, _ = self.spawn()
        self.app.open_recorder(process, "Missing mic")
        self.assertIsNone(process.poll())



if __name__ == "__main__":
    unittest.main()
