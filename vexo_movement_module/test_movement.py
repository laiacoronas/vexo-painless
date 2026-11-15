"""Standard-library tests; no band or external packages needed."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import movement_live as live
from movement_engine import MovementEngine


def window(engine, state="still", steps=100, rotation=2., movement=0.01, samples=125):
    for k in range(samples):
        wave = math.sin(2 * math.pi * k / 25 * 1.6)
        engine.motion(wave * movement, wave * movement, 1.0 + wave * movement,
                      wave * rotation, wave * rotation / 2, wave * rotation / 4)
    engine.activity(steps, 0 if state == "still" else 95, state)
    return engine.finish_window()


class EngineTests(unittest.TestCase):
    def test_missing_data_not_zero_and_never_calibrates(self):
        e = MovementEngine()
        e.activity(100, 0, "still")
        r = e.finish_window()
        self.assertEqual(r["movement_status"], "insufficient_data")
        self.assertIsNone(r["movement_deviation_index"])
        self.assertIsNone(r["features"]["accel_std_g"])
        self.assertEqual(e.baseline_revision, 0)

    def test_baseline_deviation_and_activity_matching(self):
        e = MovementEngine(rate=25, seconds=5, baseline_windows=4)
        self.assertEqual(window(e, steps=100)["movement_status"], "calibrating")
        for i in range(3):
            self.assertEqual(window(e, steps=100)["movement_status"], "calibrating")
        stable = window(e, steps=100)
        self.assertEqual(stable["movement_status"], "typical")
        self.assertLess(stable["movement_deviation_index"], 5)
        move = window(e, steps=100, rotation=200, movement=.40)
        self.assertEqual(move["movement_status"], "typical")  # one burst must not cause a HIGH alert
        self.assertGreater(move["movement_deviation_raw"], 50)
        self.assertLess(move["movement_deviation_index"], 35)
        # New activity requires its own baseline; do not mix walk and still.
        walk = window(e, state="walk", steps=110, rotation=20, movement=.15)
        self.assertEqual(walk["movement_status"], "calibrating")
        self.assertEqual(walk["baseline"]["collected_windows"], 1)
        self.assertEqual(e.baseline_revision, 5)

    def test_step_counter_guard_does_not_inflate(self):
        e = MovementEngine(rate=25,seconds=5,baseline_windows=1)
        window(e,steps=500)
        normal = window(e,steps=507)
        self.assertEqual(normal["features"]["steps_in_window"], 7)
        jump = window(e,steps=999)
        self.assertIsNone(jump["features"]["steps_in_window"])
        self.assertEqual(jump["steps_status"], "counter_jump")
        resync = window(e,steps=1004)
        self.assertEqual(resync["features"]["steps_in_window"], 5)
        reset = window(e,steps=2)
        self.assertEqual(reset["steps_status"], "counter_reset")
        self.assertIsNone(reset["features"]["steps_in_window"])

    def test_sustained_motion_but_not_one_gesture(self):
        e=MovementEngine(rate=25,seconds=5,baseline_windows=3)
        for k in range(3):
            self.assertEqual(window(e,steps=100)["movement_status"],"calibrating")
        self.assertEqual(window(e,steps=100)["movement_status"],"typical")
        burst=window(e,steps=100,rotation=200,movement=.35)
        self.assertEqual(burst["movement_status"],"typical")
        self.assertLess(burst["movement_deviation_index"], 40)
        states=[]
        for _ in range(8):
            states.append(window(e,steps=100,rotation=200,movement=.35)["movement_status"])
        self.assertIn("strong_change",states)

    def test_reset_on_reconnect_forgets_counter_delta_and_partial_samples(self):
        e = MovementEngine(rate=25, seconds=5)
        window(e, steps=100)
        window(e, steps=110)
        e.motion(1, 0, 0, 0, 0, 0)
        e.reset_window(reconnect=True)
        r = window(e, steps=2)
        self.assertEqual(r["sample_count"], 125)
        self.assertIsNone(r["features"]["steps_in_window"])
        self.assertEqual(r["steps_status"], "initializing")
        r2 = window(e, steps=10)
        self.assertEqual(r2["features"]["steps_in_window"], 8)


class IOTests(unittest.TestCase):
    def test_permission_error_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "latest.json"
            with patch.object(live.os, "replace", side_effect=PermissionError("File locked")):
                self.assertFalse(live.save_json(p, {"x": 1}, attempts=2))
            self.assertFalse(list(p.parent.glob("*.tmp")))
            self.assertTrue(live.save_json(p, {"x": 2}))
            self.assertEqual(json.loads(p.read_text())["x"], 2)

    def test_publish_preserves_output_keys_and_writes_baseline_once(self):
        e = MovementEngine(rate=25, seconds=5, baseline_windows=1)
        window(e, steps=100)
        with tempfile.TemporaryDirectory() as d:
            with (patch.object(live, "LATEST", Path(d) / "latest.json"),
                  patch.object(live, "BASELINE", Path(d) / "baseline.json"),
                  patch.object(live, "HISTORY", Path(d) / "history.csv"),
                  patch.object(live, "DATA", Path(d))):
                res = window(e, steps=100)
                last = live.publish(e, res, 0)
                self.assertEqual(last, 1)
                before = (Path(d) / "baseline.json").stat().st_mtime_ns
                res2 = window(e, steps=100)
                last = live.publish(e, res2, last)
                self.assertEqual(last, 1)
                self.assertEqual((Path(d) / "baseline.json").stat().st_mtime_ns, before)
                j = json.loads((Path(d) / "latest.json").read_text())
                self.assertIsNotNone(j["movement_deviation_index"])
                self.assertEqual(j["connection_status"], "connected")
                self.assertIn("gyro_rms_dps", j["features"])

    def test_run_session_identifies_early_disconnect(self):
        class Band:
            name = "mock-band"
            def on(self, *args): pass
            def configure(self, **kwargs): pass
            def wait(self, seconds): return  # connection vanished; should not emit data
        e = MovementEngine()
        with self.assertRaises(ConnectionError):
            live.run_session(e, Band(), "mock", 0)
        self.assertEqual(e.baseline_revision, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
