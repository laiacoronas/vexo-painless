"""Vexo movement features + activity-matched baseline deviation.

Not a trained pain model and not a clinical pain score. Missing sensor data
is represented by None, never by artificial zeros. No external dependencies.
"""
from __future__ import annotations

import math
import statistics as stats
from datetime import datetime, timezone
from typing import Any, Dict, Optional

FEATURES = ("accel_std_g", "gyro_rms_dps", "steps_in_window")
FLOORS = {"accel_std_g": 0.04, "gyro_rms_dps": 15.0, "steps_in_window": 2.0}
# Broad guardrail for a wrist step counter. Not a validation of the device itself.
MAX_STEPS_PER_SECOND = 4.0
VALID_STATES = ("still", "walk", "run")
NOTE = "Movement change is not a validated pain measure. Index is heuristic, not a probability."


def _median(values):
    return float(stats.median(values))


def _rms(values):
    return math.sqrt(sum(float(v) ** 2 for v in values) / len(values)) if values else None


class MovementEngine:
    def __init__(self, rate=25, seconds=5, baseline_windows=4, saved_baseline=None):
        self.rate = int(rate)
        self.seconds = float(seconds)
        self.required_windows = int(baseline_windows)
        if self.rate <= 0 or self.seconds <= 0 or self.required_windows <= 0:
            raise ValueError("rate, seconds and baseline_windows must be positive")
        self.baselines = {state: [] for state in VALID_STATES}
        if isinstance(saved_baseline, dict):
            for state in VALID_STATES:
                source = saved_baseline.get(state, [])
                if not isinstance(source, list):
                    continue
                for item in source[:self.required_windows]:
                    if (isinstance(item, dict) and all(
                        k in item and (item[k] is None or (isinstance(item[k], (int, float)) and
                        math.isfinite(float(item[k])))) for k in FEATURES
                    )):
                        self.baselines[state].append({k: float(item[k]) if item[k] is not None else None for k in FEATURES})
        self.baseline_revision = 0  # grows ONLY when a valid baseline sample is appended
        self.smoothed_index = None
        self.sustained_change_windows = 0
        self.latest_state: Optional[str] = None
        self.latest_steps: Optional[int] = None
        self.steps_at_last_window: Optional[int] = None
        self.latest_cadence: Optional[float] = None
        self.reset_window()

    def reset_window(self, reconnect=False):
        """Discard partial windows. On reconnect invalidate activity/steps references."""
        self.samples = []
        self.states = []
        self.activity_updates = 0
        self.step_updates = 0
        if reconnect:
            self.latest_steps = None
            self.steps_at_last_window = None
            self.latest_state = None
            self.latest_cadence = None
            self.smoothed_index = None
            self.sustained_change_windows = 0

    def motion(self, ax, ay, az, gx, gy, gz):
        try:
            values = tuple(float(v) for v in (ax, ay, az, gx, gy, gz))
        except (TypeError, ValueError):
            return
        if all(math.isfinite(v) for v in values):
            self.samples.append(values)

    def activity(self, steps, cadence, state):
        if steps is not None:
            try:
                next_steps = int(steps)
                if next_steps >= 0:
                    self.latest_steps = next_steps
                    self.step_updates += 1
            except (TypeError, ValueError, OverflowError):
                pass
        if cadence is not None:
            try:
                value = float(cadence)
                if math.isfinite(value):
                    self.latest_cadence = value
            except (TypeError, ValueError):
                pass
        if state in VALID_STATES:
            self.latest_state = state
            self.states.append(state)
            self.activity_updates += 1

    def _state(self):
        if not self.states:
            return None  # never silently reuse stale activity state
        counts = {s: self.states.count(s) for s in VALID_STATES}
        max_count = max(counts.values())
        best = [s for s in VALID_STATES if counts[s] == max_count]
        # In a tie, prefer most recently reported state.
        return next(s for s in reversed(self.states) if s in best)

    def _baseline_status(self, state):
        base = self.baselines.get(state, [])
        return {"state": state, "collected_windows": len(base),
                "required_windows": self.required_windows,
                "ready": len(base) >= self.required_windows}

    def finish_window(self, source="vexo-band") -> Dict[str, Any]:
        samples = self.samples
        state = self._state()
        activity_received = self.activity_updates > 0
        steps_received = self.step_updates > 0
        expected = self.rate * self.seconds
        coverage = len(samples) / expected
        sufficient_motion = len(samples) >= max(5, math.ceil(expected * 0.8))
        # The SDK counter is treated as *cumulative*, but the real semantics must
        # be verified on the actual bracelet. Never trust resets or huge jumps.
        steps = None
        steps_status = "no_update"
        if steps_received and self.latest_steps is not None:
            if self.steps_at_last_window is None:
                steps_status = "initializing"
            elif self.latest_steps < self.steps_at_last_window:
                steps_status = "counter_reset"
            else:
                delta = self.latest_steps - self.steps_at_last_window
                if delta > math.ceil(MAX_STEPS_PER_SECOND * self.seconds + 2):
                    steps_status = "counter_jump"
                else:
                    steps_status = "valid"
                    steps = delta
            self.steps_at_last_window = self.latest_steps
        else:
            self.steps_at_last_window = None

        # A delayed still/walk label can disagree with an active step stream.
        if state == "still" and steps is not None and steps >= 3:
            state = "walk"

        if samples:
            mags = [math.sqrt(ax*ax + ay*ay + az*az) for ax, ay, az, _, _, _ in samples]
            gyros = [math.sqrt(gx*gx + gy*gy + gz*gz) for _, _, _, gx, gy, gz in samples]
            accel_std = round(stats.pstdev(mags), 5) if len(mags) >= 2 else None
            gyro_rms = round(_rms(gyros), 4)
            gyro_max = round(max(gyros), 4)
        else:
            accel_std = gyro_rms = gyro_max = None

        features = {"accel_std_g": accel_std, "gyro_rms_dps": gyro_rms,
                    "gyro_max_dps": gyro_max, "steps_in_window": steps,
                    "cadence_spm": round(self.latest_cadence, 2) if activity_received and self.latest_cadence is not None else None}
        good = sufficient_motion and activity_received and state is not None
        status = "insufficient_data"
        deviation = None
        raw_deviation = None
        if good:
            base = self.baselines[state]
            if len(base) < self.required_windows:
                # Steps may be unavailable; train a baseline from the motion
                # features anyway. A missing counter is NOT a zero step count.
                base.append({k: features[k] for k in FEATURES})
                self.baseline_revision += 1
                status = "calibrating"
                self.smoothed_index = None
                self.sustained_change_windows = 0
            else:
                zscores = []
                for k in FEATURES:
                    if features[k] is None:
                        continue
                    vals = [float(row[k]) for row in base if row.get(k) is not None]
                    if not vals:
                        continue
                    center = _median(vals)
                    mad = _median([abs(x - center) for x in vals])
                    scale = max(1.4826 * mad, FLOORS[k], abs(center) * 0.15)
                    zscores.append(abs(float(features[k]) - center) / scale)
                if zscores:
                    avg_z = sum(zscores) / len(zscores)
                    raw_deviation = round(min(100., 100. * (1. - math.exp(-avg_z / 4.5))), 1)
                    # Smoothing prevents a single wrist gesture from flashing HIGH.
                    self.smoothed_index = (raw_deviation * 0.32 if self.smoothed_index is None
                                           else 0.68 * self.smoothed_index + 0.32 * raw_deviation)
                    deviation = round(self.smoothed_index, 1)
                    self.sustained_change_windows = (self.sustained_change_windows + 1
                                                     if raw_deviation >= 55 else 0)
                    if deviation >= 55 and self.sustained_change_windows >= 3:
                        status = "strong_change"
                    elif deviation >= 35 and self.sustained_change_windows >= 2:
                        status = "change"
                    else:
                        status = "typical"
        # An invalid window does not update the smoothed estimate or baseline.

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source, "connection_status": "connected",
            "window_seconds": self.seconds, "sample_count": len(samples),
            "coverage_fraction": round(min(1., coverage), 3),
            "activity_state": state, "steps_total": self.latest_steps if steps_received else None,
            "features": features, "steps_status": steps_status,
            "movement_deviation_raw": raw_deviation,
            "baseline": self._baseline_status(state),
            "movement_deviation_index": deviation, "movement_status": status,
            "pain_estimate": None, "note": NOTE,
        }
        self.reset_window()
        return result

    def status_result(self, status="disconnected", source="vexo-band"):
        """A fresh JSON record prevents the UI from presenting stale scores as live."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source, "connection_status": status,
            "window_seconds": self.seconds, "sample_count": 0,
            "coverage_fraction": 0., "activity_state": None, "steps_total": None,
            "steps_status": "disconnected", "movement_deviation_raw": None,
            "features": {"accel_std_g": None, "gyro_rms_dps": None,
                         "gyro_max_dps": None, "steps_in_window": None, "cadence_spm": None},
            "baseline": self._baseline_status(None),
            "movement_deviation_index": None, "movement_status": status,
            "pain_estimate": None, "note": NOTE,
        }

    def export_baseline(self):
        return {state: list(self.baselines[state]) for state in VALID_STATES}
