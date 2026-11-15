"""Resilient Vexo movement streamer; exports JSON/CSV for an independent UI.

Put movement_live.py and movement_engine.py together in vexo_movement_module/.
Expected SDK folder: vexo-pain-monitor/vexo-open-kit/python (or sibling).
Ctrl+C stops; Bluetooth reconnection uses exponential backoff.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from movement_engine import MovementEngine

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LATEST = DATA / "movement_latest.json"
HISTORY = DATA / "movement_windows.csv"
BASELINE = DATA / "movement_baseline.json"
SIGNAL = DATA / "movement_signal.json"  # fast, decimated IMU preview; not a pain score

# Retain previous JSON keys: downstream Streamlit code can still read them.
CSV_KEYS = ("timestamp", "source", "activity_state", "movement_status",
            "movement_deviation_index", "baseline_ready", "sample_count",
            "coverage_fraction", "steps_total", "accel_std_g", "gyro_rms_dps",
            "gyro_max_dps", "steps_in_window", "cadence_spm", "steps_status", "movement_deviation_raw")


def save_json(path, obj, attempts=6):
    """Atomic writes with unique temp file, Windows lock retries, nonfatal failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
            for attempt in range(attempts):
                try:
                    os.replace(tmp, path)
                    return True
                except (PermissionError, OSError) as exc:
                    if attempt + 1 == attempts:
                        print(f"[SAVE WARNING] Cannot update {path.name}: {exc}. Streaming continues.", flush=True)
                        return False
                    time.sleep(min(0.08 * (attempt + 1), 0.4))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    except (OSError, TypeError, ValueError) as exc:
        print(f"[SAVE WARNING] Cannot write {path.name}: {exc}. Streaming continues.", flush=True)
        return False


def save_history(result):
    row = {
        "timestamp": result["timestamp"], "source": result["source"],
        "activity_state": result["activity_state"], "movement_status": result["movement_status"],
        "movement_deviation_index": result["movement_deviation_index"],
        "baseline_ready": result["baseline"]["ready"],
        "sample_count": result["sample_count"],
        "coverage_fraction": result["coverage_fraction"],
        "steps_total": result["steps_total"],
        "steps_status": result.get("steps_status"),
        "movement_deviation_raw": result.get("movement_deviation_raw"),
        **result["features"],
    }
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with HISTORY.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_KEYS, extrasaction="ignore")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        print(f"[SAVE WARNING] Cannot append CSV: {exc}. Streaming continues.", flush=True)


def publish(engine, result, last_saved_revision):
    save_json(LATEST, result)
    # Persist the baseline only when NEW, VALID calibration windows were added.
    if engine.baseline_revision != last_saved_revision:
        if save_json(BASELINE, engine.export_baseline()):
            last_saved_revision = engine.baseline_revision
            engine.persisted_revision = last_saved_revision
    save_history(result)
    f = result["features"]
    accel = "--" if f["accel_std_g"] is None else f"{f['accel_std_g']:.3f}g"
    gyro = "--" if f["gyro_rms_dps"] is None else f"{f['gyro_rms_dps']:.1f}dps"
    print(f"{str(result['activity_state'] or '--'):>5} | {result['movement_status']:<18} "
          f"index={str(result['movement_deviation_index']):>5} | "
          f"accel_std={accel} | gyro_RMS={gyro} | "
          f"steps={str(f['steps_in_window'])} | "
          f"baseline {result['baseline']['collected_windows']}/{engine.required_windows} | "
          f"samples={result['sample_count']}", flush=True)
    return last_saved_revision


def run_session(engine, band, source, last_saved_revision, poll_interval=0.4, debug_steps=False):
    """Single BLE owner. Poll SDK frequently; aggregate features in 5s windows.

    IMU is *captured* at the configured device sampling rate (e.g. 25 Hz).
    For visualization we decimate to <=10 Hz and publish snapshots ~2 Hz.
    A different file holds the 5-second baseline analysis.
    """
    recent = deque(maxlen=240)  # up to 24 seconds at 10 Hz
    last_sample_at = time.monotonic()
    last_preview_sample_at = 0.0

    def on_motion(m):
        nonlocal last_sample_at, last_preview_sample_at
        engine.motion(m.ax, m.ay, m.az, m.gx, m.gy, m.gz)
        now_mono = time.monotonic()
        last_sample_at = now_mono
        if now_mono - last_preview_sample_at >= 0.10:
            try:
                acceleration = math.sqrt(m.ax*m.ax + m.ay*m.ay + m.az*m.az)
                rotation = math.sqrt(m.gx*m.gx + m.gy*m.gy + m.gz*m.gz)
                if math.isfinite(acceleration) and math.isfinite(rotation):
                    recent.append({"t": round(time.time(), 3),
                                   "accel_magnitude_g": round(acceleration, 4),
                                   "gyro_magnitude_dps": round(rotation, 3)})
                    last_preview_sample_at = now_mono
            except (TypeError, ValueError, OverflowError):
                pass

    def on_activity(a):
        engine.activity(a.steps, a.cadence, a.state)
        if debug_steps:
            print(f"[STEPS RAW] counter={a.steps} cadence={a.cadence} state={a.state}", flush=True)

    band.on("motion", on_motion)
    band.on("activity", on_activity)
    band.on("notice", lambda e: print("Vexo notice:", getattr(e, "kind", e), flush=True))
    band.configure(motion=engine.rate, sensors="both", activity=True)
    print(f"[LIVE] {band.name}: IMU {engine.rate} Hz; graph ~2 updates/s; deviation every {engine.seconds:g}s", flush=True)
    invalid_windows = 0
    window_started = time.monotonic()
    next_preview = window_started
    while True:
        # Do NOT run an independent second Bluetooth connection for Streamlit.
        before_wait = time.monotonic()
        band.wait(poll_interval)
        now = time.monotonic()
        if now - before_wait < 0.02:
            # SDK may return early on transport loss; avoid a tight CPU loop.
            time.sleep(0.04)
            now = time.monotonic()
        if now - last_sample_at > max(3.0, 2.5 * poll_interval):
            raise ConnectionError("no IMU samples for 3+ seconds; reconnecting")
        if now >= next_preview:
            # Trim old samples; plot decimated dynamics, NOT inferred pain.
            limit = time.time() - 22.0
            while recent and recent[0]["t"] < limit:
                recent.popleft()
            save_json(SIGNAL, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source, "connection_status": "connected",
                "sampling_rate_hz": engine.rate,
                "preview_rate_hz": 10,
                "samples": list(recent),
            }, attempts=2)
            next_preview = now + 0.45
        if now - window_started >= engine.seconds:
            result = engine.finish_window(source=source)
            invalid_windows = invalid_windows + 1 if result["movement_status"] == "insufficient_data" else 0
            last_saved_revision = publish(engine, result, last_saved_revision)
            window_started = now
            if invalid_windows >= 3:
                raise ConnectionError("three consecutive invalid movement windows; reconnecting")


def locate_sdk():
    candidates = [
        ROOT.parent / "vexo-open-kit" / "python",
        ROOT / "vexo-open-kit" / "python",
        ROOT.parent.parent / "vexo-open-kit" / "python",
    ]
    sdk = next((p for p in candidates if p.is_dir()), None)
    if sdk is None:
        raise SystemExit("Cannot find vexo-open-kit/python next to or above this module. Run --synthetic for testing.")
    sys.path.insert(0, str(sdk))
    return sdk


def run_band(engine, band_number, usb=False, sim=False, poll_interval=0.4, debug_steps=False):
    locate_sdk()
    from vexo_open.sync import Band

    source = "vexo-simulator" if sim else "vexo-band"
    transport = "serial" if usb else "sim" if sim else None
    consecutive_failures = 0
    last_saved_revision = engine.baseline_revision
    while True:
        band = None
        try:
            print(f"[CONNECTING] Vexo-HACK-{band_number} ...", flush=True)
            band = (Band.connect(band_number, transport=transport) if transport
                    else Band.connect(band_number))
            engine.reset_window(reconnect=True)
            # On any broken stream this session throws, triggering a fresh connect.
            run_session(engine, band, source, getattr(engine, "persisted_revision", last_saved_revision), poll_interval=poll_interval, debug_steps=debug_steps)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            consecutive_failures += 1
            engine.reset_window(reconnect=True)
            print(f"[DISCONNECTED] {type(exc).__name__}: {exc}", flush=True)
            save_json(LATEST, engine.status_result(source=source))
            save_json(SIGNAL, {"timestamp": datetime.now(timezone.utc).isoformat(),
                               "source": source, "connection_status": "disconnected",
                               "samples": []}, attempts=2)
            delay = min(2 ** min(consecutive_failures, 4), 15)
            print(f"[RECONNECTING] Retrying in {delay}s. Keep original dashboard closed.", flush=True)
            # Disconnect before sleeping, in the finally block below.
        finally:
            if band is not None:
                try:
                    band.disconnect()
                except Exception as exc:
                    print(f"[NOTICE] Error closing old Bluetooth session: {exc}", flush=True)
        time.sleep(delay)


def run_synthetic(engine):
    """No hardware required: stress test dashboard integration, never pain inference."""
    print("[SYNTHETIC] Fake data only. Press Ctrl+C to stop.", flush=True)
    window_idx = 0
    total_steps = 150
    last_saved_revision = engine.baseline_revision
    while True:
        phase = (window_idx // engine.required_windows) % 3
        if phase == 0:
            state, wiggle, rot, step_inc, cadence = "still", 0.01, 1.5, 0, 0
        elif phase == 1:
            state, wiggle, rot, step_inc, cadence = "walk", 0.12, 30, 8, 95
        else:
            state, wiggle, rot, step_inc, cadence = "still", 0.08, 24, 0, 0
        for k in range(int(engine.rate * engine.seconds)):
            s = k / engine.rate
            wave = math.sin(2 * math.pi * 1.6 * s)
            engine.motion(wiggle * wave + random.gauss(0, 0.004),
                          wiggle * 0.6 * wave, 1 + wiggle * 0.8 * wave,
                          rot * wave, rot * 0.4 * wave, rot * 0.2 * wave)
        total_steps += step_inc
        engine.activity(total_steps, cadence, state)
        result = engine.finish_window(source="synthetic")
        last_saved_revision = publish(engine, result, last_saved_revision)
        window_idx += 1
        time.sleep(engine.seconds)


def main():
    parser = argparse.ArgumentParser(description="Resilient Vexo movement monitoring")
    parser.add_argument("--band", type=int, default=79)
    parser.add_argument("--window", type=float, default=5)
    parser.add_argument("--rate", type=int, default=25)
    parser.add_argument("--poll", type=float, default=0.4, help="SDK read/publish interval in seconds; default 0.4")
    parser.add_argument("--debug-steps", action="store_true", help="Print raw activity step counts to verify counter semantics")
    parser.add_argument("--baseline-windows", type=int, default=4)
    parser.add_argument("--reset-baseline", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sim", action="store_true", help="Vexo SDK simulator")
    modes.add_argument("--usb", action="store_true", help="Connect through serial USB")
    modes.add_argument("--synthetic", action="store_true", help="Generate clearly labeled test data")
    args = parser.parse_args()
    if not 0.15 <= args.poll <= 5.0:
        parser.error("--poll must be between 0.15 and 5 seconds")

    saved = None
    if args.reset_baseline:
        try:
            BASELINE.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[SAVE WARNING] Could not delete old baseline ({exc}). Resetting in memory.", flush=True)
    elif BASELINE.exists():
        try:
            saved = json.loads(BASELINE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[SAVE WARNING] Cannot load baseline ({exc}); recalibrating.", flush=True)
    engine = MovementEngine(args.rate, args.window, args.baseline_windows, saved)
    print("Output:", LATEST, "and", HISTORY, "| Baseline:", BASELINE, flush=True)
    if args.synthetic:
        run_synthetic(engine)
    else:
        run_band(engine, args.band, usb=args.usb, sim=args.sim, poll_interval=args.poll, debug_steps=args.debug_steps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Ctrl+C received. Goodbye.", flush=True)
