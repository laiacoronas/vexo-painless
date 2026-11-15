"""Local bridge for the official Vexo Web Bluetooth SDK.

Chrome/Edge owns Bluetooth and streams real sensor samples to this localhost
server. The server NEVER scans for or connects to the band. Run alongside the
Streamlit dashboard, which reads its JSON/CSV output as before.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MODULE = PROJECT / "vexo_movement_module"
KIT_CANDIDATES = (PROJECT / "vexo-open-kit", PROJECT.parent / "vexo-open-kit")
KIT = next((p for p in KIT_CANDIDATES if (p / "js" / "src" / "index.js").exists()), None)
sys.path.insert(0, str(MODULE))
from movement_engine import MovementEngine  # noqa: E402

DATA = MODULE / "data"
LATEST = DATA / "movement_latest.json"
SIGNAL = DATA / "movement_signal.json"
HISTORY = DATA / "movement_windows.csv"
BASELINE = DATA / "movement_baseline.json"
CSV_KEYS = ("timestamp", "source", "activity_state", "movement_status", "movement_deviation_index",
            "baseline_ready", "sample_count", "coverage_fraction", "steps_total", "accel_std_g",
            "gyro_rms_dps", "gyro_max_dps", "steps_in_window", "cadence_spm", "steps_status", "movement_deviation_raw")
SOURCE = "vexo-band"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj, attempts=5):
    """Best effort, atomic, nonblocking enough for an interactive demonstration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        for attempt in range(attempts):
            try:
                os.replace(tmp, path)
                return True
            except OSError as exc:
                if attempt == attempts - 1:
                    print(f"[STORAGE] Could not update {path.name}: {exc}", flush=True)
                    return False
                time.sleep(.06 * (attempt + 1))
    except (OSError, TypeError, ValueError) as exc:
        print(f"[STORAGE] Could not write {path.name}: {exc}", flush=True)
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def save_history(result):
    row = {"timestamp": result["timestamp"], "source": result["source"],
           "activity_state": result["activity_state"], "movement_status": result["movement_status"],
           "movement_deviation_index": result["movement_deviation_index"],
           "baseline_ready": result["baseline"]["ready"], "sample_count": result["sample_count"],
           "coverage_fraction": result["coverage_fraction"], "steps_total": result["steps_total"],
           "steps_status": result.get("steps_status"),
           "movement_deviation_raw": result.get("movement_deviation_raw"), **result["features"]}
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with HISTORY.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_KEYS, extrasaction="ignore")
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
    except OSError as exc:
        print(f"[STORAGE] Could not update CSV: {exc}", flush=True)


def finite(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError, OverflowError):
        return None


class StreamProcessor:
    def __init__(self, rate=25, seconds=5, reset_baseline=False):
        self.lock = threading.RLock()
        if reset_baseline:
            saved = None
        else:
            try:
                saved = json.loads(BASELINE.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                saved = None
        self.engine = MovementEngine(rate=rate, seconds=seconds, baseline_windows=4, saved_baseline=saved)
        self.state = "disconnected"
        self.last_imu = 0.0
        self.last_post = 0.0
        self.window_start = time.monotonic()
        self.last_preview = 0.0
        self.last_preview_sample = 0.0
        self.preview = deque(maxlen=240)
        self.persisted_baseline_revision = 0
        self.last_quality = None
        atomic_json(LATEST, self.engine.status_result(status="disconnected", source=SOURCE))
        atomic_json(SIGNAL, {"timestamp": now_iso(), "source": SOURCE,
                             "connection_status": "disconnected", "samples": []})

    def transition(self, new):
        if new not in {"connected", "reconnecting", "disconnected"}:
            raise ValueError("Invalid status")
        with self.lock:
            if self.state == new:
                self.last_post = time.monotonic()  # keep-alive from the Web Bluetooth page
                return
            self.state = new
            self.last_post = time.monotonic()
            self.engine.reset_window(reconnect=True)
            self.preview.clear()
            self.window_start = time.monotonic()
            self.last_imu = 0
            self.last_preview_sample = 0
            self.last_quality = None
            if new != "connected":
                atomic_json(LATEST, self.engine.status_result(status="disconnected", source=SOURCE))
                atomic_json(SIGNAL, {"timestamp": now_iso(), "source": SOURCE,
                                     "connection_status": "disconnected", "samples": []})
            print(f"[BLE] {new}", flush=True)

    def ingest(self, packet):
        if not isinstance(packet, dict):
            raise ValueError("Expected an object")
        motion = packet.get("motion", [])
        activity = packet.get("activity", [])
        if not isinstance(motion, list) or not isinstance(activity, list) or len(motion) > 200 or len(activity) > 30:
            raise ValueError("Invalid batch size")
        accepted = 0
        with self.lock:
            if self.state != "connected":
                return 0
            now = time.monotonic()
            self.last_post = now
            for m in motion:
                if not isinstance(m, dict):
                    continue
                six = tuple(finite(m.get(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz"))
                if any(v is None for v in six):
                    continue
                # The Vexo accelerometer is in g and the gyro in degrees/sec.
                if max(abs(v) for v in six[:3]) > 32 or max(abs(v) for v in six[3:]) > 15000:
                    continue
                self.engine.motion(*six)
                accepted += 1
                self.last_imu = now
                stamp = finite(m.get("time_ms"))
                ts = (stamp / 1000.0 if stamp and abs(stamp / 1000.0 - time.time()) < 60 else time.time())
                if ts - self.last_preview_sample >= .085 or ts < self.last_preview_sample:
                    self.preview.append({"t": round(ts, 3),
                                         "accel_magnitude_g": round(math.sqrt(sum(v*v for v in six[:3])), 4),
                                         "gyro_magnitude_dps": round(math.sqrt(sum(v*v for v in six[3:])), 3)})
                    self.last_preview_sample = ts
            for a in activity:
                if not isinstance(a, dict):
                    continue
                self.engine.activity(a.get("steps"), a.get("cadence"), a.get("state"))
        return accepted

    def tick(self):
        with self.lock:
            now = time.monotonic()
            if self.state != "connected":
                return
            if now - self.last_post > 9:
                # The browser may have closed, frozen, or the user switched tabs.
                self.transition("disconnected")
                return
            if now - self.last_imu > 4:
                # Browser BLE is still logically connected, but no motion is arriving.
                if self.last_quality != "insufficient_data":
                    r = self.engine.status_result(status="insufficient_data", source=SOURCE)
                    r["connection_status"] = "connected"
                    atomic_json(LATEST, r)
                    atomic_json(SIGNAL, {"timestamp": now_iso(), "source": SOURCE,
                                         "connection_status": "connected", "samples": []})
                    self.last_quality = "insufficient_data"
                self.engine.reset_window()
                self.window_start = now
                return
            if now - self.last_preview >= .9:
                cutoff = time.time() - 22
                while self.preview and self.preview[0]["t"] < cutoff:
                    self.preview.popleft()
                atomic_json(SIGNAL, {"timestamp": now_iso(), "source": SOURCE,
                                     "connection_status": "connected", "sampling_rate_hz": self.engine.rate,
                                     "preview_rate_hz": 10, "samples": list(self.preview)})
                self.last_preview = now
            if now - self.window_start >= self.engine.seconds:
                result = self.engine.finish_window(source=SOURCE)
                atomic_json(LATEST, result)
                if self.engine.baseline_revision != self.persisted_baseline_revision:
                    if atomic_json(BASELINE, self.engine.export_baseline()):
                        self.persisted_baseline_revision = self.engine.baseline_revision
                save_history(result)
                self.last_quality = result.get("movement_status")
                self.window_start = now
                print(f"[IMU] {result['sample_count']} samples | {result['movement_status']} | "
                      f"index={result['movement_deviation_index']} | steps={result['features']['steps_in_window']}", flush=True)

    def status(self):
        with self.lock:
            return {"status": self.state, "rate": self.engine.rate,
                    "last_imu_age_s": round(time.monotonic() - self.last_imu, 2) if self.last_imu else None}


def make_handler(processor, kit_root):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PROJECT), **kwargs)

        def translate_path(self, path):
            raw = unquote(urlsplit(path).path)
            if raw.startswith("/vexo-open-kit/"):
                if kit_root is None:
                    return str(PROJECT / "___missing_sdk___")
                rel = raw.removeprefix("/vexo-open-kit/")
                candidate = (kit_root / rel).resolve()
                if candidate.is_relative_to(kit_root.resolve()):
                    return str(candidate)
                return str(PROJECT / "___invalid___")
            return super().translate_path(path)

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/ble_bridge/index.html")
                self.end_headers()
                return
            if path == "/api/health":
                self._json({"bridge": "ready", **processor.status()})
                return
            return super().do_GET()

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in {"/api/status", "/api/push"}:
                self._json({"error": "unknown endpoint"}, 404)
                return
            try:
                content_type = self.headers.get("Content-Type", "").lower()
                origin = self.headers.get("Origin", "")
                if not content_type.startswith("application/json"):
                    raise ValueError("Expected JSON content type")
                if origin and urlsplit(origin).hostname not in ("localhost", "127.0.0.1"):
                    raise ValueError("Only local browser requests are permitted")
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 120000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(n))
                if path == "/api/status":
                    processor.transition(str(data.get("status", "")))
                    self._json({"ok": True})
                else:
                    count = processor.ingest(data)
                    self._json({"ok": True, "accepted": count})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                print(f"[HTTP] {type(exc).__name__}: {exc}", flush=True)
                self._json({"error": "internal server error"}, 500)

        def log_message(self, fmt, *args):
            # Avoid printing a line for every 500 ms BLE packet batch.
            if not self.path.startswith("/api/"):
                super().log_message(fmt, *args)
    return Handler


def main():
    p = argparse.ArgumentParser(description="Vexo Web Bluetooth → Streamlit local bridge")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rate", type=int, default=25, choices=(10, 25, 50, 100))
    p.add_argument("--reset-baseline", action="store_true")
    args = p.parse_args()
    if KIT is None:
        raise SystemExit("Could not find vexo-open-kit/js/src/index.js. Put vexo-open-kit next to or inside vexo-pain-monitor.")
    processor = StreamProcessor(rate=args.rate, reset_baseline=args.reset_baseline)
    handler = make_handler(processor, KIT)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.daemon_threads = True
    print("\n[BRIDGE] Official Vexo JavaScript SDK (Web Bluetooth)\n"
          f"[BRIDGE] Open http://localhost:{args.port}/ble_bridge/index.html in Chrome or Edge\n"
          f"[BRIDGE] SDK directory: {KIT}\n"
          "[BRIDGE] This process does NOT connect to Bluetooth. Click Connect in the browser.\n", flush=True)
    def monitor():
        while True:
            try:
                processor.tick()
            except Exception as exc:
                print(f"[MONITOR] {type(exc).__name__}: {exc}", flush=True)
            time.sleep(.25)
    threading.Thread(target=monitor, daemon=True).start()
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print("\n[BRIDGE] Stopped.", flush=True)
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
