"""Local data adapters for the Vexo pain-patterns hackathon demonstration.

Historical example events are illustrative and not actual user observations. Bluetooth data is only read from
movement_live.py's JSON/CSV output. No pain inference is performed here.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
ROOT = Path(__file__).resolve().parents[1]
MOVEMENT_CANDIDATES = (
    ROOT / "vexo_movement_module" / "data" / "movement_latest.json",
    ROOT / "data" / "movement_latest.json",
)
MOTION_SIGNAL_CANDIDATES = (
    ROOT / "vexo_movement_module" / "data" / "movement_signal.json",
    ROOT / "data" / "movement_signal.json",
)
MOVEMENT_HISTORY_CANDIDATES = (
    ROOT / "vexo_movement_module" / "data" / "movement_windows.csv",
    ROOT / "data" / "movement_windows.csv",
)
VOICE_CANDIDATES = (
    ROOT / "data" / "voice_latest.json",
    ROOT / "vexo_voice_module" / "data" / "voice_latest.json",
)
DEMO_HISTORY_FILE = ROOT / "demo_data" / "example_history.csv"
CHECKINS = ROOT / "data" / "vexo_demo_checkins.csv"
MOMENTS = ROOT / "data" / "vexo_demo_moments.csv"
EXPERIMENT = ROOT / "data" / "vexo_demo_experiment.json"


def choose_file(candidates, env_name=None):
    """Prefer explicit override, then first existing file, then standard path."""
    if env_name and os.environ.get(env_name):
        return Path(os.environ[env_name]).expanduser()
    return next((p for p in candidates if p.is_file()), candidates[0])


def parse_timestamp(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None  # never guess a timezone for live sensor timestamps
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def read_json(path):
    """Fail closed on missing or partially written records."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, PermissionError, OSError, ValueError, UnicodeError):
        return None


def live_movement(path, max_age_seconds=18, now=None, require_hardware=True):
    """Return (payload, state, age_s). Never report stale/sim data as live.

    State: connected, calibrating, insufficient_data, disconnected,
           stale, simulated_source, waiting, malformed.
    """
    payload = read_json(path)
    if payload is None:
        return None, "waiting", None
    source = payload.get("source")
    if require_hardware and source != "vexo-band":
        return payload, "simulated_source", None
    timestamp = parse_timestamp(payload.get("timestamp"))
    if timestamp is None:
        return payload, "malformed", None
    now = now or datetime.now(timezone.utc)
    age = (now - timestamp).total_seconds()
    if age < -5 or age > max_age_seconds:
        return payload, "stale", age
    if payload.get("connection_status") != "connected":
        return payload, "disconnected", age
    status = payload.get("movement_status")
    if status in {"calibrating", "insufficient_data"}:
        return payload, status, age
    if status not in {"typical", "change", "strong_change"}:
        return payload, "malformed", age
    value = safe_float(payload.get("movement_deviation_index"))
    if value is None or not 0 <= value <= 100:
        return payload, "insufficient_data", age
    if safe_float(payload.get("sample_count")) is None or payload["sample_count"] < 1:
        return payload, "insufficient_data", age
    return payload, "connected", age


def demo_movement(scenario="Typical", now=None):
    """A moving, clearly example stream. No claim of clinical significance."""
    now = now or datetime.now(timezone.utc)
    tick = now.timestamp() / 3
    offset = 3.8 * math.sin(tick * 0.63)
    cases = {
        "Typical": (17, "typical", "still", 0.012, 5.5, 0),
        "Moderate": (49, "change", "walk", 0.105, 65, 7),
        "High": (82, "strong_change", "walk", 0.380, 290, 10),
    }
    base, status, activity, accel, gyro, steps = cases.get(scenario, cases["Typical"])
    index = max(0., min(100., base + offset))
    return {
        "timestamp": now.isoformat(), "source": "example-preview", "connection_status": "connected",
        "window_seconds": 5, "sample_count": 125, "coverage_fraction": 1.0,
        "activity_state": activity, "steps_total": None,
        "features": {
            "accel_std_g": round(accel * (1 + .07 * math.sin(tick)), 4),
            "gyro_rms_dps": round(gyro * (1 + .08 * math.sin(tick * .37)), 1),
            "gyro_max_dps": None, "steps_in_window": steps,
            "cadence_spm": 88 if activity == "walk" else 0,
        },
        "baseline": {"state": activity, "collected_windows": 4, "required_windows": 4, "ready": True},
        "movement_deviation_index": round(index, 1), "movement_status": status,
        "pain_estimate": None,
        "note": "Illustrative movement deviations. Not a pain estimate or clinical measurement.",
    }


def live_signal(path, now=None, max_age_seconds=3.5):
    """Read the real IMU preview. Never mistake example/replayed data for BLE."""
    payload = read_json(path)
    if not payload or payload.get("source") != "vexo-band":
        return [], "waiting"
    timestamp = parse_timestamp(payload.get("timestamp"))
    now = now or datetime.now(timezone.utc)
    if not timestamp or not -5 <= (now-timestamp).total_seconds() <= max_age_seconds:
        return [], "stale"
    if payload.get("connection_status") != "connected":
        return [], "disconnected"
    points=[]
    for row in (payload.get("samples") or [])[-240:]:
        if not isinstance(row, dict):
            continue
        t=safe_float(row.get("t")); a=safe_float(row.get("accel_magnitude_g")); g=safe_float(row.get("gyro_magnitude_dps"))
        if t is None or a is None or g is None:
            continue
        if now.timestamp()-25 <= t <= now.timestamp()+5 and 0 <= a < 40 and 0 <= g < 10000:
            points.append({"timestamp":datetime.fromtimestamp(t,timezone.utc).astimezone(TZ),
                           "accel":a,"gyro":g})
    return points, "connected" if points else "waiting"


def demo_signal(scenario="Typical", now=None):
    """A visually dynamic *explicitly example* IMU trace, independent of BLE."""
    now = now or datetime.now(timezone.utc)
    strength={"Typical":(0.035,8),"Moderate":(0.17,65),"High":(0.35,155)}.get(scenario,(.035,8))
    vals=[]
    for i in range(220):
        t=now.timestamp()-(219-i)*.1
        wave=math.sin(2*math.pi*1.45*t)
        acc=1+strength[0]*wave + .35*strength[0]*math.sin(5.1*t)
        gyro=abs(strength[1]*(wave+.2*math.sin(3.2*t)))
        vals.append({"timestamp":datetime.fromtimestamp(t,timezone.utc).astimezone(TZ),
                     "accel":round(acc,4),"gyro":round(gyro,2)})
    return vals

def live_voice(path, max_age_seconds=18, now=None):
    """Optional teammate integration. Missing model is the expected default."""
    data = read_json(path)
    if not data:
        return None, "pending"
    if data.get("source") != "voice-model":
        return None, "pending"
    ts = parse_timestamp(data.get("timestamp"))
    now = now or datetime.now(timezone.utc)
    if not ts or (now - ts).total_seconds() > max_age_seconds or (now - ts).total_seconds() < -5:
        return None, "stale"
    level = str(data.get("voice_level", "")).lower()
    if level not in {"normal", "moderate", "high"}:
        return None, "invalid"
    if data.get("voice_status") != "ready":
        return None, "pending"
    idx = safe_float(data.get("voice_deviation_index"))
    if idx is not None and not 0 <= idx <= 100:
        idx = None
    return {"level": level, "index": idx, "features": data.get("features", {})}, "ready"


def read_demo_events(file_path=DEMO_HISTORY_FILE, as_of=None):
    """Shift the fixed 14-day example example to end on the current local date."""
    as_of = as_of or datetime.now(TZ).date()
    first_day = as_of - timedelta(days=13)
    events = []
    try:
        with Path(file_path).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                day = first_day + timedelta(days=int(row["day_index"]))
                hour = int(row["hour"])
                minute = int(row["minute"])
                local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=TZ)
                events.append({
                    "timestamp": local, "day": day, "hour_float": hour + minute / 60,
                    "severity": row["severity"], "context": row["context"],
                    "after_long_sitting": row["after_long_sitting"] == "1",
                    "had_lunch_walk": row["had_lunch_walk"] == "1",
                    "body_area": row["body_area"], "source": "example_history",
                    "day_index": int(row["day_index"]),
                })
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return []
    return sorted(events, key=lambda r:r["timestamp"])


def movement_rows(path, recent_hours=24):
    """Read actual windows, excluding example/test and invalid rows."""
    now = datetime.now(timezone.utc)
    rows = []
    try:
        with Path(path).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("source") != "vexo-band":
                    continue
                ts = parse_timestamp(row.get("timestamp"))
                idx = safe_float(row.get("movement_deviation_index"))
                if not ts or not 0 <= (idx if idx is not None else -1) <= 100:
                    continue
                if not 0 <= (now - ts).total_seconds() <= recent_hours * 3600:
                    continue
                if row.get("movement_status") not in ("typical", "change", "strong_change"):
                    continue
                rows.append({"timestamp": ts.astimezone(TZ), "index": idx,
                             "status": row["movement_status"],
                             "steps": safe_float(row.get("steps_in_window"))})
    except (FileNotFoundError, PermissionError, OSError, csv.Error, UnicodeError):
        return []
    return rows[-800:]


def save_csv_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_checkin(pain, activity, body_area, note, payload, connection_state):
    """Label one time-aligned sensor snapshot; never attach a stale sensor score."""
    valid = connection_state == "connected" and isinstance(payload, dict)
    sensor = payload if valid else {}
    features = sensor.get("features") or {}
    voice, voice_state = live_voice(choose_file(VOICE_CANDIDATES, "VEXO_VOICE_JSON"))
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "self_reported_pain_0_10": int(pain), "reported_activity": activity,
        "reported_body_area": body_area, "note": note,
        "movement_source": (payload or {}).get("source", "none"),
        "movement_connection_state": connection_state,
        "movement_window_timestamp": sensor.get("timestamp"),
        "movement_deviation_index": sensor.get("movement_deviation_index"),
        "movement_status": sensor.get("movement_status", "unavailable"),
        "accel_std_g": features.get("accel_std_g"),
        "gyro_rms_dps": features.get("gyro_rms_dps"),
        "steps_in_window": features.get("steps_in_window"),
        "voice_source": "voice-model" if voice_state == "ready" else None,
        "voice_level": voice.get("level") if voice else None,
        "voice_deviation_index": voice.get("index") if voice else None,
    }
    save_csv_row(CHECKINS, row)
    return row


def log_moment(kind, payload, connection_state, context=""):
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback": kind, "context": context,
        "movement_source": (payload or {}).get("source", "none"),
        "movement_connection_state": connection_state,
        "movement_status": (payload or {}).get("movement_status", "unavailable"),
        "movement_deviation_index": (payload or {}).get("movement_deviation_index"),
    }
    save_csv_row(MOMENTS, row)
    return row


def read_checkins():
    try:
        with CHECKINS.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except (FileNotFoundError, OSError, csv.Error):
        return []


def summaries(events):
    moderate = [e for e in events if e["severity"] == "moderate"]
    high = [e for e in events if e["severity"] == "high"]
    morning = [e for e in moderate if 7 <= e["hour_float"] < 9]
    weekdays = [e for e in high if e["day"].weekday() in (0, 3)]
    sitting = [e for e in high if e["after_long_sitting"]]
    return {
        "moderate":len(moderate), "high": len(high), "morning": len(morning),
        "mon_thu":len(weekdays), "long_sitting":len(sitting)
    }


def weekly_counts(events, as_of=None):
    as_of = as_of or datetime.now(TZ).date()
    days = [as_of - timedelta(days=i) for i in range(6, -1, -1)]
    counts = {d:{"moderate":0,"high":0} for d in days}
    for e in events:
        if e["day"] in counts and e["severity"] in counts[e["day"]]:
            counts[e["day"]][e["severity"]] += 1
    return days, counts


def csv_download(rows):
    if not rows:
        return "No check-ins saved yet.\n"
    buf = io.StringIO()
    w=csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
