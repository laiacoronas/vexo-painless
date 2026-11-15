"""Elementary acoustic features only, not a pain detector."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "vexo-open-kit"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 milliseconds at 16 kHz


def summarize_pcm(pcm):
    """Summarise 16-bit mono PCM bytes without classifying pain or emotions."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    count = len(samples) // FRAME_SAMPLES
    if count == 0:
        raise ValueError("The recording has too few samples to analyse")
    frames = samples[:count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    dbfs = np.maximum(-100.0, 20.0 * np.log10(np.maximum(rms, 1e-5)))
    # Relative activity is NOT validated voice-activity detection. It can include background noise.
    baseline = float(np.percentile(dbfs, 20))
    activity_threshold = max(-45.0, baseline + 12.0)
    return {
        "duration_seconds": round(len(samples) / SAMPLE_RATE, 3),
        "median_dbfs": round(float(np.median(dbfs)), 2),
        "p90_dbfs": round(float(np.percentile(dbfs, 90)), 2),
        "variability_db": round(float(np.std(dbfs)), 2),
        "high_energy_fraction": round(float(np.mean(dbfs > activity_threshold)), 3),
        "activity_threshold_dbfs": round(activity_threshold, 2),
        "timeline_seconds": [round(float(i * FRAME_SAMPLES / SAMPLE_RATE), 2) for i in range(count)],
        "energy_dbfs": [round(float(v), 2) for v in dbfs],
    }


def analyze_ogg(path):
    if not (KIT / "python" / "vexo_open").exists():
        raise FileNotFoundError("Clone vexo-open-kit into the project first")
    sys.path.insert(0, str(KIT / "python"))
    from vexo_open import audio
    pcm = audio.decode_packets(audio.ogg_packets(path))
    return summarize_pcm(pcm)
