"""Publish real predictions through the dashboard's existing voice contract."""
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def publish_voice(event, path=None):
    score = float(event['score'])
    if not math.isfinite(score):
        raise ValueError('Pain score must be finite')
    level = 'high' if score >= 8 else 'moderate' if score >= 4 else 'normal'
    path = Path(path or os.environ.get('VEXO_VOICE_JSON', ROOT / 'data/voice_latest.json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**event, 'source':'voice-model', 'voice_status':'ready',
               'voice_level':level, 'voice_deviation_index':None}
    # Keep the raw 0–10 prediction separate from the movement/deviation scale.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.voice-',suffix='.tmp')
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(payload,f,allow_nan=False)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
