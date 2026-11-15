"""Local hackathon dashboard for Vexo Open 79."""

import os
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
from src.analyze import analyze_ogg

st.set_page_config(page_title="Vexo Voice Lab", page_icon="🎙️", layout="wide")
st.title("🎙️ Vexo Voice Lab")
st.caption("Hackathon prototype · Vexo Open 79 · Recording + transcript + acoustic exploration")
st.info("Prototype for research, not a pain diagnosis or validated pain prediction tool.")

st.subheader("1. Capture a session")
st.write("Close the official Vexo dashboard before recording; only one Bluetooth client can connect.")
a, b, c = st.columns(3)
with a:
    seconds = st.slider("Duration (seconds)", 5, 30, 15)
with b:
    with_text = st.checkbox("Live transcription via Deepgram", value=False)
with c:
    simulate = st.checkbox("Demo mode (simulated band)", value=False)
language = st.selectbox("Transcription language", ["es", "en"], index=0)
if st.button("Record session", type="primary"):
    cmd = [sys.executable, "-m", "src.capture", "--seconds", str(seconds), "--language", language]
    if with_text:
        cmd.append("--text")
    if simulate:
        cmd.append("--sim")
    with st.spinner("Capturing audio from band 79 ..."):
        try:
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=100)
        except subprocess.TimeoutExpired:
            st.error("Capture timed out. Check Bluetooth and close the original dashboard.")
        else:
            if result.returncode == 0:
                st.success("Recording saved locally.")
            else:
                st.error("Capture failed or finished with a warning; see the log below.")
            st.code((result.stdout + "\n" + result.stderr).strip(), language="text")

st.subheader("2. Review session data")
files = sorted((ROOT / "data").glob("*/session.json"), reverse=True)
if not files:
    st.write("No recordings yet. Record one above, or test with Demo mode.")
    st.stop()
labels = [p.parent.name for p in files]
chosen_label = st.selectbox("Session", labels)
json_path = files[labels.index(chosen_label)]
session = json.loads(json_path.read_text(encoding="utf-8"))
audio_path = json_path.parent / session["audio_file"]

left, right = st.columns([1, 2])
with left:
    st.write("**Device:**", session["device"])
    st.write("**Captured:**", session["captured_at_utc"])
    st.write("**Lost packets:**", session["packet_loss"])
    st.audio(audio_path.read_bytes(), format="audio/ogg")
    st.write("**Transcript:**")
    st.write(session.get("transcription") or "(No transcription captured)")
    pain = session.get("pain_score_self_report_0_to_10")
    score = st.selectbox("Optional self-reported pain (0–10)", [None] + list(range(11)), index=0 if pain is None else int(pain) + 1)
    if st.button("Save self-report"):
        session["pain_score_self_report_0_to_10"] = score
        json_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        st.success("Label saved locally.")
with right:
    try:
        data = analyze_ogg(audio_path)
    except Exception as exc:
        st.warning("Could not decode audio; install 'av' and check the file. " + str(exc))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Audio duration", f"{data['duration_seconds']} s")
        c2.metric("Median energy", f"{data['median_dbfs']} dBFS")
        c3.metric("Energy variability", f"{data['variability_db']} dB")
        st.line_chart(data["energy_dbfs"], x_label="20 ms frames", y_label="dBFS")
        st.caption("Audio level and relative high-energy activity are descriptive acoustic features, not estimates of pain.")
        st.metric("High-energy frame fraction", f"{100 * data['high_energy_fraction']:.1f}%")

st.subheader("3. Next research milestone")
st.write("Collect consented labeled recordings, compare sessions within each participant, evaluate robust features and validate any supervised model on held-out participants.")
