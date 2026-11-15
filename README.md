<div align="center">

# VEXO / PAIN MONITOR

### Understanding pain beyond words.

**A personalized wearable experience that brings together voice, movement, and self-reported pain to reveal changes in everyday behavior.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Vexo](https://img.shields.io/badge/Vexo-Wearable%20SDK-5576A7?style=flat-square)](https://github.com/VexoAI-Org/vexo-open-kit)
![Status](https://img.shields.io/badge/Status-Hackathon%20prototype-8493A8?style=flat-square)

**Live movement monitoring** · **Personal baselines** · **Voice-ready architecture** · **Patient feedback**

</div>

---

## AST trained on TAME

[Base model, regression heads, training protocol, and results](models/ast_tame/README.md).

## The problem

Pain is personal, dynamic, and difficult to communicate. Most pain assessments depend on a person stopping what they are doing and answering a question such as *“How much does it hurt, from 0 to 10?”*

That self-report is essential — it is the closest thing we have to ground truth for the person's experience — but it provides only a snapshot. Between check-ins, changes in behavior may go unnoticed.

At the same time, a change in a person's voice or movement **does not automatically mean they are in pain**. The same signals can change with exercise, excitement, stress, fatigue, or the surrounding environment.

**Our question:** *Can a wearable recognize meaningful changes from someone's own usual patterns, then use brief self-reports to help interpret those changes over time?*

## Our approach

Vexo Pain Monitor explores three complementary sources of information:

| Signal | What we observe | Why it matters |
|:--|:--|:--|
| 🎙️ **Voice** | Acoustic features such as pitch, energy, and spectral patterns | An additional, potentially informative behavioral signal; voice inference is being integrated |
| ⌚ **Movement** | Accelerometer, gyroscope, activity state, steps, and cadence | Detects deviations from a person's usual activity-specific movement patterns |
| 💬 **Self-reported pain** | Voluntary 0–10 pain rating and optional context | Captures the person's own experience; used for feedback, evaluation, and future personalization |

Rather than comparing everyone with a single universal threshold, we focus on **changes relative to each person's baseline**.

> **Core principle:** *Movement is a signal. Voice is a signal. Pain is an experience.* We should learn how they relate — not treat them as interchangeable.

## How it works

```text
               VEXO WEARABLE — BAND 99
                          │
          Official Vexo Web Bluetooth SDK
            (one shared BLE connection)
                          │
               ┌──────────┴──────────┐
               │                     │
        MOVEMENT STREAM         VOICE STREAM
        Accel · Gyro            Audio · Features
        Steps · Cadence         (integration pending)
               │                     │
        5-second windows        Voice model
        + quality checks        + voice baseline
               │                     │
        Activity-matched               │
        personal baseline              │
               │                     │
        Movement deviation      Voice assessment
               └──────────┬──────────┘
                          │
                  STREAMLIT DASHBOARD
                   Live · Timeline
                  Patterns · Check-in
                          │
                VOLUNTARY PAIN RATING
                    0–10 + context
                          │
              Future evaluation and
             individualized calibration
```

### 1. Personal movement baseline

The monitor calculates movement features in rolling windows and compares them against an activity-matched reference, rather than treating all motion as equivalent.

- **Accelerometer:** variability of acceleration magnitude.
- **Gyroscope:** rotational movement intensity (RMS angular speed).
- **Activity context:** still, walking, or running, when the device reports it.
- **Steps / cadence:** additional context from the wearable's activity stream.
- **Quality control:** incomplete windows, disconnected sensors, and questionable step-counter jumps are flagged instead of converted to fake zero values.

The resulting **Movement Deviation Index** (0–100) summarizes how different the current window is from the learned reference. It is a *heuristic measure of movement change*, **not** a pain score or a probability of pain. A short movement spike should not by itself trigger a sustained high alert.

### 2. Voice module — designed for integration

The planned voice model processes speech-derived acoustic characteristics and produces a separate assessment. The team is developing it independently so that it can be connected to the dashboard without changing the movement pipeline.

Our proposed research direction uses labeled speech/pain datasets (for example, TAME Pain) to explore relationships between acoustic patterns and reported pain. Any trained model needs participant-level evaluation, calibration, and checks for generalization before its output can be interpreted reliably.

**Current behavior:** if no valid recent model output exists, the dashboard shows **Model pending** — it does not invent a voice prediction.

### 3. User check-ins — the reference for actual pain

Users can enter their own pain rating from **0 (no pain) to 10 (worst imaginable pain)**, plus an optional note describing the situation.

We store the rating alongside the available time-stamped sensor summaries. The reported value is **not fed into a model to predict that same value**. Instead, repeated check-ins could eventually support individual calibration, feedback on false alarms, and longitudinal analysis.

## Prototype experience

| Screen | What it shows |
|:--|:--|
| **Now** | Live wrist-motion trace, activity, movement deviation, connectivity, voice placeholder, and pain check-in |
| **Timeline** | Changes over time and an illustrative two-week history |
| **Patterns** | Examples of recurring situations that might be worth investigating |
| **Experiment** | An optional movement-break concept for exploring behavior changes |
| **Moment** | A focused check-in when a movement deviation is observed |
| **Share** | A local summary and downloadable self-report records |

The prototype includes **Live device** for genuine wearable readings and **Preview** for guided walkthroughs. The two-week example history is illustrative and is kept distinct from collected sensor data.

## Technology

| Layer | Implementation |
|:--|:--|
| Wearable | Vexo band **99** |
| Device connection | Official [Vexo Open Kit](https://github.com/VexoAI-Org/vexo-open-kit) JavaScript SDK + Web Bluetooth (Chrome / Edge) |
| Local bridge | Python HTTP service (`ble_bridge/server.py`) |
| Movement processing | Python baseline and feature engine (`vexo_movement_module/movement_engine.py`) |
| Interface | Streamlit + Plotly (`app/vexo_demo.py`) |
| Data exchange | Local JSON / CSV files |
| Voice integration | Optional `data/voice_latest.json` contract |

Movement samples are requested at **25 Hz** in the current configuration. The display uses a reduced-rate motion trace for responsiveness, while the baseline assessment updates approximately every **5 seconds**. Effective rates depend on the hardware and browser connection.

## Quick start

**Requirements:** Windows with Bluetooth, Chrome or Edge, Conda environment `envlaia`, and a local copy of `vexo-open-kit` (in this project directory or its parent). Install dashboard dependencies if needed:

```powershell
conda activate envlaia
cd C:\Users\lclai\Desktop\vexo-project\vexo-pain-monitor
python -m pip install -r .\requirements_demo.txt
```

**Terminal 1 — Start the local Web Bluetooth bridge**

```powershell
conda activate envlaia
python .\ble_bridge\server.py
```

Open **http://localhost:8765/ble_bridge/index.html** in Chrome or Edge. Double-tap the wearable, select **Connect Vexo 99**, and choose `Vexo-HACK-99` in the browser's device picker. Keep the connector tab open.

**Terminal 2 — Start the dashboard**

```powershell
conda activate envlaia
streamlit run .\app\vexo_demo.py
```

Open **http://localhost:8501** (or the local address printed by Streamlit) and choose **Live device**.

> **One Bluetooth owner only.** Do not run the original Vexo dashboard or `movement_live.py` at the same time. When the voice model is connected, it should share the existing Web Bluetooth session rather than create a second connection to the band.

**First-time calibration:** stop the bridge and restart it with:

```powershell
python .\ble_bridge\server.py --reset-baseline
```

Collect a few valid windows while still and walking normally. Baselines are learned separately by activity state. Use the normal command on subsequent runs to retain calibration.

## Data contracts

The bridge writes local files that the dashboard reads:

```text
vexo-pain-monitor/
├── app/
│   ├── vexo_demo.py
│   └── vexo_demo_data.py
├── ble_bridge/
│   ├── index.html
│   └── server.py
├── vexo_movement_module/
│   ├── movement_engine.py
│   └── data/
│       ├── movement_latest.json
│       ├── movement_signal.json
│       ├── movement_windows.csv
│       └── movement_baseline.json
├── data/
│   ├── voice_latest.json          # optional; produced by the voice module
│   └── vexo_demo_checkins.csv     # created when users submit check-ins
├── demo_data/
│   └── example_history.csv
└── requirements_demo.txt
```

**Optional voice-model output:** the dashboard accepts this format when a recent, real prediction becomes available:

```json
{
  "timestamp": "2026-11-14T23:00:00+00:00",
  "source": "voice-model",
  "voice_status": "ready",
  "voice_level": "moderate",
  "voice_deviation_index": 48.2,
  "features": {
    "pitch_hz": 190.4
  }
}
```

*The JSON above is a schema example, not a model result.* The voice status must be `ready`, the level must be `normal`, `moderate`, or `high`, and the timestamp must be fresh for the dashboard to display it.

## What works today / what comes next

| Capability | Status |
|:--|:--|
| Live accelerometer and gyroscope from Vexo | ✅ Implemented; device testing required |
| Activity-specific movement baseline | ✅ Implemented |
| Movement deviation + quality checks | ✅ Implemented |
| Live Streamlit dashboard | ✅ Implemented |
| User-entered pain ratings | ✅ Implemented |
| Example timeline and insights | ✅ Available for walkthrough |
| Trained voice-model output | 🚧 Teammate integration pending |
| Joint voice + movement pain prediction | 🔬 Future research; requires synchronized labeled data |
| Clinical validation | ❌ Not performed |

## Responsible interpretation

This is a **hackathon research prototype**, not a medical device, diagnosis, or treatment recommendation.

- **Atypical movement is not necessarily pain.** Walking, exercising, and moving the wrist can all alter sensor readings.
- **Voice changes are not specific to pain.** They may reflect many other factors.
- **The movement index is not clinical pain severity.** Its 0–100 range is a visualization of deviation, not a validated medical scale.
- **Self-reports remain distinct from predictions.** They reflect what the user reports experiencing.
- **Illustrative histories are labeled as examples.** They must not be represented as real patient observations.
- **Data privacy matters.** Check-ins and sensor summaries are local in the current prototype. Obtain appropriate consent and define retention/security policies before collecting identifiable health data beyond a demo.

## Where we want to go

The long-term goal is a **personalized, context-aware system** that learns which combinations of vocal, movement, and self-reported changes are meaningful for each person. With longitudinal labeled data, we could investigate calibrated multimodal fusion, activity-aware comparisons, adaptive check-in timing, and interpretable weekly summaries.

For now, the demo establishes the essential foundation: **a real wearable stream, a personal movement baseline, an honest feedback loop, and a modular path to voice integration.**

---

<div align="center">

**Beyond a single pain score. Toward a better understanding of change.**

*Built for a Vexo hackathon · Powered by the Vexo Open Kit*

</div>
