# Model development plan

This repository currently extracts **descriptive audio features**, not a trained pain predictor.

1. Pre-register a protocol and consent flow. Record controlled speech tasks under varied conditions.
2. Collect participant ID (pseudonym), voice sessions, timestamp, self-reported pain (0–10), context (activity, noise, microphone placement), and other relevant covariates.
3. Extract pitch/F0, loudness, spectral measures, pauses, voice quality and speech embeddings from the *audio* (text alone loses prosody).
4. Compare to per-user baselines; evaluate data quality, packet loss and confounders.
5. Only after enough data: split data by participant and, when relevant, time. Compare simple baselines against supervised regression/classification and report MAE/calibration/uncertainty.
6. Never market the prototype as diagnosing pain or make clinical claims from an unvalidated model.
