# Gemini Live voice panel

Google Gemini Live handles microphone input, transcription and spoken replies over one persistent connection. The Voice panel sits outside the dashboard refresh cycle.

Install `vexo_voice_module/requirements.txt`, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment or repository `.env`, then run:

```sh
python -m uvicorn vexo_voice_module.server:app --host 127.0.0.1 --port 8768
```

`VEXO_ENV_FILE` can point to an existing local credential file. `VEXO_VOICE_URL` configures the dashboard iframe URL. Keys stay server-side.

Optional: set `VEXO_PAIN_FACTORY=module:factory` to an adapter exposing `score_pcm(pcm)` for 24-kHz mono PCM16. Analysis runs in the background and publishes `data/voice_latest.json`; moderate/high events retain timestamp, score and transcript. Model weights are configured separately.
