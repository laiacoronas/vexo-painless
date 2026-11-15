# Speech routing

The Voice card opens this local conversation service. Audio flows through
Deepgram Nova-3 → Gemini → Deepgram Flux speech. The Gemini prompt uses Vexo's
conversation identity. The default pause is 0.3 seconds. Agent keepalives preserve complete replies; interruptions
require recognized words.

From the repository root:

```sh
python -m pip install -r vexo_voice_module/requirements.txt
python -m uvicorn vexo_voice_module.server:app --host 127.0.0.1 --port 8766
```

Set `DEEPGRAM_API_KEY` and `GEMINI_API_KEY` in the repository's `.env` file or
server environment. `GEMINI_MODEL` defaults to `gemini-4-carbon`. Keys stay on the
server. Open the dashboard's Voice card and select **Open conversation**.
`VEXO_VOICE_URL` overrides the link when the service uses a different local port.

## Pain model hook

Set `VEXO_PAIN_FACTORY=module:factory` in the server environment. The callable
must return an object with `score_pcm(pcm)`; PCM is mono, signed little-endian
16-bit audio at 24 kHz. Return `None` for insufficient speech, or a dictionary
with finite `score`, `level` (`low`, `moderate`, `high`), and optional model/chunk
metadata. Analyze all chunks of at most three seconds and average their scores.
Use moderate for 4–<8 and high for 8+. Model weights remain separately configured.

Milan's existing Lamita adapter can be used with the Painless workspace on
`PYTHONPATH` and `VEXO_PAIN_FACTORY=demo.server.pain:PainModel`.

The toggle is available when a model is configured. With prediction enabled,
each completed utterance is scored in the background while the reply starts. The
default silence threshold is 0.3 seconds. A completed estimate can inform the next
reply, with its original utterance and timestamp retained. A fresh
result is atomically published to the dashboard's existing
`data/voice_latest.json` contract (`VEXO_VOICE_JSON` can override that path).
Raw pain scores remain separate from the dashboard's 0–100 deviation index.
Moderate/high events retain timestamp, score and transcript in
`vexo_voice_module/data/events.jsonl`. Audio stays in memory; Deepgram receives
the audio stream and Gemini receives conversation text and enabled estimates.
Changing the toggle starts a fresh conversation.

The service uses browser microphone input and binds to localhost. It uses the
existing dashboard's model-output contract without opening another Bluetooth
connection. Server-side authentication and same-origin WebSocket checks protect
the local conversation endpoint.

Checks:

```sh
python -m unittest discover -s tests -p 'test_voice_routing.py'
```
