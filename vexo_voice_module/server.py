"""Local speech service: Gemini Live audio input, transcription and spoken responses."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import secrets
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .conversation import PROMPT
from .output import publish_voice
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'vexo_voice_module/data'
RATE = 24000


def load_pain_model():
    factory = os.environ.get('VEXO_PAIN_FACTORY', '').strip()
    if not factory:
        return None
    module, separator, name = factory.partition(':')
    if not separator:
        raise ValueError('VEXO_PAIN_FACTORY must be module:factory')
    return getattr(importlib.import_module(module), name)()


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True, exist_ok=True)
    token_path = DATA / 'connection-key.txt'
    if not token_path.exists():
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(secrets.token_urlsafe(24))
    app.state.token = token_path.read_text().strip()
    app.state.model = await asyncio.to_thread(load_pain_model)
    yield

app = FastAPI(lifespan=lifespan)
WEB = Path(__file__).resolve().parent / 'web'
app.mount('/static', StaticFiles(directory=WEB), name='static')

@app.middleware('http')
async def local_only(request: Request, call_next):
    if request.url.hostname not in ('localhost', '127.0.0.1', '::1'):
        return JSONResponse({'detail':'Use localhost for this demo'},status_code=403)
    return await call_next(request)

@app.get('/')
async def index():
    response = FileResponse(WEB / 'index.html')
    response.set_cookie('vexo_session',app.state.token,httponly=True,samesite='strict')
    return response

@app.get('/health')
async def health():
    return {'status':'ready','pain_model_ready':app.state.model is not None}

@app.get('/events')
async def events(request: Request):
    if not secrets.compare_digest(request.cookies.get('vexo_session',''),app.state.token):
        raise HTTPException(403)
    path = DATA / 'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()][-500:][::-1] if path.exists() else []



class Session:
    def __init__(self, client, prediction, pause, model, http):
        self.client, self.prediction, self.pause = client, prediction, pause
        self.model, self.http = model, http
        self.pending = []
        self.awaiting_speech_confirmation = True
        self.jobs = asyncio.Queue(maxsize=16)
        self.generation = 0
        self.reply_task = None
        self.history = []
        self.latest_analysis = None
        self.analysis_tasks = set()
        self.analysis_lock = asyncio.Lock()
        self.session_id = str(uuid.uuid4())
        self.send_lock = asyncio.Lock()

    async def send(self, payload):
        async with self.send_lock:
            if isinstance(payload, bytes):
                await self.client.send_bytes(payload)
            else:
                await self.client.send_json(payload)





    async def analyze_phrase(self, text, pcm):
        async with self.analysis_lock:
            analysis = await asyncio.to_thread(self.model.score_pcm, pcm)
            if not analysis:
                return
            event = {'id': str(uuid.uuid4()), 'timestamp': datetime.now(timezone.utc).isoformat(),
                     'transcript': text, 'session_id': self.session_id, **analysis}
            publish_voice(event)
            await self.send({'type': 'prediction', **event})
            if analysis['level'] in ('moderate', 'high'):
                path = DATA / 'events.jsonl'
                with path.open('a') as f:
                    f.write(json.dumps(event, allow_nan=False) + '\n')
                await self.send({'type': 'event', **event})
            self.latest_analysis = {**analysis, 'utterance_transcript': text,
                                    'timestamp': event['timestamp']}

    def analysis_finished(self, task):
        self.analysis_tasks.discard(task)
        if not task.cancelled():
            # Retrieve errors so a failed optional analysis cannot stop conversation.
            error = task.exception()
            if error:
                asyncio.create_task(self.report_analysis_error())

    async def report_analysis_error(self):
        try:
            await self.send({'type': 'analysis_error',
                             'text': 'Pain analysis unavailable for this utterance.'})
        except (RuntimeError, WebSocketDisconnect):
            pass






def safe_error(error):
    if isinstance(error, (RuntimeError, ValueError)):
        return str(error)
    return 'Connection interrupted. Check the server and try again.'


@app.websocket('/conversation')
async def conversation(client: WebSocket):
    origin = client.headers.get('origin')
    expected_origin = ('https' if client.url.scheme == 'wss' else 'http') + '://' + client.headers.get('host', '')
    if client.url.hostname not in ('localhost', '127.0.0.1', '::1') or (origin and origin != expected_origin):
        await client.close(code=1008)
        return
    supplied = client.headers.get('authorization', '').removeprefix('Bearer ') or client.cookies.get('vexo_session', '')
    if not secrets.compare_digest(supplied, app.state.token):
        await client.close(code=1008)
        return
    await client.accept()
    try:
        config = await asyncio.wait_for(client.receive_json(), timeout=10)
        if type(config.get('prediction')) is not bool:
            raise ValueError('Choose the pain prediction mode')
        if config['prediction'] and app.state.model is None:
            raise ValueError('Pain model unavailable. Restore the model files and restart the app.')
        pause = int(config.get('pause_ms', 300))
        if not 200 <= pause <= 3000:
            raise ValueError('Pause duration must be between 0.2 and 3 seconds')
        from .gemini import GeminiDirectSession
        session = GeminiDirectSession(client, config['prediction'], pause, app.state.model, None)
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception as error:
        try:
            await client.send_json({'type': 'error', 'text': safe_error(error)})
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        try:
            await client.close()
        except (RuntimeError, WebSocketDisconnect):
            pass
