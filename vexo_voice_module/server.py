"""Local speech service: Deepgram STT → Gemini → streamed TTS."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import secrets
import uuid
from urllib.parse import urlencode
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from websockets.asyncio.client import connect
from .credentials import load_api_key
from .agent import AGENT_URL, configure, settings
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

class AudioRing:
    def __init__(self):
        self.data = bytearray()
        self.offset = 0

    def append(self, data):
        self.data.extend(data)
        excess = len(self.data) - RATE * 2 * 120
        if excess > 0:
            self.data = self.data[excess:]
            self.offset += excess

    def segment(self, start, end):
        first = max(0, int(start * RATE) * 2)
        last = int(end * RATE) * 2
        if first < self.offset or last > self.offset + len(self.data):
            raise ValueError('Speech segment is outside the audio buffer')
        return bytes(self.data[first-self.offset:last-self.offset])


class Session:
    def __init__(self, client, prediction, pause, model, http):
        self.client, self.prediction, self.pause = client, prediction, pause
        self.model, self.http = model, http
        self.ring = AudioRing()
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

    async def interrupt(self):
        self.generation += 1
        if self.reply_task:
            self.reply_task.cancel()
            await asyncio.gather(self.reply_task, return_exceptions=True)
            self.reply_task = None
        await self.send({'type': 'interrupt'})

    async def finish_phrase(self):
        if not self.pending:
            return
        parts, self.pending = self.pending, []
        text = ' '.join(p['text'] for p in parts).strip()
        start, end = parts[0]['start'], parts[-1]['end']
        pcm = self.ring.segment(max(0, start - .1), end)
        await self.send({'type': 'transcript', 'role': 'user', 'text': text})
        self.jobs.put_nowait((self.generation, text, pcm))

    async def receive_stt(self, stt):
        async for message in stt:
            event = json.loads(message)
            kind = event.get('type')
            if kind == 'SpeechStarted':
                self.awaiting_speech_confirmation = True
            elif kind == 'Results':
                alt = event['channel']['alternatives'][0]
                text = alt.get('transcript', '').strip()
                if text and self.awaiting_speech_confirmation:
                    self.awaiting_speech_confirmation = False
                    await self.interrupt()
                if text and event.get('is_final'):
                    words = alt.get('words', [])
                    start = words[0]['start'] if words else event['start']
                    end = words[-1]['end'] if words else event['start'] + event['duration']
                    self.pending.append({'text': text, 'start': start, 'end': end})
                if event.get('speech_final'):
                    await self.finish_phrase()
            elif kind == 'UtteranceEnd':
                await self.finish_phrase()
            elif kind == 'Error':
                raise RuntimeError('Speech transcription failed')
        raise RuntimeError('Speech connection closed')

    async def receive_phone(self, stt):
        while True:
            message = await self.client.receive()
            if message['type'] == 'websocket.disconnect':
                return
            pcm = message.get('bytes')
            if pcm is not None:
                if len(pcm) > RATE * 2 or len(pcm) % 2:
                    raise ValueError('Invalid microphone audio')
                self.ring.append(pcm)
                await stt.send(pcm)
            elif message.get('text'):
                command = json.loads(message['text'])
                if command.get('type') == 'stop':
                    return

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

    async def process_phrases(self):
        while True:
            generation, text, pcm = await self.jobs.get()
            previous_analysis = self.latest_analysis if self.prediction else None
            self.history.append({'role': 'user', 'content': text})
            if generation == self.generation:
                if self.reply_task:
                    self.reply_task.cancel()
                    await asyncio.gather(self.reply_task, return_exceptions=True)
                self.reply_task = asyncio.create_task(self.reply(generation, previous_analysis))
            if self.prediction:
                task = asyncio.create_task(self.analyze_phrase(text, pcm))
                self.analysis_tasks.add(task)
                task.add_done_callback(self.analysis_finished)

    async def reply(self, generation, analysis):
        try:
            await self.send({'type': 'status', 'text': 'Thinking'})
            system = PROMPT
            if analysis:
                system += ('\nMost recent completed acoustic estimate (for its named earlier utterance): ' + json.dumps({**analysis, 'event_saved': analysis['level'] in ('moderate', 'high'), 'outside_label_range': not 0 <= analysis['score'] <= 10}) +
                           '\nThis estimate uses Lamita’s calibrated model averaged across chunks of at most three seconds.')
            configuration = settings(system)
            history = self.history[-20:]
            if len(history) > 1:
                configuration['agent']['context'] = {'messages': [
                    {'type': 'History', **message} for message in history[:-1]]}
            key = load_api_key()
            async with connect(AGENT_URL, additional_headers={'Authorization': 'Token ' + key},
                               open_timeout=15, close_timeout=3, compression=None) as agent:
                await configure(agent, configuration)
                await agent.send(json.dumps({'type': 'InjectUserMessage', 'content': history[-1]['content']}))
                text_parts = []
                reply_id = str(uuid.uuid4())
                async def receive_reply():
                    async for message in agent:
                        if generation != self.generation:
                            return
                        if isinstance(message, bytes):
                            await self.send(message)
                            continue
                        event = json.loads(message)
                        kind = event.get('type')
                        if kind == 'ConversationText' and event.get('role') == 'assistant':
                            text = event.get('content', '')
                            if text:
                                text_parts.append(text)
                                await self.send({'type': 'transcript', 'role': 'assistant', 'text': text, 'turn_id': reply_id})
                        elif kind == 'AgentStartedSpeaking':
                            await self.send({'type': 'status', 'text': 'Speaking'})
                        elif kind == 'AgentAudioDone':
                            if text_parts:
                                self.history.append({'role': 'assistant', 'content': ' '.join(text_parts)})
                            await self.send({'type': 'audio_done'})
                            return
                        elif kind in ('Error', 'InjectionRefused'):
                            raise RuntimeError('Conversation model error: ' + event.get('code', kind))
                    raise RuntimeError('Conversation model connection closed')
                receiver = asyncio.create_task(receive_reply())
                heartbeat = asyncio.create_task(self.keepalive(agent))
                try:
                    done, _ = await asyncio.wait(
                        [receiver, heartbeat], timeout=60,
                        return_when=asyncio.FIRST_COMPLETED)
                    if not done:
                        raise TimeoutError('Conversation reply timed out')
                    for task in done:
                        task.result()
                finally:
                    for task in (receiver, heartbeat):
                        task.cancel()
                    await asyncio.gather(receiver, heartbeat, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.send({'type': 'error', 'text': safe_error(error)})

    async def keepalive(self, stt):
        while True:
            await asyncio.sleep(5)
            await stt.send(json.dumps({'type': 'KeepAlive'}))

    async def run(self, key):
        url = 'wss://api.deepgram.com/v1/listen?' + urlencode({
            'model': 'nova-3', 'language': 'en', 'encoding': 'linear16', 'sample_rate': RATE,
            'channels': 1, 'interim_results': 'true', 'vad_events': 'true',
            'endpointing': self.pause, 'utterance_end_ms': max(1000, self.pause), 'punctuate': 'true'})
        async with connect(url, additional_headers={'Authorization': 'Token ' + key},
                           open_timeout=15, close_timeout=3) as stt:
            await self.send({'type': 'ready'})
            tasks = [asyncio.create_task(f(stt)) for f in
                     (self.receive_phone, self.receive_stt, self.keepalive)]
            tasks.append(asyncio.create_task(self.process_phrases()))
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                if self.reply_task:
                    tasks.append(self.reply_task)
                tasks.extend(self.analysis_tasks)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


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
        key = load_api_key()
        async with aiohttp.ClientSession(headers={'Authorization': 'Token ' + key},
             timeout=aiohttp.ClientTimeout(total=60)) as http:
            session = Session(client, config['prediction'], pause, app.state.model, http)
            await session.run(key)
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
