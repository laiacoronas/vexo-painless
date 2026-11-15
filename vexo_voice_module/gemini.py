"""Persistent Gemini Live replies over the existing transcript/pain routing."""
import asyncio
import json
import os
import shlex
from pathlib import Path
import uuid
from google import genai
from .server import Session, ROOT
from .conversation import PROMPT


def api_key():
    values={}
    path=Path(os.environ.get('VEXO_ENV_FILE', ROOT/'.env'))
    if path.exists():
        for line in path.read_text().splitlines():
            name,sep,value=line.strip().removeprefix('export ').partition('=')
            if sep and name.strip() in ('GEMINI_API_KEY','GOOGLE_API_KEY'):
                parts=shlex.split(value,comments=True)
                values[name.strip()]=parts[0] if len(parts)==1 else ''
    for name in ('GEMINI_API_KEY','GOOGLE_API_KEY'):
        key=os.environ.get(name,'').strip() or values.get(name,'')
        if key:return key
    raise ValueError('Set GEMINI_API_KEY in the project .env file.')


class GeminiDirectSession(Session):
    """Google handles audio input, transcription, turn detection and audio output."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.live=None
        self.reply_id=str(uuid.uuid4())
        self.user_id=str(uuid.uuid4())
        self.user_text=[]
        self.input_audio=bytearray()
        self.response_started=False
        self.input_started=False

    async def analyze_input(self, text, pcm):
        try:
            await self.analyze_phrase(text, pcm)
            if self.latest_analysis:
                await self.live.send_client_content(turns={'role':'user','parts':[{
                    'text':'Application context: completed background acoustic estimate for its named utterance: '+json.dumps(self.latest_analysis)
                }]},turn_complete=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.report_analysis_error()

    async def finish_input(self):
        text=''.join(self.user_text).strip()
        pcm=bytes(self.input_audio)
        self.user_text=[]
        self.input_audio.clear()
        self.input_started=False
        self.user_id=str(uuid.uuid4())
        if self.prediction and text and pcm:
            task=asyncio.create_task(self.analyze_input(text,pcm))
            self.analysis_tasks.add(task)
            task.add_done_callback(self.analysis_tasks.discard)

    async def receive_phone(self):
        while True:
            message=await self.client.receive()
            if message['type']=='websocket.disconnect':return
            pcm=message.get('bytes')
            if pcm is not None:
                if len(pcm)>48000 or len(pcm)%2:raise ValueError('Invalid microphone audio')
                # Bound the rolling audio and preserve onset before transcription arrives.
                self.input_audio.extend(pcm)
                limit=24000*2*(120 if self.input_started else 2)
                if len(self.input_audio)>limit:del self.input_audio[:-limit]
                await self.live.send_realtime_input(audio={'data':pcm,'mime_type':'audio/pcm;rate=24000'})
            elif message.get('text') and json.loads(message['text']).get('type')=='stop':return

    async def receive_live(self):
        while True:
            async for message in self.live.receive():
                content=message.server_content
                if not content:continue
                if content.interrupted:
                    self.response_started=False
                    self.reply_id=str(uuid.uuid4())
                    await self.send({'type':'interrupt'})
                transcription=content.input_transcription
                if transcription and transcription.text:
                    self.input_started=True
                    self.user_text.append(transcription.text)
                    await self.send({'type':'transcript','role':'user','text':transcription.text,
                                     'turn_id':self.user_id,'append_exact':True})
                if content.model_turn or content.output_transcription:
                    if not self.response_started:
                        self.response_started=True
                        await self.finish_input()
                        await self.send({'type':'status','text':'Speaking'})
                    if content.output_transcription and content.output_transcription.text:
                        await self.send({'type':'transcript','role':'assistant','text':content.output_transcription.text,
                                         'turn_id':self.reply_id,'append_exact':True})
                    if content.model_turn:
                        for part in content.model_turn.parts or []:
                            if part.inline_data and part.inline_data.data:await self.send(part.inline_data.data)
                if content.turn_complete:
                    if self.user_text:await self.finish_input()
                    self.response_started=False
                    self.reply_id=str(uuid.uuid4())
                    await self.send({'type':'audio_done'})

    async def run(self,key=None):
        client=genai.Client(api_key=api_key(),http_options={'api_version':'v1beta'})
        try:
            async with client.aio.live.connect(model='gemini-3.8-live',config={
                'response_modalities':['AUDIO'],'system_instruction':PROMPT,
                'input_audio_transcription':{},'output_audio_transcription':{},
                'realtime_input_config':{'automatic_activity_detection':{'silence_duration_ms':self.pause}},
                'speech_config':{'voice_config':{'prebuilt_voice_config':{'voice_name':'Charon'}}},
            }) as live:
                self.live=live
                await self.send({'type':'ready'})
                tasks=[asyncio.create_task(self.receive_live()),asyncio.create_task(self.receive_phone())]
                try:
                    done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                    for task in done:task.result()
                finally:
                    tasks.extend(self.analysis_tasks)
                    for task in tasks:task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
        finally:await client.aio.aclose()
