import asyncio
import json
import sys
from .conversation import PROMPT, think_settings

AGENT_URL = 'wss://agent.deepgram.com/v1/agent/converse'
SAMPLE_RATE = 24000
BLOCK_FRAMES = 480
DEFAULT_PROMPT = PROMPT


def settings(prompt=DEFAULT_PROMPT):
    return {
        'type': 'Settings',
        'audio': {
            'input': {'encoding': 'linear16', 'sample_rate': SAMPLE_RATE},
            'output': {'encoding': 'linear16', 'sample_rate': SAMPLE_RATE, 'container': 'none'},
        },
        'agent': {
            'language': 'en',
            'listen': {'provider': {'type': 'deepgram', 'model': 'nova-3'}},
            'think': think_settings(prompt),
            'speak': {'provider': {'type': 'deepgram', 'version': 'v2', 'model': 'flux-kit-en'}},
        },
    }


def check_event(event):
    if event.get('type') == 'Error':
        # Report the error code, keeping credentials and echoed payloads private.
        raise RuntimeError(f"Deepgram error: {event.get('code', 'UNKNOWN')}")
    if event.get('type') == 'Warning':
        print(f"Deepgram warning: {event.get('code', 'UNKNOWN')}", file=sys.stderr)


async def wait_event(socket, expected):
    async def receive():
        while True:
            message = await socket.recv()
            if isinstance(message, bytes):
                raise RuntimeError(f'Unexpected audio while waiting for {expected}.')
            event = json.loads(message)
            check_event(event)
            if event.get('type') == expected:
                return
    await asyncio.wait_for(receive(), timeout=30)


async def configure(socket, configuration):
    await wait_event(socket, 'Welcome')
    await socket.send(json.dumps(configuration))
    await wait_event(socket, 'SettingsApplied')

