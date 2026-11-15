import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from vexo_voice_module.gemini import GeminiDirectSession

class GeminiVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_transcript_and_audio_reach_browser(self):
        client=AsyncMock()
        session=GeminiDirectSession(client,False,300,None,None)
        content=NS(interrupted=False,input_transcription=NS(text='Hello'),
                   output_transcription=NS(text='Hi there.'),
                   model_turn=NS(parts=[NS(inline_data=NS(data=b'\x00\x00'))]),turn_complete=True)
        class Live:
            async def receive(self):
                yield NS(server_content=content)
                raise RuntimeError('test complete')
        session.live=Live()
        with self.assertRaisesRegex(RuntimeError,'test complete'):
            await session.receive_live()
        events=[c.args[0] for c in client.send_json.await_args_list]
        self.assertEqual([e['role'] for e in events if e['type']=='transcript'],['user','assistant'])
        client.send_bytes.assert_awaited_once_with(b'\x00\x00')
        self.assertIn({'type':'audio_done'},events)
        self.assertFalse(session.analysis_tasks)

    async def test_microphone_audio_goes_directly_to_gemini(self):
        client=AsyncMock()
        client.receive.side_effect=[{'type':'websocket.receive','bytes':b'\x00\x00'}, {'type':'websocket.disconnect'}]
        session=GeminiDirectSession(client,False,300,None,None)
        session.live=AsyncMock()
        await session.receive_phone()
        session.live.send_realtime_input.assert_awaited_once_with(audio={'data':b'\x00\x00','mime_type':'audio/pcm;rate=24000'})
