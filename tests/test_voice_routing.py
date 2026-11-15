import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from vexo_voice_module.server import Session

class ModeTests(unittest.IsolatedAsyncioTestCase):
    async def process(self, enabled):
        client = AsyncMock()
        model = Mock()
        model.score_pcm.return_value = {'score':8.2, 'level':'high'}
        session = Session(client, enabled, 1000, model, None)
        session.reply = AsyncMock()
        await session.jobs.put((0,'Hello', b'pcm'))
        publisher = patch('vexo_voice_module.server.publish_voice')
        publisher.start()
        self.addCleanup(publisher.stop)
        task = asyncio.create_task(session.process_phrases())
        for _ in range(100):
            if session.reply.await_count: break
            await asyncio.sleep(.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if session.reply_task: await session.reply_task
        await asyncio.gather(*session.analysis_tasks)
        return session, model, client

    async def test_off_skips_model_and_event_storage(self):
        with tempfile.TemporaryDirectory() as directory, patch('vexo_voice_module.server.DATA',Path(directory)):
            session, model, client = await self.process(False)
            model.score_pcm.assert_not_called()
            session.reply.assert_awaited_once_with(0,None)
            self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_on_replies_first_and_records_background_result_without_audio(self):
        with tempfile.TemporaryDirectory() as directory, patch('vexo_voice_module.server.DATA',Path(directory)):
            session, model, client = await self.process(True)
            model.score_pcm.assert_called_once_with(b'pcm')
            session.reply.assert_awaited_once_with(0,None)
            self.assertEqual(session.latest_analysis['score'],8.2)
            files=list(Path(directory).iterdir())
            self.assertEqual([p.name for p in files], ['events.jsonl'])
            event=json.loads(files[0].read_text())
            self.assertEqual(event['transcript'],'Hello')
            self.assertEqual(event['score'],8.2)
            self.assertIn('timestamp',event)

    async def test_interrupt_cancels_response_and_clears_playback(self):
        client=AsyncMock()
        session=Session(client,False,1000,None,None)
        session.reply_task=asyncio.create_task(asyncio.sleep(60))
        await session.interrupt()
        self.assertEqual(session.generation,1)
        self.assertIsNone(session.reply_task)
        client.send_json.assert_awaited_once_with({'type':'interrupt'})

if __name__=='__main__': unittest.main()

class ReplyHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_keeps_agent_alive_until_audio_finishes(self):
        heartbeat_sent = asyncio.Event()
        heartbeat_stopped = asyncio.Event()
        class Agent:
            async def send(self, message):
                if json.loads(message)['type'] == 'KeepAlive':
                    heartbeat_sent.set()
            def __aiter__(self): return self.messages()
            async def messages(self):
                await asyncio.wait_for(heartbeat_sent.wait(), 1)
                yield b'\x00\x00'
                yield json.dumps({'type':'AgentAudioDone'})
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        agent = Agent()
        session = Session(AsyncMock(), False, 1000, None, None)
        session.history = [{'role':'user','content':'Hello'}]
        async def heartbeat(connection):
            try:
                await connection.send(json.dumps({'type':'KeepAlive'}))
                await asyncio.Event().wait()
            finally:
                heartbeat_stopped.set()
        session.keepalive = heartbeat
        with patch('vexo_voice_module.server.connect',return_value=agent), \
             patch('vexo_voice_module.server.configure',new_callable=AsyncMock), \
             patch('vexo_voice_module.server.load_api_key',return_value='test-placeholder'), \
             patch('vexo_voice_module.server.settings',return_value={}):
            await session.reply(0,None)
        session.client.send_bytes.assert_awaited_once_with(b'\x00\x00')
        self.assertIn({'type':'audio_done'}, [call.args[0] for call in session.client.send_json.await_args_list])
        self.assertTrue(heartbeat_stopped.is_set())

class InterruptionConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_noise_does_not_cancel_reply_but_recognized_words_do(self):
        session = Session(AsyncMock(),False,1000,None,None)
        session.interrupt = AsyncMock()
        async def events():
            yield json.dumps({'type':'SpeechStarted'})
            yield json.dumps({'type':'Results','channel':{'alternatives':[{'transcript':''}]}})
            session.interrupt.assert_not_awaited()
            yield json.dumps({'type':'Results','channel':{'alternatives':[{'transcript':'Wait'}]}})
            yield json.dumps({'type':'Results','channel':{'alternatives':[{'transcript':'Wait please'}]}})
        with self.assertRaisesRegex(RuntimeError,'Speech connection closed'):
            await session.receive_stt(events())
        session.interrupt.assert_awaited_once()

class OutputTests(unittest.TestCase):
    def test_prediction_reaches_existing_voice_placeholder(self):
        import sys
        from datetime import datetime, timezone
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
        from vexo_demo_data import live_voice
        from vexo_voice_module.output import publish_voice
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'voice_latest.json'
            publish_voice({'score':8.2,'level':'high','timestamp':datetime.now(timezone.utc).isoformat(),'transcript':'Hello'},path)
            result,status=live_voice(path)
            self.assertEqual(status,'ready')
            self.assertEqual(result['score'],8.2)
            self.assertEqual(result['level'],'high')
            self.assertEqual(result['transcript'],'Hello')
            self.assertIsNone(result['index'])

class BackgroundAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_analysis_does_not_block_reply(self):
        release=asyncio.Event()
        started=asyncio.Event()
        session=Session(AsyncMock(),True,300,Mock(),None)
        session.reply=AsyncMock()
        async def delayed(*args):
            started.set()
            await release.wait()
            return None
        with patch('vexo_voice_module.server.asyncio.to_thread',side_effect=delayed):
            worker=asyncio.create_task(session.process_phrases())
            await session.jobs.put((0,'Hello',b'pcm'))
            await asyncio.wait_for(started.wait(),1)
            session.reply.assert_awaited_once_with(0,None)
            self.assertFalse(release.is_set())
            release.set()
            await asyncio.gather(*session.analysis_tasks)
            worker.cancel()
            await asyncio.gather(worker,return_exceptions=True)
