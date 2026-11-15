import importlib.util
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("vexo_ble_server", ROOT / "ble_bridge" / "server.py")
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


class WebBluetoothBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        data = Path(self.tmp.name)
        patches = [patch.object(bridge, 'DATA', data),
                   patch.object(bridge, 'LATEST', data/'movement_latest.json'),
                   patch.object(bridge, 'SIGNAL', data/'movement_signal.json'),
                   patch.object(bridge, 'HISTORY', data/'movement_windows.csv'),
                   patch.object(bridge, 'BASELINE', data/'movement_baseline.json')]
        for p in patches:
            p.start(); self.addCleanup(p.stop)
        self.p = bridge.StreamProcessor(rate=25, seconds=5)

    def test_starts_not_connected_and_never_fake_live(self):
        record = json.loads(bridge.LATEST.read_text())
        self.assertEqual(record['source'], 'vexo-band')
        self.assertEqual(record['connection_status'], 'disconnected')
        self.assertIsNone(record['movement_deviation_index'])

    def test_web_bluetooth_samples_and_preview(self):
        self.p.transition('connected')
        base = time.time()
        for batch_index in range(5):
            motion = []
            for i in range(25):
                j = batch_index*25+i
                motion.append({'time_ms':(base+j/25)*1000,
                               'ax':0.01*math.sin(j), 'ay':0.015*math.cos(j), 'az':1.0,
                               'gx':1.0, 'gy':0, 'gz':0})
            self.assertEqual(self.p.ingest({'motion':motion,'activity':[{'steps':100,'cadence':0,'state':'still'}]}),25)
        self.assertEqual(len(self.p.engine.samples),125)
        self.p.window_start -= 5.1
        self.p.tick()
        r=json.loads(bridge.LATEST.read_text())
        s=json.loads(bridge.SIGNAL.read_text())
        self.assertEqual(r['movement_status'],'calibrating')
        self.assertEqual(r['sample_count'],125)
        self.assertEqual(r['source'],'vexo-band')
        self.assertGreaterEqual(len(s['samples']),15)
        self.assertTrue(bridge.BASELINE.exists())

    def test_offline_never_reuses_score(self):
        self.p.transition('connected')
        self.p.transition('reconnecting')
        d=json.loads(bridge.LATEST.read_text())
        self.assertEqual(d['connection_status'],'disconnected')
        self.assertIsNone(d['movement_deviation_index'])
        self.p.transition('connected')
        self.p.last_post = time.monotonic() - 20
        self.p.tick()
        self.assertEqual(self.p.state,'disconnected')

    def test_no_data_connected_is_not_motion_zero(self):
        self.p.transition('connected')
        self.p.last_imu = time.monotonic() - 5
        self.p.tick()
        d=json.loads(bridge.LATEST.read_text())
        self.assertEqual(d['movement_status'],'insufficient_data')
        self.assertIsNone(d['features']['accel_std_g'])
        self.assertIsNone(d['movement_deviation_index'])

    def test_reject_invalid_batch(self):
        self.p.transition('connected')
        with self.assertRaises(ValueError):
            self.p.ingest({'motion':[{}]*201})
        with self.assertRaises(ValueError):
            self.p.transition('bogus')
        self.assertEqual(self.p.ingest({'motion':[{'ax':float('nan')}], 'activity':[]}),0)

    def test_keep_alive(self):
        self.p.transition('connected')
        self.p.last_post = time.monotonic() - 8
        self.p.transition('connected')
        self.assertLess(time.monotonic()-self.p.last_post,1)

if __name__ == '__main__':
    unittest.main()

# The static page and API are tested through the real local HTTP handler;
# a tiny placeholder JS file stands in for the external SDK during unit tests.
class LocalHttpTests(unittest.TestCase):
    def test_local_connector_and_post_pipeline(self):
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer
        with tempfile.TemporaryDirectory() as temp:
            scratch=Path(temp)
            kit=scratch/'vexo-open-kit'
            (kit/'js'/'src').mkdir(parents=True)
            (kit/'js'/'src'/'index.js').write_text('export const Band = {};')
            patches=[patch.object(bridge,'DATA',scratch/'data'),
                     patch.object(bridge,'LATEST',scratch/'data'/'movement_latest.json'),
                     patch.object(bridge,'SIGNAL',scratch/'data'/'movement_signal.json'),
                     patch.object(bridge,'BASELINE',scratch/'data'/'movement_baseline.json'),
                     patch.object(bridge,'HISTORY',scratch/'data'/'movement_windows.csv')]
            for p in patches:p.start()
            try:
                state=bridge.StreamProcessor()
                server=ThreadingHTTPServer(('127.0.0.1',0),bridge.make_handler(state,kit))
                thread=threading.Thread(target=server.serve_forever,daemon=True)
                thread.start()
                addr=f'http://127.0.0.1:{server.server_port}'
                try:
                    with urllib.request.urlopen(addr+'/ble_bridge/index.html') as r:
                        page=r.read().decode()
                    self.assertIn('Band.connect(79,{autoReconnect:true})',page)
                    with urllib.request.urlopen(addr+'/vexo-open-kit/js/src/index.js') as r:
                        self.assertIn('export const',r.read().decode())
                    req=urllib.request.Request(addr+'/api/status',data=b'{"status":"connected"}',
                            headers={'Content-Type':'application/json'},method='POST')
                    self.assertTrue(json.load(urllib.request.urlopen(req))['ok'])
                    req=urllib.request.Request(addr+'/api/push',data=json.dumps({'motion':[
                         {'ax':0,'ay':0,'az':1,'gx':3,'gy':0,'gz':0,'time_ms':time.time()*1000}],
                         'activity':[{'steps':159,'cadence':0,'state':'still'}]}).encode(),
                         headers={'Content-Type':'application/json'},method='POST')
                    self.assertEqual(json.load(urllib.request.urlopen(req))['accepted'],1)
                    self.assertEqual(state.engine.latest_steps,159)
                finally:
                    server.shutdown();server.server_close();thread.join(timeout=2)
            finally:
                for p in reversed(patches):p.stop()
