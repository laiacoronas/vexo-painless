import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from vexo_demo_data import (read_demo_events, summaries, live_movement, live_voice,
                            demo_movement, demo_signal, live_signal, save_csv_row, choose_file)

class DataAdapterTests(unittest.TestCase):
    def test_synthetic_41_events_and_patterns(self):
        events = read_demo_events()
        self.assertEqual(len(events), 41)
        self.assertEqual(summaries(events), {'moderate':31, 'high':10, 'morning':16,
                                             'mon_thu':9, 'long_sitting':8})
        self.assertTrue(all(x['source']=='example_history' for x in events))

    def test_synthetic_movement_marked(self):
        payload=demo_movement('High')
        self.assertEqual(payload['source'], 'example-preview')
        self.assertEqual(payload['movement_status'], 'strong_change')
        self.assertIsNone(payload['pain_estimate'])

    def test_bluetooth_rejects_synthetic(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'movement_latest.json'
            path.write_text(json.dumps(demo_movement()))
            _,state,_=live_movement(path)
            self.assertEqual(state,'simulated_source')

    def test_live_only_fresh_connected_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'movement_latest.json'
            payload=demo_movement();payload.update(source='vexo-band',sample_count=125)
            path.write_text(json.dumps(payload))
            _,state,_=live_movement(path)
            self.assertEqual(state,'connected')
            payload['timestamp']=(datetime.now(timezone.utc)-timedelta(minutes=10)).isoformat()
            path.write_text(json.dumps(payload))
            _,state,_=live_movement(path)
            self.assertEqual(state,'stale')

    def test_disconnected_is_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'movement_latest.json'
            payload=demo_movement();payload.update(source='vexo-band',connection_status='disconnected',movement_deviation_index=None)
            path.write_text(json.dumps(payload))
            _,state,_=live_movement(path)
            self.assertEqual(state,'disconnected')

    def test_voice_stays_pending_until_fresh_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'voice_latest.json'
            self.assertEqual(live_voice(path)[1],'pending')
            data=dict(timestamp=datetime.now(timezone.utc).isoformat(),source='voice-model',
                      voice_status='ready',voice_level='normal',voice_deviation_index=None)
            path.write_text(json.dumps(data))
            value,state=live_voice(path)
            self.assertEqual(state,'ready')
            self.assertIsNone(value['index'])

    def test_fast_signal_adapter_and_rejection(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'movement_signal.json'
            ts = datetime.now(timezone.utc)
            payload = dict(timestamp=ts.isoformat(), source='vexo-band',
                           connection_status='connected',
                           samples=[{'t':ts.timestamp(), 'accel_magnitude_g':1.12,
                                     'gyro_magnitude_dps':26.1}])
            path.write_text(json.dumps(payload))
            points,state=live_signal(path)
            self.assertEqual(state, 'connected')
            self.assertEqual(len(points),1)
            payload['source']='synthetic'
            path.write_text(json.dumps(payload))
            self.assertEqual(live_signal(path)[1], 'waiting')
            payload['source']='vexo-band';payload['connection_status']='disconnected'
            path.write_text(json.dumps(payload))
            self.assertEqual(live_signal(path)[1], 'disconnected')
        self.assertEqual(len(demo_signal()), 220)

    def test_csv_checkin_persist(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'checkins.csv'
            save_csv_row(p,{'timestamp':'now','self_reported_pain_0_10':5})
            with p.open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['self_reported_pain_0_10'],'5')

if __name__=='__main__':unittest.main()
