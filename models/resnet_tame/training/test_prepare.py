import csv
import tempfile
import unittest
from pathlib import Path

from prepare import prepare


class SplitTests(unittest.TestCase):
    def test_participants_and_rating_blocks_stay_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metadata.csv'
            fields = ['PID', 'COND', 'UTTNUM', 'UTTID', 'REVISED PAIN', 'ACTION LABEL', 'NOTES']
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for person in range(51):
                    for block in range(4):
                        for offset in (1, 2):
                            writer.writerow(dict(zip(fields, [f'p{person}', 'LC', block * 2 + offset,
                                                              10 if offset == 1 else 99999,
                                                              block + 1, 0, ''])))
            manifest = prepare(path)
            splits = [set(ids) for ids in manifest['splits'].values()]
            self.assertEqual([len(s) for s in splits], [35, 8, 8])
            self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
            self.assertEqual(len(manifest['examples']), 51)
            for e in manifest['examples']:
                self.assertEqual(e['query']['score'], 4)
                self.assertEqual([r['score'] for r in e['references']], [1, 2, 3])
                for r in e['references']:
                    self.assertLess(r['block'], e['query']['utterance'])
                    self.assertNotEqual(r['clip'], e['query']['clip'])
            self.assertEqual(manifest, prepare(path))


if __name__ == '__main__':
    unittest.main()
