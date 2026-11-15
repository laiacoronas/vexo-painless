"""Build participant-separated, chronological TAME Pain examples."""
import argparse
import bisect
import collections
import csv
import hashlib
import json
import math
import random
from pathlib import Path


def numeric(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def prepare(metadata, seed=42, references=3):
    with open(metadata, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    required = {'PID', 'COND', 'UTTNUM', 'UTTID', 'REVISED PAIN', 'ACTION LABEL', 'NOTES'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Metadata must contain {sorted(required)}')
    people = sorted({r['PID'] for r in rows})
    if len(people) != 51:
        raise ValueError(f'Expected 51 participants; found {len(people)}. Review split.')
    random.Random(seed).shuffle(people)
    splits = {'train': people[:35], 'validation': people[35:43], 'test': people[43:]}
    membership = {p: s for s, ps in splits.items() for p in ps}
    sessions = collections.defaultdict(list)
    for r in rows:
        sessions[(r['PID'], r['COND'])].append(r)
    examples, excluded = [], collections.Counter()
    score_counts = collections.Counter()
    for (pid, condition), session in sorted(sessions.items()):
        session.sort(key=lambda r: int(r['UTTNUM']))
        reports = {int(r['UTTNUM']): r for r in session if int(r['UTTID']) == 99999}
        report_times = sorted(reports)
        blocks = collections.defaultdict(list)
        for r in session:
            score = numeric(r['REVISED PAIN'])
            if score is not None:
                score_counts[str(score)] += 1
            if int(r['UTTID']) == 99999:
                excluded['spoken_rating'] += 1
                continue
            if score is None or not 1 <= score <= 10:
                excluded['missing_or_invalid_score'] += 1
                continue
            if numeric(r['ACTION LABEL']) != 0:
                excluded['audio_quality'] += 1
                continue
            notes = r['NOTES'].lower()
            if 'copied' in notes or 'no pain rating' in notes:
                excluded['copied_rating'] += 1
                continue
            utt = int(r['UTTNUM'])
            idx = bisect.bisect_left(report_times, utt)
            if idx == len(report_times):
                excluded['no_following_report'] += 1
                continue
            report = report_times[idx]
            if numeric(reports[report]['REVISED PAIN']) != score:
                excluded['report_label_mismatch'] += 1
                continue
            clip = f"{pid}.{condition}.{utt}.{int(r['UTTID'])}"
            blocks[report].append({'clip': clip, 'score': score, 'block': report, 'utterance': utt})
        times = sorted(blocks)
        if len(times) <= references:
            excluded['session_insufficient_blocks'] += 1
            continue
        # One neutral sentence from each of the first three usable rating blocks.
        # Every sentence in those blocks is reserved from query evaluation.
        refs = [blocks[t][0] for t in times[:references]]
        for t in times[references:]:
            for query in blocks[t]:
                assert all(r['block'] < query['utterance'] for r in refs)
                examples.append({'participant': pid, 'condition': condition,
                                 'split': membership[pid], 'query': query, 'references': refs})
    counts = {}
    for split, ids in splits.items():
        subset = [e for e in examples if e['split'] == split]
        present = {e['participant'] for e in subset}
        if present != set(ids):
            raise ValueError(f'{split}: participants lacking eligible queries: {set(ids)-present}')
        counts[split] = {'participants': len(present), 'queries': len(subset)}
    return {'seed': seed, 'reference_count': references, 'splits': splits,
            'metadata_sha256': hashlib.sha256(Path(metadata).read_bytes()).hexdigest(),
            'reference_policy': 'first usable sentence in each of first three usable rating blocks within each PID/COND session',
            'examples': examples,
            'audit': {'raw_rows': len(rows), 'score_counts': dict(score_counts),
                      'exclusions': dict(excluded), 'splits': counts}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='data/tame/meta_audio.csv')
    parser.add_argument('--output', default='runs/ast_reference/manifest.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.metadata, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'{path} exists; use a new output path to preserve split history')
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['audit'], indent=2))
