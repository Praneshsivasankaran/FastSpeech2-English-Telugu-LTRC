"""Behavioral split tests; no model, audio download or accelerator needed.

Run: python -m unittest discover -s tests -p test_splits.py
"""
import copy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tts.preprocessing import canonical, split_rows


def row(name, text=None, audio=None, seconds=1.0, speaker=0):
    return {'id':name, 'text':text if text is not None else f'Unique transcript {name}',
            'audio_sha256':audio if audio is not None else f'audio-{name}',
            'duration_seconds':seconds, 'speaker_id':speaker}


def ids(parts):
    return {split: [r['id'] for r in rows] for split, rows in parts.items()}


class SplitTests(unittest.TestCase):
    def test_seed_reproducibility_is_independent_of_input_order(self):
        original = [row(f'{i:03d}') for i in range(100)]
        permuted = copy.deepcopy(original)
        random.Random(91).shuffle(permuted)
        first, totals = split_rows(original, seed=47, train_seconds=40, val_seconds=20)
        again, again_totals = split_rows(permuted, seed=47, train_seconds=40, val_seconds=20)
        self.assertEqual(ids(first), ids(again))
        self.assertEqual(totals, again_totals)
        self.assertEqual(totals, {'train':40., 'val':20., 'unused':40.})
        self.assertEqual(original, [row(f'{i:03d}') for i in range(100)])

    def test_duplicate_audio_and_normalized_text_form_transitive_groups(self):
        source = [row('link-a', text='  CAFÉ   entry ', audio='audio-shared'),
                  row('link-b', text='cafe\u0301 entry', audio='audio-other'),
                  row('link-c', text='A different transcript', audio='audio-other')]
        source += [row(f'normal-{i:02d}') for i in range(40)]
        for seed in range(10):
            parts, _ = split_rows(source, seed=seed, train_seconds=20, val_seconds=10)
            owner = {r['id']:name for name, rows in parts.items() for r in rows}
            self.assertEqual(len(owner), len(source))
            self.assertEqual(owner['link-a'], owner['link-b'])
            self.assertEqual(owner['link-b'], owner['link-c'])
            names = list(parts)
            for i,left in enumerate(names):
                for right in names[i+1:]:
                    self.assertTrue({canonical(r['text']) for r in parts[left]}.isdisjoint(
                                    {canonical(r['text']) for r in parts[right]}))
                    self.assertTrue({r['audio_sha256'] for r in parts[left]}.isdisjoint(
                                    {r['audio_sha256'] for r in parts[right]}))

    def test_large_duplicate_group_cannot_overshoot_duration_limit(self):
        # Regression: the old allocator put this entire 10-second group into
        # the 5-second training budget at seed 1, silently returning 10 seconds.
        source = [row(str(i), text='duplicate', seconds=1) for i in range(10)]
        source += [row(str(i), seconds=1) for i in range(10,25)]
        parts, totals = split_rows(source, seed=1, train_seconds=5, val_seconds=2)
        self.assertLessEqual(totals['train'], 5 * 1.1)
        self.assertGreaterEqual(totals['train'], 5 * .9)
        self.assertLessEqual(totals['val'], 2 * 1.1)
        self.assertGreaterEqual(totals['val'], 2 * .9)
        unused = {r['id'] for r in parts['unused']}
        self.assertTrue({str(i) for i in range(10)} <= unused)
        self.assertEqual(sum(totals.values()),25.)

    def test_insufficient_training_duration_refuses(self):
        source = [row(str(i)) for i in range(5)]
        with self.assertRaisesRegex(ValueError, 'Insufficient train audio'):
            split_rows(source, train_seconds=10, val_seconds=2)

    def test_insufficient_validation_duration_refuses(self):
        source = [row('only', seconds=1)]
        # Seed allocation puts the sole row in validation; a compatible empty
        # train speaker set check may fail first, but no invalid split may return.
        with self.assertRaises(ValueError):
            split_rows(source, train_seconds=0, val_seconds=10)

    def test_validation_speaker_absent_from_training_refuses(self):
        source = [row('a', seconds=10, speaker=0), row('b', seconds=10, speaker=1)]
        with self.assertRaisesRegex(ValueError, 'Validation speakers absent from training'):
            split_rows(source, seed=1, train_seconds=10, val_seconds=10)

    def test_duplicate_source_ids_refuse_even_when_content_differs(self):
        source = [row('same', text='One'), row('same', text='Two')]
        with self.assertRaisesRegex(ValueError, 'Duplicate source IDs'):
            split_rows(source, train_seconds=1, val_seconds=1)


if __name__ == '__main__':
    unittest.main()
