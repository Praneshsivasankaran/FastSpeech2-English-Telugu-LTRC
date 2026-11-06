"""Tests for the opt-in candidate; no checkpoint or final audio is modified."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tts import contextual_frontend, frontend


def word_phones(detail, word):
    return [phone for phone, number in zip(detail['eipa'], detail['word_num'])
            if detail['words'][number] == word and phone != '-']


class ContextualFrontendTests(unittest.TestCase):
    def test_generic_sentence_context_changes_verb_vowel(self):
        noun = contextual_frontend.phoneme_detail('His conduct was praised.', 'en')
        verb = contextual_frontend.phoneme_detail('They will conduct an experiment.', 'en')
        self.assertEqual(word_phones(noun, 'conduct'), ['k', 'ɑː', 'n', 'd', 'ʌ', 'k', 't'])
        self.assertEqual(word_phones(verb, 'conduct'), ['k', 'ə', 'n', 'd', 'ʌ', 'k', 't'])
        self.assertFalse(verb['contextual_frontend']['stress_markers_encoded'])

    def test_number_expansion_falls_back_to_original(self):
        text = 'Meet at 10:45 AM.'
        result = contextual_frontend.phonemize_with_metadata(text, 'en')
        self.assertEqual(result['metadata']['status'], 'fallback_word_level')
        self.assertEqual(result['metadata']['fallback_reason'], 'contextual_word_count_mismatch')
        self.assertEqual(result['phones'], frontend.phonemize(text, 'en'))

    def test_deliberate_split_is_not_repaired(self):
        result = contextual_frontend.phonemize_with_metadata('The rhyth m changed.', 'en')
        self.assertEqual(result['metadata']['input_non_silence_words'][1:3], ['rhyth', 'm'])
        self.assertEqual(result['phones'].count('SIL'), frontend.phonemize('The rhyth m changed.', 'en').count('SIL'))

    def test_training_frontend_source_and_behavior_unchanged(self):
        path = Path(frontend.__file__)
        before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        text = 'They will conduct an experiment.'
        before = frontend.phonemize(text, 'en')
        contextual_frontend.phonemize(text, 'en')
        self.assertEqual(frontend.phonemize(text, 'en'), before)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before_hash)

    def test_assignment_inputs_have_valid_inventory_and_preserve_sil_slots(self):
        for language, folder in [('en', 'English'), ('te', 'Telugu')]:
            for row in json.loads((ROOT / folder / 'test_samples.json').read_text()):
                with self.subTest(id=row['id']):
                    result = contextual_frontend.phonemize_with_metadata(row['text'], language)
                    self.assertTrue(result['phones'])
                    self.assertIn(result['metadata']['status'], ('contextual', 'fallback_word_level'))
                    self.assertEqual(result['phones'].count('SIL'), frontend.phonemize(row['text'], language).count('SIL'))
                    self.assertFalse(any('ˈ' in phone or 'ˌ' in phone for phone in result['phones']))


if __name__ == '__main__':
    unittest.main()
