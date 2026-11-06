"""Synthetic control-path regressions; these do not assert speech quality."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tts.inference import safe_predict,predict_with_energy_control,phonemize_for_inference

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('energy',torch.tensor([[4.,5.,6.]]))
        self.variance_adaptor=SimpleNamespace(duration_predictor=torch.nn.Identity(),energy_predictor=torch.nn.Identity())
    def forward(self,speakers,ids,lengths,max_length):
        log=self.variance_adaptor.duration_predictor(torch.log1p(torch.tensor([[1.,2.,3.]])))
        duration=torch.round(torch.exp(log)-1).long()
        energy=self.variance_adaptor.energy_predictor(self.energy)
        mel=torch.repeat_interleave(energy[0],duration[0])[:,None].repeat(1,80)[None]
        return mel,mel,torch.ones_like(energy),energy,log,duration,torch.zeros_like(ids,dtype=torch.bool),torch.zeros((1,6),dtype=torch.bool),lengths,torch.tensor([6])

class InferenceControls(unittest.TestCase):
    def setUp(self):
        self.model=FakeModel();self.args=(self.model,torch.tensor([0]),torch.tensor([[1,2,3]]),torch.tensor([3]))
    def test_default_is_exact_reference_path(self):
        expected=safe_predict(*self.args);actual,raw=predict_with_energy_control(*self.args)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
        self.assertTrue(torch.equal(raw,expected[3]))
    def test_override_changes_only_SIL_conditioning_and_preserves_raw(self):
        default=safe_predict(*self.args);mask=torch.tensor([[True,False,True]])
        pred,raw=predict_with_energy_control(*self.args,silence_mask=mask,zero_energy_value=-.75)
        self.assertTrue(torch.equal(raw,default[3]));self.assertTrue(torch.equal(self.model.energy,default[3]))
        self.assertTrue(torch.equal(pred[3][~mask],raw[~mask]));self.assertTrue(torch.all(pred[3][mask]==-.75))
        self.assertTrue(torch.equal(pred[2],default[2]));self.assertTrue(torch.equal(pred[5],default[5]))
        self.assertEqual(int(pred[5].sum()),pred[1].shape[1]);self.assertTrue(torch.equal(pred[9],default[9]))
    def test_raw_NaN_is_rejected_even_when_masked(self):
        self.model.energy[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):predict_with_energy_control(*self.args,silence_mask=torch.tensor([[True,False,False]]))
        self.assertEqual(len(self.model.variance_adaptor.energy_predictor._forward_hooks),0)
        self.assertEqual(len(self.model.variance_adaptor.duration_predictor._forward_hooks),0)
    def test_bad_masks_and_nonfinite_control_are_rejected(self):
        for mask in [torch.ones((1,3)),torch.ones((1,2),dtype=torch.bool)]:
            with self.assertRaises(ValueError):predict_with_energy_control(*self.args,silence_mask=mask)
        with self.assertRaises(ValueError):predict_with_energy_control(*self.args,silence_mask=torch.ones((1,3),dtype=torch.bool),zero_energy_value=float('nan'))
    def test_default_frontend_unchanged_and_metadata_explicit(self):
        with patch('tts.frontend.phonemize',return_value=['SIL','a','SIL']) as original:
            phones,metadata=phonemize_for_inference('text','en',{})
        original.assert_called_once_with('text','en');self.assertEqual(phones,['SIL','a','SIL']);self.assertEqual(metadata['used'],'trained_word_level')
    def test_context_fallback_is_not_misreported_as_contextual(self):
        details={'status':'fallback_word_level','fallback_reason':'contextual_word_count_mismatch'}
        with patch('tts.contextual_frontend.phonemize_with_metadata',return_value={'phones':['a'],'metadata':details}):
            phones,metadata=phonemize_for_inference('text','en',{'frontend':'sentence_context_v1'})
        self.assertEqual(metadata['used'],'trained_word_level');self.assertEqual(metadata['fallback_reason'],'contextual_word_count_mismatch')
    def test_context_option_does_not_change_Telugu_frontend(self):
        with patch('tts.frontend.phonemize',return_value=['a']) as original:
            _,metadata=phonemize_for_inference('text','te',{'frontend':'sentence_context_v1'})
        original.assert_called_once_with('text','te');self.assertEqual(metadata['used'],'trained_word_level')
        self.assertEqual(metadata['fallback_reason'],'sentence_context_v1_is_English_only')

if __name__=='__main__':unittest.main()
