"""Adversarial checks of the independent acceptance oracles (CPU only)."""
import importlib.util
from pathlib import Path
import unittest

import torch

_PATH=Path(__file__).resolve().parents[2]/'scripts/plana/validation_helpers.py'
_SPEC=importlib.util.spec_from_file_location('plana_validation_helpers',_PATH)
helpers=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(helpers)


class TestIndependentOracles(unittest.TestCase):
    def row(self):
        return {'id':'seam','input_ids':[5,8,13,21,34], 'loss_mask':[0,0,1,1,1]}

    def features(self):
        row=self.row()
        return {'input_ids':torch.tensor([row['input_ids']]),'loss_mask':torch.tensor([row['loss_mask']]),
                'hidden_states':torch.arange(30,dtype=torch.bfloat16).reshape(1,5,6),
                'target_last_hidden_states':torch.arange(10,dtype=torch.bfloat16).reshape(1,5,2)}

    def test_every_token_and_integer_mask_are_validated(self):
        for corrupted in ('input_ids','loss_mask'):
            values=self.features();values[corrupted][0,2]+=1
            with self.subTest(corrupted=corrupted),self.assertRaises(AssertionError):
                helpers.validate_features(values,self.row(),hidden_size=2,layers=[1,3,5])

    def test_missing_last_hidden_and_truncated_suffix_rejected(self):
        for mode in ('missing','truncated','dtype'):
            values=self.features()
            if mode=='missing': del values['target_last_hidden_states']
            elif mode=='truncated':values['hidden_states']=values['hidden_states'][:,:-1]
            else:values['hidden_states']=values['hidden_states'].float()
            with self.subTest(mode=mode),self.assertRaises(AssertionError):
                helpers.validate_features(values,self.row(),hidden_size=2,layers=[1,3,5])

    def test_full_tensor_oracle_detects_seam_and_layer_permutation(self):
        expected=self.features()['hidden_states']
        for actual in (expected.roll(1,dims=1),expected.reshape(1,5,3,2).flip(2).reshape(1,5,6)):
            error=helpers.tensor_errors(actual,expected,rtol=0,atol=0)
            self.assertFalse(error['passed']);self.assertGreater(error['mismatches'],0)

    def test_gradient_scale_defect_cannot_hide_in_cosine(self):
        expected=torch.tensor([1e-6,.5,-1.,3.])
        self.assertFalse(helpers.tensor_errors(expected*1.1,expected,rtol=.02,atol=1e-5)['passed'])

    def test_nonfinite_values_fail_loudly(self):
        values=self.features();values['target_last_hidden_states'][0,2,0]=float('nan')
        with self.assertRaises(AssertionError):helpers.validate_features(values,self.row(),hidden_size=2,layers=[1,3,5])
        with self.assertRaises(AssertionError):helpers.tensor_errors(torch.tensor([float('inf')]),torch.ones(1),rtol=.02,atol=.02)

    def test_success_preserves_exact_complete_sample(self):
        values=self.features();helpers.validate_features(values,self.row(),hidden_size=2,layers=[1,3,5])
        self.assertTrue(helpers.tensor_errors(values['hidden_states'],values['hidden_states'].clone(),rtol=0,atol=0)['passed'])


if __name__=='__main__':unittest.main()
