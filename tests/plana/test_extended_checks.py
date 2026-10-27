"""Adversarial checks of metric and negotiated-diagnostic acceptance."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/plana'))
from extended_checks import compare_metric_trees,validate_diagnostics


class TestExtendedEvidenceChecks(unittest.TestCase):
    def test_loss_denominator_or_teacher_count_change_is_compared(self):
        reference={'loss_terms':[50.,8.],'ratio_metrics':{'teacher/top1':[4.,8.]}}
        for changed in ({'loss_terms':[50.,9.],'ratio_metrics':{'teacher/top1':[4.,8.]}},
                        {'loss_terms':[50.,8.],'ratio_metrics':{'teacher/top1':[5.,8.]}}):
            self.assertFalse(all(v['passed'] for v in compare_metric_trees(changed,reference,rtol=.02,atol=.02)))

    def test_missing_metric_rejected(self):
        with self.assertRaises(AssertionError):compare_metric_trees({}, {'loss_terms':[1.,2.]},rtol=.02,atol=.02)

    def test_legacy_full_prefill_is_explicit_and_never_claims_cache(self):
        self.assertEqual(validate_diagnostics({},100,require_cache=False)['L'],0)
        with self.assertRaises(AssertionError):validate_diagnostics({},100,require_cache=True)

    def test_h_is_not_used_as_engine_l_and_absolute_span_is_required(self):
        result={'H':768,'L':256,'N':1024,'span':[256,1024],'execution_epoch':0,'rank':0,'issue_sequence':1,'attempt_id':'a',
                'reservation_token':'t','server_epoch':'e','capture_signature':'s','capture_namespace':'n','reused_tokens':256}
        self.assertEqual(validate_diagnostics(result,1024,require_cache=True)['L'],256)
        for patch in ({'L':800},{'span':[0,768]},{'reused_tokens':768}):
            with self.subTest(patch=patch),self.assertRaises(AssertionError):validate_diagnostics({**result,**patch},1024,require_cache=True)


if __name__=='__main__':unittest.main()
