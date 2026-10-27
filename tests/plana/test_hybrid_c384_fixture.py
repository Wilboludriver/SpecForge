"""Adversarial acceptance-oracle checks, not simulated model acceptance."""
import importlib.util
from pathlib import Path
import unittest

_path = Path(__file__).resolve().parents[2] / 'scripts/plana/hybrid_c384_fixture.py'
_spec = importlib.util.spec_from_file_location('hybrid_c384_fixture', _path)
fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixture)


class HybridC384OracleTests(unittest.TestCase):
    def evidence(self):
        cases = [{'input_ids': [1] * n, 'id': str(i)} for i, n in enumerate((256, 1024, 512, 1024, 1024, 1024, 1024))]
        diagnostics = [{'H': h, 'L': l, 'N': len(row['input_ids']), 'span': [l, len(row['input_ids'])], 'reused_tokens': l}
                       for row, h, l in zip(cases, (0, 0, 384, 768, 768, 0, 0), (0, 0, 192, 448, 0, 0, 0), strict=True)]
        actions = {'engine_flush': {'status': 200, 'hs_cache_unchanged': True},
                   'later_checkpoint_probe': {'probes': [
                       {'input_ids': cases[2]['input_ids'], 'cached_tokens': 256, 'has_spec_capture': False, 'hs_cache_unchanged': True},
                       {'input_ids': cases[2]['input_ids'], 'cached_tokens': 448, 'has_spec_capture': False, 'hs_cache_unchanged': True}]},
                   'ordinary_uncaptured_kv': {'cached_tokens': 960, 'has_spec_capture': False, 'hs_cache_unchanged': True}}
        return cases, diagnostics, actions

    def test_actual_restore_is_recorded_without_forcing_an_arithmetic_grid(self):
        cases, diagnostics, actions = self.evidence()
        result = fixture.validate_outcome(cases, diagnostics, actions)
        self.assertEqual(result['actual_L'][2], 192)
        diagnostics[2]['L'] = diagnostics[2]['reused_tokens'] = 256
        diagnostics[2]['span'][0] = 256
        self.assertEqual(fixture.validate_outcome(cases, diagnostics, actions)['actual_L'][2], 256)

    def test_missing_causes_or_false_kv_claim_fail(self):
        for mode in ('maximal-L', 'flush-reuse', 'lost-HS', 'no-KV', 'captured-KV', 'KV-changes-HS', 'no-later-checkpoint'):
            cases, diagnostics, actions = self.evidence()
            if mode == 'maximal-L': diagnostics[2]['L'] = diagnostics[2]['H']
            elif mode == 'flush-reuse': diagnostics[4]['L'] = 64
            elif mode == 'lost-HS': actions['engine_flush']['hs_cache_unchanged'] = False
            elif mode == 'no-KV': actions['ordinary_uncaptured_kv']['cached_tokens'] = 0
            elif mode == 'captured-KV': actions['ordinary_uncaptured_kv']['has_spec_capture'] = True
            elif mode == 'KV-changes-HS': actions['ordinary_uncaptured_kv']['hs_cache_unchanged'] = False
            else: actions['later_checkpoint_probe']['probes'][-1]['cached_tokens'] = 384
            with self.subTest(mode=mode), self.assertRaises(AssertionError):
                fixture.validate_outcome(cases, diagnostics, actions)

    def test_later_checkpoint_must_be_legal_for_the_exact_bounded_request(self):
        for mode in ('longer-request', 'N-minus1-violation', 'third-probe', 'first-probe-HS-mutation'):
            cases, diagnostics, actions = self.evidence()
            probes = actions['later_checkpoint_probe']['probes']
            if mode == 'longer-request': probes[-1]['input_ids'] = [1] * 1024
            elif mode == 'N-minus1-violation': probes[-1]['cached_tokens'] = 960
            elif mode == 'third-probe': probes.append(dict(probes[-1]))
            else: probes[0]['hs_cache_unchanged'] = False
            with self.subTest(mode=mode), self.assertRaises(AssertionError):
                fixture.validate_outcome(cases, diagnostics, actions)

    def test_true_schema_drives_exact_two_chunk_byte_budget(self):
        self.assertEqual(fixture.cache_capacity_bytes({'hidden_size': 2560, 'layers': list(range(8))}), 35389440)


if __name__ == '__main__':
    unittest.main()
