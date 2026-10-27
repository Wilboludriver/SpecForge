"""CPU controls for symmetric policy application without hiding missing wiring."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/plana'
sys.path.insert(0, str(SCRIPTS))
from performance_helpers import (POLL_DEFAULTS, POLL_SOURCES, audit_polling_receipts, comparison_contract,
    content_fingerprint, polling_policy_call, validate_polling_policy)


POLICY = dict(zip(POLL_DEFAULTS, (0.003, 0.007, 0.011)))


class PolicyTests(unittest.TestCase):
    def test_complete_summary_uses_actual_git_identity_manifest_schema(self):
        import run_performance as runner
        project = next(p for p in SCRIPTS.parents if (p/'NEXT_SESSION_PROMPT.txt').exists())
        source = project/'src/SpecForge-official'
        identity = runner.git_identity(source, runner.OFFICIAL_SF)
        self.assertIn('path', identity)
        self.assertNotIn('root', identity)
        row = {'input_ids': [1, 2], 'loss_mask': [1, 1]}
        workload = {'intended_training': [row, row], 'training_warmup_steps': 1,
                    'measured_steps': 1, 'kind': 'hot'}
        polling = [{'field': k, 'value': v, 'mode': 'official-api', 'applied': True,
                    'explicit_from_caller': False, 'source_file': str(source/POLL_SOURCES[k][0]),
                    'callable': POLL_SOURCES[k][1], 'monotonic': 0.} for k, v in POLICY.items()]
        events = {
            'polling-policy': polling,
            'consumed': [{'step': i, 'sample_ids': [f's{i}'], 'optimizer_stepped': True,
                          'monotonic': float(i)} for i in (1, 2)],
            'fetch-validation': [{'sample_id': f's{i}', 'content_fingerprint': content_fingerprint(row['input_ids'], row['loss_mask']),
                'validation_wall_s': .001, 'monotonic': float(i)} for i in (1, 2)],
            'training-steps': [{'step': i, 'metrics': {'perf/optimizer_step_time_s': .1},
                                'monotonic': float(i)} for i in (1, 2)],
            'captures': [{'sample_id': f's{i}', 'attempt_id': f'a{i}', 'N': 2, 'L': 0,
                          'span': [0, 2], 'monotonic': float(i)} for i in (1, 2)],
            'fit': [{'measured_fit_boundary_wall_s': .1, 'monotonic': 3.}],
            'draft-initial': [{'sha256': 'synthetic-cpu-fixture', 'monotonic': 0.}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'events').mkdir()
            (root/'manifest.json').write_text(json.dumps({'specforge': identity,
                'polling_policy': POLICY, 'polling_mode': 'official-api'}))
            for kind, records in events.items():
                (root/'events'/f'{kind}-1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            result = runner.summarize(root, workload, 'host-off')
            self.assertTrue(result['polling']['all_components_verified'])
            self.assertTrue(result['consumed_order']['full_order_matches'])
            self.assertEqual(result['capture']['computed_tokens'], 2)
            self.assertEqual(result['training']['optimizer_step_wall_s'], .1)
            polling[0]['source_file'] = '/tmp/foreign-source.py'
            (root/'events'/'polling-policy-1.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in polling))
            with self.assertRaisesRegex(AssertionError, 'source differs'):
                runner.summarize(root, workload, 'host-off')

    def test_nonfinite_zero_negative_boolean_and_missing_fields_fail(self):
        self.assertEqual(validate_polling_policy(POLICY), POLICY)
        for field in POLICY:
            for invalid in (0, -0.1, float('nan'), float('inf'), -float('inf'), True):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    validate_polling_policy(dict(POLICY, **{field: invalid}))
        with self.assertRaises(ValueError):
            validate_polling_policy({'producer_backpressure_poll_s': .003})

    def test_official_api_can_inject_but_candidate_missing_or_wrong_call_fails(self):
        calls = []
        receipts = []
        def producer(*, backpressure_poll_s=.2, high_watermark=4, low_watermark=2):
            calls.append((backpressure_poll_s, high_watermark, low_watermark))
            return 'drive'
        wrap = lambda mode: polling_policy_call(producer, argument='backpressure_poll_s',
            field='producer_backpressure_poll_s', value=.003, mode=mode, emit=receipts.append)
        self.assertEqual(wrap('official-api')(), 'drive')
        self.assertEqual(calls, [(.003, 4, 2)])
        self.assertFalse(receipts[-1]['explicit_from_caller'])
        for kwargs in ({}, {'backpressure_poll_s': .2}, {'backpressure_poll_s': True}):
            with self.assertRaises(AssertionError):
                wrap('candidate-config')(**kwargs)
        self.assertEqual(len(calls), 1)  # Broken candidate wiring never reaches the factory.
        wrap('candidate-config')(backpressure_poll_s=.003)
        self.assertTrue(receipts[-1]['explicit_from_caller'])
        with self.assertRaises(AssertionError):
            wrap('official-api')(backpressure_poll_s=.2)

    def test_actual_queue_constructor_retains_interval_and_other_parameters(self):
        from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue
        original = StreamingRefQueue.__init__
        receipts = []
        channel = object()
        for mode, kwargs in [('official-api', {}), ('candidate-config', {'poll_s': .007})]:
            queue = StreamingRefQueue.__new__(StreamingRefQueue)
            call = polling_policy_call(original, argument='poll_s', field='consumer_ref_poll_s',
                value=.007, mode=mode, emit=receipts.append)
            call(queue, channel, idle_timeout_s=123., **kwargs)
            self.assertEqual(queue.poll_s, .007)
            self.assertIs(queue.channel, channel)
            self.assertEqual(queue.idle_timeout_s, 123.)
        self.assertEqual(len(receipts), 2)

    def test_constructor_that_drops_parameter_cannot_forge_application_receipt(self):
        class BrokenQueue:
            def __init__(self, *, poll_s=.1):
                self.poll_s = .1
        receipts = []
        call = polling_policy_call(BrokenQueue.__init__, argument='poll_s', field='consumer_ref_poll_s',
            value=.007, mode='candidate-config', emit=receipts.append)
        with self.assertRaises(AssertionError):
            call(BrokenQueue.__new__(BrokenQueue), poll_s=.007)
        self.assertEqual(receipts, [])

    def test_all_three_actual_receipts_are_required_and_conflicts_rejected(self):
        source = SCRIPTS.parents[1]
        rows = [{'field': k, 'value': v, 'mode': 'candidate-config', 'applied': True,
                 'explicit_from_caller': True, 'source_file': str(source/POLL_SOURCES[k][0]),
                 'callable': POLL_SOURCES[k][1]} for k, v in POLICY.items()]
        self.assertTrue(audit_polling_receipts(rows, POLICY, 'candidate-config', source_root=source)['all_components_verified'])
        with self.assertRaises(AssertionError):
            audit_polling_receipts(rows[:-1], POLICY, 'candidate-config')
        for field, value in [('value', .2), ('explicit_from_caller', False), ('mode', 'official-api')]:
            bad = copy.deepcopy(rows); bad[0][field] = value
            with self.assertRaises(AssertionError):
                audit_polling_receipts(bad, POLICY, 'candidate-config')
        for field, value in [('callable', 'unrelated.factory'), ('source_file', '/tmp/foreign-source.py')]:
            bad = copy.deepcopy(rows); bad[0][field] = value
            with self.assertRaises(AssertionError):
                audit_polling_receipts(bad, POLICY, 'candidate-config', source_root=source)

    def test_contract_equivalence_does_not_hide_candidate_config_conflict(self):
        config = {'model': {}, 'data': {}, 'runtime': {'producer_concurrency': 1},
                  'deployment': {'disaggregated': {'managed_local': {'capture_servers': [{}]}}}}
        candidate = copy.deepcopy(config); candidate['runtime'].update(POLICY)
        args = ('data', {}, 64 << 20, 256)
        self.assertEqual(comparison_contract(config, *args, polling_policy=POLICY),
                         comparison_contract(candidate, *args, polling_policy=POLICY))
        self.assertEqual(config['runtime'], {'producer_concurrency': 1})
        candidate['runtime']['ref_distributor_poll_s'] = .05
        with self.assertRaises(ValueError):
            comparison_contract(candidate, *args, polling_policy=POLICY)

    def test_matrix_passes_same_explicit_policy_to_all_three_groups(self):
        import run_performance_matrix as matrix_runner
        project = next(p for p in SCRIPTS.parents if (p / 'NEXT_SESSION_PROMPT.txt').exists())
        matrix = json.loads((project / 'implementation/configs/performance/matrix-9f4af5a-6e81950-r3.json').read_text())
        matrix['polling_policy'] = POLICY
        cases = matrix_runner.cases(matrix, ['qwen3-4b'], ['hot'], hot_repetitions=1)
        self.assertEqual(len(cases), 3)
        for case in cases:
            argv, _, _ = matrix_runner.harness_command(matrix, case, '/tmp/policy-plan')
            for field, value in POLICY.items():
                flag = '--' + field.replace('_', '-')
                self.assertEqual(argv[argv.index(flag) + 1], str(value))


if __name__ == '__main__':
    unittest.main()
