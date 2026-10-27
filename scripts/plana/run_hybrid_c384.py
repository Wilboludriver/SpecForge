#!/usr/bin/env python3
"""Actual hybrid supplemental gate: earlier legal L, HS-only and uncaptured KV.

Requires a separate explicit GPU/port allocation and passing original native16
official baseline. This seven-capture test supplements the original 16-case
gate. It never substitutes for TP/model, cancellation or performance gates.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import time
import traceback

from hybrid_c384_fixture import CHUNK_TOKENS, build_cases, cache_capacity_bytes, validate_outcome
from run_validation import (capture_pass, compare_features, git_identity, isolated_config,
                            replay_gradients, runtime_manifest, start_services)
from validation_helpers import canonical_model_info, load_cases, write_json


def cache_snapshot(stats):
    return {key: stats['cache'][key] for key in ('ready_blocks', 'ready_bytes', 'cache_bytes')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'run-dir', 'cases', 'baseline-dir', 'specforge-source', 'sglang-source'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--capture-only', action='store_true')
    args = parser.parse_args()
    args.require_cache = True
    args.feature_rtol = args.feature_atol = args.loss_rtol = args.loss_atol = .02
    args.gradient_rtol, args.gradient_atol, args.gradient_relative_l2 = .02, 1e-5, .02
    def terminate(signum, frame):
        raise KeyboardInterrupt(f'owned C384 harness received signal {signum}')
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, terminate)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    started, manifest, actions = [], {}, {}
    result = {'passed': False, 'scope': 'supplemental seven-case hybrid runtime gate; original native16 required separately'}
    try:
        import requests
        import yaml
        from specforge.application import resolve_run
        from specforge.config import load_config
        from specforge.inference.adapters.server_capture import _SessionPost
        from specforge.launch_plan import build_launch_plan, _stop_services
        manifest = runtime_manifest(args)
        write_json(run_dir / 'manifest.json', manifest)
        write_json(run_dir / 'tolerances-predeclared.json', {key: value for key, value in vars(args).items()
                                                          if 'rtol' in key or 'atol' in key or key == 'gradient_relative_l2'})
        native = load_cases(args.cases)
        baseline = Path(args.baseline_dir)
        if not json.loads((baseline / 'result.json').read_text()).get('passed'):
            raise AssertionError('requires completed passing official native16 baseline')
        if json.loads((baseline / 'cases.json').read_text()) != native:
            raise AssertionError('official baseline native16 inputs changed')
        expected_info = canonical_model_info(json.loads((baseline / 'model-info.json').read_text()))
        rows = build_cases(native)
        write_json(run_dir / 'cases.json', rows)
        config_path = isolated_config(args, run_dir)
        raw = yaml.safe_load(config_path.read_text())
        extra = raw['model'].setdefault('sglang_extra_args', [])
        if '--spec-feature-cache' not in extra or '--enable-deterministic-inference' not in extra:
            raise AssertionError('supplement requires explicit HS cache and matched deterministic configuration')
        if any(str(value).split('=', 1)[0] in ('--spec-feature-chunk-tokens', '--spec-feature-cache-bytes') for value in extra):
            raise AssertionError('base config must not carry competing C384 cache limits')
        capacity = cache_capacity_bytes(expected_info)
        extra.extend(['--spec-feature-chunk-tokens', str(CHUNK_TOKENS), '--spec-feature-cache-bytes', str(capacity)])
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = load_config(str(config_path))
        resolved = resolve_run(cfg)
        plan = build_launch_plan(resolved.config, algorithm=resolved.algorithm, config_path=str(config_path), env=os.environ)
        (run_dir / 'launch-plan.json').write_text(plan.render())
        if sum(service.command.label.startswith('capture-server-') for service in plan.services) != 1:
            raise AssertionError('one target service required')
        started = start_services(plan, run_dir)
        ordinary_post = _SessionPost()

        def stats(adapter):
            capability = adapter._capture_protocol.capability()
            if capability['limits']['chunk_tokens'] != CHUNK_TOKENS or capability['limits']['cache_bytes'] != capacity:
                raise AssertionError('actual server did not apply exact C384/two-chunk capacity')
            value = adapter._capture_protocol.call({'action': 'stats', 'server_epoch': capability['server_epoch']})
            if any(value[name] for name in ('active_attempts', 'active_readers', 'pending_deliveries')):
                raise AssertionError('fixture action requires a fully drained previous delivery')
            return capability, value

        def before_case(adapter, index, row):
            if index not in (4, 6):
                return
            capability, before = stats(adapter)
            if before['cache']['ready_blocks'] != 2:
                raise AssertionError('fixture requires exactly two ready HS chunks before action')
            if index == 4:
                response = requests.post(adapter.base_url + '/flush_cache', json={}, timeout=60)
                response.raise_for_status()
                _, after = stats(adapter)
                actions['engine_flush'] = {'status': response.status_code, 'body': response.text,
                                           'before': before, 'after': after,
                                           'hs_cache_unchanged': cache_snapshot(before) == cache_snapshot(after)}
            else:
                response = ordinary_post(adapter.base_url + '/generate', json_body={
                    'input_ids': row['input_ids'], 'extra_key': capability['capture_namespace'],
                    'sampling_params': {'temperature': 0.0, 'max_new_tokens': 0}}, timeout=60)
                meta = response['meta_info']
                _, after = stats(adapter)
                actions['ordinary_uncaptured_kv'] = {'response': response,
                    'cached_tokens': meta['cached_tokens'], 'has_spec_capture': 'spec_capture' in meta,
                    'before': before, 'after': after,
                    'hs_cache_unchanged': cache_snapshot(before) == cache_snapshot(after)}
            write_json(run_dir / 'fixture-actions.json', actions)

        actual, info, diagnostics = capture_pass(args, cfg, resolved, plan, run_dir, 'candidate', rows,
                                                 before_case=before_case)
        write_json(run_dir / 'model-info.json', info)
        if canonical_model_info(info) != expected_info:
            raise AssertionError('supplement capture/draft schema differs from official baseline')
        result['topology'] = validate_outcome(rows, diagnostics, actions)
        server = next(service for service in plan.services if service.command.label.startswith('capture-server-'))
        server_log = Path(server.log_path).read_text()
        if 'ComponentType.FULL' not in server_log or 'ComponentType.MAMBA' not in server_log or 'UnifiedTreeCore' not in server_log:
            raise AssertionError('actual target did not report FULL+MAMBA UnifiedTreeCore')
        _stop_services(started, grace_s=30)
        started = []
        reference = [baseline / 'off-0' / f'{row["source_case_index"]:02d}-{row["source_case_id"]}.pt' for row in rows]
        result['feature_comparison'] = compare_features(args, rows, reference, actual, info, run_dir, diagnostics)
        if not args.capture_only:
            result['loss_gradient_comparison'] = replay_gradients(args, cfg, resolved, rows, reference, actual, run_dir)
        result['passed'] = True
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        (run_dir / 'failure.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        if started:
            from specforge.launch_plan import _stop_services
            _stop_services(started, grace_s=30)
        if manifest:
            after = {key: git_identity(getattr(args, key + '_source')) for key in ('specforge', 'sglang')}
            write_json(run_dir / 'source-after.json', after)
            if any(after[key] != manifest[key] for key in after):
                result['passed'], result['error'] = False, 'source identity changed while supplemental gate ran'
        result['ended_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        write_json(run_dir / 'result.json', result)
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
