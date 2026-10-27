"""Real CUDA graph overwrite through the actual capture snapshot/controller.

This tests graph-buffer ownership, not target-model graph execution. It starts
no server and uses no ports. Run only with an explicitly assigned physical GPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future
import gc
import json
import os
from pathlib import Path
import subprocess
from threading import Event
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--sglang-source', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--physical-gpu', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(args.sglang_source).resolve()
    def identity():
        return {name: subprocess.check_output(['git', '-C', str(source), *command], text=True).strip()
                for name, command in {'commit': ['rev-parse', 'HEAD'], 'status': ['status', '--porcelain']}.items()}
    before = identity()
    assert before == {'commit': args.expected_commit, 'status': ''}, before
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.physical_gpu
    import torch
    import sglang.srt.spec_feature_bridge as bridge_module
    from sglang.srt.spec_feature_bridge import AttemptRegistry, SpecFeatureBridge
    from sglang.srt.spec_feature_cache import (
        CaptureController, ExecutionIdentity, FeatureSchema, FeatureSpec, PrefixFeatureCache,
    )
    assert Path(bridge_module.__file__).resolve().is_relative_to(source/'python')
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    result = {'passed': False, 'scope': __doc__, 'source_before': before,
              'bridge_import': bridge_module.__file__, 'physical_gpu': args.physical_gpu, 'cases': []}
    try:
        for name, width, layers in (
            ('qwen3', 2560, (1,9,17,25,33)),
            ('qwen35', 2560, (1,5,9,13,17,21,25,29)),
            ('qwen36-a3b', 2048, (1,6,11,16,22,27,32,37)),
        ):
            schema = FeatureSchema((FeatureSpec('aux', width*len(layers), 'bfloat16', layers),
                                    FeatureSpec('last_hidden', width, 'bfloat16', ())))
            cache = PrefixFeatureCache(schema, 256, 64<<20)
            controller = CaptureController(cache, schema, 64<<20, 128<<20, 64<<20)
            owner = object.__new__(SpecFeatureBridge)
            owner.s = SimpleNamespace(ps=SimpleNamespace(tp_rank=0))
            owner.writer = owner.enabled = True
            owner.epoch, owner.namespace, owner.signature = name, name, schema.signature
            owner.cache, owner.controller = cache, controller
            owner.registry = AttemptRegistry(name, 8)
            owner.jobs, owner.batches, owner.history = {}, [], []
            entered, release_copy = Event(), Event()
            original_copy = controller._copy_suffix
            def gated_copy(job):
                entered.set()
                if not release_copy.wait(30):
                    raise RuntimeError('graph test copy barrier timed out')
                return original_copy(job)
            controller._copy_suffix = gated_copy
            gate = Future()
            submitted = []
            def publish(items):
                submitted.extend(items)
                return gate
            try:
                n = 512
                scalar = torch.tensor(3., dtype=torch.bfloat16, device='cuda')
                aux = torch.empty((n, width*len(layers)), dtype=torch.bfloat16, device='cuda')
                hidden = torch.empty((n, width), dtype=torch.bfloat16, device='cuda')
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        aux.copy_(scalar.expand_as(aux))
                        hidden.copy_((scalar+1).expand_as(hidden))
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    aux.copy_(scalar.expand_as(aux))
                    hidden.copy_((scalar+1).expand_as(hidden))
                graph.replay()
                execution = ExecutionIdentity(name, 'graph-request', 'graph-attempt', 0)
                tokens = tuple(range(n))
                lease = cache.lookup_and_pin(name, tokens, n)
                reservation = controller.try_reserve_output(execution, n, 0, n*16)
                assert reservation is not None
                plan = controller.freeze_plan(execution, name, tokens, n, 0, lease, reservation)
                entry = owner.registry.begin(dict(store_id=name, sample_id='graph-sample', gen=1,
                                                 attempt_id='graph-attempt', request_digest='fixture'))
                req = SimpleNamespace(rid='graph-request', origin_input_ids=list(tokens),
                    spec_feature_attempt=entry, spec_feature_execution_epoch=0,
                    spec_feature_plan=plan, extend_range=SimpleNamespace(length=n), spec_capture={})
                entry.request = req
                entry.last_plan = {'H':0,'L':0,'N':n,'span':[0,n],'execution_epoch':0,'rank':0}
                owner.snapshot(SimpleNamespace(reqs=[req]), SimpleNamespace(logits_output=SimpleNamespace(
                    hidden_states=aux, last_hidden_states=hidden)))
                assert entered.wait(30), 'real controller worker did not reach copy barrier'
                # The shared graph buffers are overwritten while actual D2H is
                # blocked. Only the bridge's owned snapshot can retain 3/4.
                scalar.fill_(9)
                graph.replay()
                torch.cuda.synchronize()
                assert bool(torch.all(aux == 9)) and bool(torch.all(hidden == 10))
                blocked = controller.stats()
                assert blocked['device_bytes'] > 0 and blocked['output_bytes'] > 0
                release_copy.set()
                output = owner.queue_captures([(req, {}, None, None, False, 0.)],
                                              SimpleNamespace(submit_samples=publish))
                deadline = time.monotonic()+30
                while not submitted and time.monotonic() < deadline:
                    owner.poll()
                    time.sleep(.001)
                assert len(submitted) == 1
                _, actual_aux, actual_hidden = submitted[0]
                assert torch.equal(actual_aux, torch.full_like(actual_aux, 3))
                assert torch.equal(actual_hidden, torch.full_like(actual_hidden, 4))
                assert not output.done() and controller.stats()['output_bytes'] > 0
                scalar.fill_(17)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(actual_aux, torch.full_like(actual_aux, 3))
                gate.set_result([{'sample_id':'graph-sample'}])
                owner.poll()
                assert output.result(timeout=5)[0]['state'] == 'PUBLISHED'
                del actual_aux, actual_hidden
                submitted.clear()
                owner.poll()
                gc.collect()
                final = controller.stats()
                for field in ('pinned_bytes','device_bytes','output_bytes','active_leases',
                              'fill_reservations','pending_copies','pending_executions'):
                    assert final[field] == 0, (field, final)
                result['cases'].append({'model_schema':name,'rows':n,'graph_replays':3,
                    'blocked':blocked,'final':final,'passed':True})
            finally:
                release_copy.set()
                if not gate.done():
                    gate.set_exception(RuntimeError('test cleanup'))
                owner.shutdown()
        result['source_after'] = identity()
        assert result['source_after'] == before
        result['passed'] = True
    except BaseException as exc:
        import traceback
        result['error'] = repr(exc)
        (root/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (root/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'passed':result['passed'],'cases':len(result['cases'])}))


if __name__ == '__main__':
    main()
