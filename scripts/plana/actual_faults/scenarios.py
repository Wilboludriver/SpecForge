"""External real adapter/store client, full-feature oracles and fault assertions."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time
import uuid
from actual_faults import append_json, atomic_json, read_json, wait_for


def rank_events(directory, case_id=None):
    rows = []
    for path in Path(directory).glob('rank-*.jsonl'):
        text=path.read_text()
        lines=text.splitlines()
        if lines and not text.endswith('\n'):lines=lines[:-1]
        for line in lines:
            row = json.loads(line)
            if case_id is None or row.get('case_id') == case_id:
                rows.append(row)
    return rows


def command(directory, action, *, tp_size, timeout, **values):
    identifier = uuid.uuid4().hex
    atomic_json(Path(directory) / 'command.json', {'id': identifier, 'action': action, **values})
    paths = [Path(directory) / f'ack-{identifier}-rank{rank}.json' for rank in range(tp_size)]
    wait_for(lambda: all(path.exists() for path in paths), timeout, 'all-rank test command acknowledgment')
    receipts = [read_json(path) for path in paths]
    failures = [row for row in receipts if row.get('error')]
    if failures:
        raise RuntimeError('actual rank test command failed: '+json.dumps(failures))
    return receipts


def assert_cuda_cancel_receipts(rows, case_id, marker):
    selected = [row for row in rows if row.get('case_id') == case_id]
    def one(event):
        found = [row for row in selected if row['event'] == event]
        if len(found) != 1:
            raise AssertionError('CUDA cancellation requires exactly one '+event)
        return found[0]
    queued = one('delayed_real_cuda_event')
    cancelled = one('cancel_event_observation')
    applied = one('cancel_control_applied')
    released = one('cuda_gate_released')
    if (not marker['pending'] or marker['stage'] != 'after_real_aux_d2h'
            or marker['issued'] != ['aux'] or not queued['event_pending']
            or queued['stage'] != marker['stage'] or queued['issued'] != ['aux']
            or not cancelled['event_pending'] or released['reason'] != 'observed_cancel'
            or not queued['seq'] < cancelled['seq'] < applied['seq'] < released['seq']
            or applied['state'] != 'DRAINING' or applied['writer_drained'] is not False
            or not queued['monotonic_ns'] <= cancelled['monotonic_ns'] <= applied['monotonic_ns'] <= released['monotonic_ns']
            or not queued['event_handle'] == cancelled['event_handle'] == marker['event_handle']
            or queued['event_handle'] == queued['producer_event_handle']
            or set(queued['source_pointers']) != {'aux', 'last_hidden'}
            or any(not pointer for pointer in queued['source_pointers'].values())
            or any(row['rank'] != 0 for row in (queued,cancelled,applied,released))):
        raise AssertionError('real DMA cancellation lacks pending-event identity, order or explicit release proof')
    return {'event_handle':queued['event_handle'],'pending_at_cancel':True,
            'release_reason':released['reason'],'stage':marker['stage']}


def assert_clean_server(stats):
    for name in ('active_attempts', 'active_readers', 'pending_deliveries'):
        if stats[name]:
            raise AssertionError(f'server retained {name}: {stats[name]}')
    for name in ('active_leases', 'fill_reservations', 'writing_blocks'):
        if stats['cache'][name]:
            raise AssertionError(f'cache retained {name}')
    for name in ('pinned_bytes', 'output_bytes', 'device_bytes', 'pending_executions', 'pending_copies', 'output_reservations', 'quarantined_executions'):
        if stats['controller'][name]:
            raise AssertionError(f'controller retained {name}')
    capacity = stats['limits']['pinned_capacity_bytes']
    if stats['controller']['pinned_arena_bytes'] > capacity:
        raise AssertionError('retained physical pinned arena exceeds advertised capacity')


def check_features(features, row, model, *, reference=None, rtol=.02, atol=.02):
    import torch
    N = len(row['input_ids'])
    expected = {'hidden_states': (1, N, model['hidden_size'] * len(model['layers'])),
                'target_last_hidden_states': (1, N, model['hidden_size']),
                'input_ids': (1, N), 'loss_mask': (1, N)}
    if set(features) != set(expected):
        raise AssertionError('full required feature set differs: ' + str(set(features)))
    for name, shape in expected.items():
        value = features[name]
        if tuple(value.shape) != shape:
            raise AssertionError(f'{name}: incomplete full sample {tuple(value.shape)} != {shape}')
        if name in ('input_ids', 'loss_mask'):
            if value.dtype != torch.int64 or value.tolist() != [row[name]]:
                raise AssertionError(name + ' does not exactly match its own request')
        else:
            if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
                raise AssertionError(name + ' dtype/finiteness contract failed')
            if reference is not None:
                torch.testing.assert_close(value, reference[name], rtol=rtol, atol=atol)
    return {name: value.detach().cpu().clone() for name, value in features.items()}


def run_namespace(directory):
    # Include the unique outer run directory even when the supervisor calls
    # every inner driver directory "engine". Restart children remain separate.
    return 'actual-' + hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:24]


class Client:
    def __init__(self, config, directory, base_url):
        from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter, ServerCaptureSchema, _SessionPost
        from specforge.inference.capture import CaptureConfig
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        self.config, self.directory, self.base_url = config, Path(directory), base_url
        self.model = config['model']
        self.run_id = run_namespace(self.directory)
        atomic_json(self.directory/'sample-namespace.json',
                    {'directory':str(self.directory.resolve()),'store_id':self.run_id})
        self.tp_size = config['engine_kwargs']['tp_size']
        self.timeout = float(config.get('timeout_s', 90))
        self.store = MooncakeFeatureStore(store_id=self.run_id, setup_kwargs=config['store_setup'],
                                         receive_buffers='pageable', max_resident_bytes=1 << 30)
        self.transport = _SessionPost()
        self.local = __import__('threading').local()
        def post(url, *, json_body, timeout):
            sent_ns = time.monotonic_ns()
            if url.endswith('/generate'):
                body = dict(json_body)
                body['rid'] = [self.local.rid]
                if self.local.force_full:
                    body['spec_capture'] = [dict(value, force_full_prefill=True) for value in body['spec_capture']]
                json_body = body
            result = self.transport(url, json_body=json_body, timeout=timeout)
            append_json(self.directory / 'http-responses.jsonl', {'url': url, 'response': result,
                        'sent_monotonic_ns':sent_ns,'received_monotonic_ns':time.monotonic_ns()})
            return result
        self.post = post
        self.schema = ServerCaptureSchema('hidden_states', 'target_last_hidden_states',
                                          (('input_ids', 'input_ids', ()), ('loss_mask', 'loss_mask', ())))
        self.adapter_type = SGLangServerCaptureAdapter
        self.contract = CaptureConfig.from_strategy(required_features={'input_ids', 'loss_mask', 'hidden_states', 'target_last_hidden_states'},
                                                     aux_hidden_state_layer_ids=tuple(self.model['layers']),
                                                     target_repr='hidden_state', target_hidden_size=self.model['hidden_size'])
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='actual-fault-client')
        self.expected_keys = set()

    def warm_ordinary_checkpoint(self, row, label):
        """At most two exact-row normal requests; actual KV hit, unchanged HS."""
        from specforge.inference.capture_protocol import CaptureProtocol
        capability=CaptureProtocol(self.base_url,self.post,self.timeout).capability()
        keys=('ready_blocks','ready_bytes','cache_bytes');probes=[]
        for _ in range(2):
            before=self.cleanup()
            response=self.transport(self.base_url+'/generate',json_body={
                'input_ids':list(row['input_ids']),'extra_key':capability['capture_namespace'],
                'sampling_params':{'temperature':0.0,'max_new_tokens':0}},timeout=self.timeout)
            after=self.cleanup();meta=response['meta_info'];cached=meta['cached_tokens']
            unchanged=all(before['cache'][key]==after['cache'][key] for key in keys)
            probe={'input_ids':list(row['input_ids']),'cached_tokens':cached,
                   'has_spec_capture':'spec_capture' in meta,'hs_cache_unchanged':unchanged,
                   'namespace':capability['capture_namespace'],'response':response}
            probes.append(probe);atomic_json(self.directory/f'ordinary-kv-probes-{label}.json',probes)
            if (type(cached) is not int or not 0<=cached<len(row['input_ids'])
                    or probe['has_spec_capture'] or not unchanged):
                raise AssertionError('ordinary KV prewarm changed HS or violated exact-row capture-free N-1 bounds')
            if cached>0:return probes
        raise AssertionError('bounded same-row ordinary prewarm did not prove an actual positive KV checkpoint')

    def task(self, task_id, row):
        from specforge.runtime.contracts import PromptTask
        return PromptTask(task_id=task_id, run_id=self.run_id, source_id='actual-faults',
                          payload={name: list(row[name]) for name in ('input_ids', 'loss_mask')}, max_length=len(row['input_ids']))

    def submit(self, task_id, row, *, rid=None, force_full=False, http_timeout=None):
        rid = rid or 'actual-' + uuid.uuid4().hex
        def run():
            self.local.rid, self.local.force_full = rid, force_full
            adapter = self.adapter_type(self.base_url, self.store, run_id=self.run_id, algorithm='dflash',
                                        schema=self.schema, server_capture='patch', post_fn=self.post, control_fn=self.post,
                                        timeout_s=http_timeout or self.timeout)
            task = self.task(task_id, row)
            spec = adapter._spec_capture_payload(task)
            self.expected_keys.update(self.store._tkey(spec['sample_id'], 1, name)
                                      for name in [*spec['features'].values(), *(item['name'] for item in spec['passthrough'])])
            return adapter.produce_refs([task], capture=self.contract)[0]
        return self.executor.submit(run), rid

    def fetch(self, future, row, *, reference=None):
        from specforge.runtime.contracts import SampleRef
        ref = future.result(timeout=self.timeout)
        if not isinstance(ref, SampleRef):
            raise AssertionError('expected successful full SampleRef: ' + repr(ref))
        features, handle = self.store.get(ref)
        try:
            output = check_features(features, row, self.model, reference=reference,
                                    **self.config['feature_tolerances'])
            append_json(self.directory / 'sample-results.jsonl', {'ref': asdict(ref), 'complete_features': True})
        finally:
            self.store.release(handle)
        return output, ref

    def cleanup(self):
        self.store.discard_external_attempts(reason='actual-fault-cleanup')
        self.store.drain_pending_removals(max_attempts=100, retry_interval_s=.02)
        health = self.store.health()
        for name in ('resident_samples', 'provisional_external', 'active_leases', 'release_pending',
                     'capture_pending_begin', 'capture_fences', 'capture_ack_pending', 'capture_read_intents', 'capture_active_reads'):
            if health.get(name, 0):
                raise AssertionError(f'consumer/producer retained {name}: {health}')
        left = [key for key in self.expected_keys if self.store._store_exists(key)]
        if left:
            raise AssertionError('exact sample keys remain after ownership drain: ' + str(left))
        from specforge.inference.capture_protocol import CaptureProtocol
        protocol = CaptureProtocol(self.base_url, self.post, self.timeout)
        capability = protocol.capability()
        stats = protocol.call({'action': 'stats', 'server_epoch': capability['server_epoch']})
        assert_clean_server(stats)
        atomic_json(self.directory / 'last-clean-stats.json', {'server': stats, 'store': health, 'keys_checked': len(self.expected_keys)})
        return stats

    def close(self):
        self.executor.shutdown(wait=True)
        self.store.close()


def arm(directory, case_id, kind, rid, sample_id, ranks, *, require_cow=False, **options):
    value = {'case_id': case_id, 'kind': kind, 'rid': rid, 'sample_id': sample_id,
             'ranks': ranks, 'require_cow': require_cow, **options}
    atomic_json(Path(directory) / 'arm.json', value)
    return value


def assert_fault_receipts(directory, case_id, tp_size, targets, *, require_cow):
    rows = rank_events(directory, case_id)
    fired = {row['rank'] for row in rows if row['event'] == 'fault_fired'}
    if fired != set(targets):
        raise AssertionError(f'one-shot fault did not fire exactly on target ranks: {fired} != {targets}')
    traces={rank:[(row['transaction'],row['method']) for row in rows
                  if row['rank']==rank and row['event']=='admit_collective'] for rank in range(tp_size)}
    if not traces[0] or any(trace!=traces[0] for trace in traces.values()):
        raise AssertionError('actual admission collective sequence differs across ranks: '+repr(traces))
    if require_cow:
        for rank in range(tp_size):
            proofs = [row for row in rows if row['rank'] == rank and row['event'] == 'after_actual_match' and row['fresh_cow'] and row['transaction'] == 1]
            if not proofs:
                raise AssertionError(f'rank{rank}: no actual fresh Mamba COW allocation observed')
            returned = [row for row in rows if row['rank'] == rank and row['event'] == 'admit_return' and row['transaction'] == 1]
            if not returned:
                raise AssertionError('no actual admission return receipt')
            fresh = [slot for row in proofs for slot in [row['slot']]]
            freed = [slot for row in returned for slot in row['freed_slots']]
            # Match-error may rematch and succeed, but the first failed COW must
            # have been returned. Refusal / after-append failures return all.
            if not all(slot in freed for slot in fresh):
                raise AssertionError(f'fresh COW slot not freed on rank{rank}: {fresh}, {freed}')
    return rows


def assert_prefill_retract_receipts(directory, case_id, tp_size, diagnostics):
    from actual_faults.rank_hooks import assert_retract_allocation_return
    rows = assert_fault_receipts(directory, case_id, tp_size, list(range(tp_size)), require_cow=False)
    plans, old_plans = [], []
    for rank in range(tp_size):
        applied = [row for row in rows if row['rank'] == rank and row['event'] == 'prefill_retract_applied']
        forwarded = [row for row in rows if row['rank'] == rank and row['event'] == 'prefill_retract_forward']
        admissions = [row for row in rows if row['rank'] == rank and row['event'] == 'admit_return']
        fired = [row for row in rows if row['rank'] == rank and row['event'] == 'fault_fired']
        if len(fired) != 1 or len(applied) != 1 or len(forwarded) != 1 or len(admissions) != 2:
            raise AssertionError('prefill retract lacks exactly one real retract/forward and two admissions on every rank')
        before, after = applied[0], forwarded[0]
        if (before['old_plan']['execution_epoch'] != 0 or before['old_plan']['L'] <= 0
                or before['old_plan']['H'] < before['old_plan']['L']
                or before['execution_epoch'] != 1 or before['retraction_count'] != 1
                or not before['force_full'] or not before['old_drains_done'] or not before['pre_forward_stream_drained']
                or after['plan']['execution_epoch'] != 1 or after['plan']['L'] != 0
                or after['retraction_count'] != 1 or after['transaction'] != 2):
            raise AssertionError('prefill retract did not prove real old ownership drain and full epoch1 retry')
        if rank==0:
            if (before['controller_before'] is None or before['controller_before']['output_reservations'] != 1
                    or before['controller_before']['output_bytes'] <= 0 or before['cache_before'] is None
                    or before['cache_before']['active_leases'] <= 0):
                raise AssertionError('writer rank did not prove an active old HS lease and output reservation')
        if [row['transaction'] for row in admissions] != [1, 2] or not all(row['member'] for row in admissions):
            raise AssertionError('prefill retract did not use two genuine committed admissions')
        assert_retract_allocation_return(before['allocation_return'], before['before'], before['after'])
        old_plans.append({key: before['old_plan'][key] for key in ('execution_epoch', 'N', 'L', 'span', 'plan_digest')})
        plans.append({key: after['plan'][key] for key in ('execution_epoch', 'N', 'L', 'span', 'plan_digest')})
    if any(plan != old_plans[0] for plan in old_plans[1:]):
        raise AssertionError('prefill retract old positive-L plans differ across TP ranks')
    if any(plan != plans[0] for plan in plans[1:]):
        raise AssertionError('prefill retract retry plans differ across TP ranks')
    if any(diagnostics.get(key) != plans[0][key] for key in ('execution_epoch', 'N', 'L', 'span')):
        raise AssertionError('delivered prefill retract diagnostics differ from the actual epoch1 forward')
    return rows


def unresolved_ownership(store):
    """Snapshot retained identities; absence of remote keys is not retirement."""
    with store._lock:
        fences={sid:{'wire':dict(fence.wire),'names':list(fence.names),
                     'writer_drained':fence.writer_drained,'removed':fence.removed,
                     'acknowledged':fence.acknowledged,'cleanup_token':fence.cleanup_token}
                for sid,fence in store._capture_fences.items()}
        begins={sid:{'capability':intent[1],'payload':intent[2],'tokens':intent[3],
                     'attempt_id':intent[4],'names':intent[5]}
                for sid,intent in store._capture_pending_begin.items()}
        readers={owner:{key:value for key,value in intent.items() if key not in ('fence','handle')}
                 for owner,intent in store._capture_read_intents.items()}
    return {'health':store.health(),'fences':fences,'begin_intents':begins,'reader_intents':readers,
            'cleanup_proven':False,'scope':'old server epoch unavailable; retain ownership, no forced removal'}


def assert_restart_fail_closed(client, ref, new_epoch, retry):
    """Exercise actual tracked stale-ref/control/store paths before native I/O."""
    from contextlib import ExitStack
    from actual_faults.rank_hooks import patched_attribute
    from specforge.inference.capture_protocol import CaptureControlRejected
    store=client.store
    with store._lock:
        fence=store._capture_fences[ref.sample_id]
        before=dict(fence.wire)
    old_epoch=before['server_epoch']
    if old_epoch==new_epoch or ref.metadata['capture_fence']['server_epoch']!=old_epoch:
        raise AssertionError('restart fixture did not change the exact old reference epoch')
    physical={name:0 for name in ('fetch','remove','put','generate')}
    def forbidden(kind):
        def call(*args,**kwargs):
            physical[kind]+=1
            raise AssertionError('stale restart ownership reached forbidden physical '+kind)
        return call
    original_transport=client.transport
    controls=[]
    def transport(url, *, json_body, timeout):
        if url.endswith('/generate'):
            return forbidden('generate')()
        if url.endswith('/spec_capture/control') and json_body.get('action')!='capability':
            controls.append({name:json_body.get(name) for name in ('action','server_epoch','sample_id','attempt_id')})
        return original_transport(url,json_body=json_body,timeout=timeout)
    errors={}
    with ExitStack() as stack:
        for name,kind in (('_get_tensors','fetch'),('_store_remove','remove'),('_store_put_tensor','put')):
            stack.enter_context(patched_attribute(store,name,forbidden(kind)))
        stack.enter_context(patched_attribute(client,'transport',transport))
        for label,operation in (('old-ref-get',lambda:store.get(ref)),
                                ('old-resolve',lambda:fence.control('resolve',cancel=False)),
                                ('old-store-abort',lambda:store.abort(ref.sample_id,reason='server-restart-probe',attempt_token=fence.token)),
                                ('same-sample-retry',retry)):
            try:operation()
            except CaptureControlRejected as exc:errors[label]=str(exc)
            else:raise AssertionError('stale restart operation did not explicitly reject old ownership: '+label)
    if any(physical.values()):
        raise AssertionError('restart crossed a forbidden physical boundary: '+repr(physical))
    if not controls or any(value['server_epoch']!=old_epoch for value in controls):
        raise AssertionError('stale restart operations did not preserve exact original epoch identity')
    with store._lock:
        if (store._capture_fences.get(ref.sample_id) is not fence or fence.wire!=before
                or fence.removed or fence.acknowledged or fence.cleanup_token is not None):
            raise AssertionError('stale restart refusal discarded or retired uncertain ownership')
    return {'old_epoch':old_epoch,'new_epoch':new_epoch,'explicit_rejections':errors,
            'physical_operations':physical,'old_controls':controls,'cleanup_proven':False,
            'unresolved_ownership':unresolved_ownership(store)}
