#!/usr/bin/env python3
"""Formal actual-training Plan A benchmark. Default is a resource-free plan.

--execute is for root-assigned resources only. Production sources remain
unchanged: child workers install observational wrappers and run the real CLI.
The independent run_validation.py and its numerical acceptance stay read-only.
"""
from __future__ import annotations

import argparse
import atexit
from collections import deque
from contextlib import contextmanager
import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback

from performance_helpers import (GROUPS, WORKLOADS, OFFICIAL_PATCH_SHA256, ProcessSampler,
    aggregate_capture, aggregate_training, append_json, build_workload, canonical_digest,
    comparison_contract, compare_runs, content_fingerprint, isolate_config, load_rows, rewrite_capture_argv,
    rewrite_worker_argv, validate_rdma_proof, write_json, optimizer_step_number,
    serialize_for_epoch_order, audit_consumed_order, assigned_gpu_indices, performance_run_id,
    POLL_DEFAULTS, validate_polling_policy, polling_policy_call, audit_polling_receipts)

OFFICIAL_SF = '716b02d7bf886cf5761685c0e57de887814ff15f'
OFFICIAL_SG = '74be0a60ebfcc7d8e261096bca445ac51e383f03'
SCRIPT = Path(__file__).resolve()
_PENDING_EVENTS = deque()
_EVENTS_LOCK = threading.Lock()
_CONTEXT = threading.local()
_INSTALLED = False


class DevicePublicationUnavailable(RuntimeError):
    """Only an actual native CUDA-registration/RDMA-transfer proof may raise this."""


def verify_runtime_imports():
    selected = {'specforge':Path(os.environ['PLANA_PERF_SPECFORGE_SOURCE']).resolve(),
                'sglang':Path(os.environ['PLANA_PERF_SGLANG_SOURCE']).resolve()/'python'}
    imports = {}
    for name, root in selected.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise AssertionError(f'wrong actual child import {name}: {path}')
        imports[name] = str(path)
    _emit('runtime-imports', {'imports':imports,'python':sys.executable})


def _event_path(kind):
    return Path(os.environ['PLANA_PERF_RUN_DIR']) / 'events' / f'{kind}-{os.getpid()}.jsonl'


def _emit(kind, values):
    append_json(_event_path(kind), {'pid': os.getpid(), 'role': os.environ.get('PLANA_PERF_ROLE'),
                                  'wall_time': time.time(), 'monotonic': time.perf_counter(), **values})


def _deferred(stage, start, end, context):
    with _EVENTS_LOCK:
        _PENDING_EVENTS.append((stage, start, end, dict(context)))


def _poll_events(final=False):
    with _EVENTS_LOCK:
        pending = list(_PENDING_EVENTS)
        _PENDING_EVENTS.clear()
    for stage, start, end, context in pending:
        if final:
            end.synchronize()  # At process drain only, never measured hot path.
        if end.query():
            _emit('server-stages', {'stage': stage, 'milliseconds': start.elapsed_time(end), **context})
        else:
            with _EVENTS_LOCK:
                _PENDING_EVENTS.append((stage, start, end, context))


def _scalar_context(batch):
    return {'sample_ids': [str(req.spec_capture['sample_id']) for req in batch.reqs if req.spec_capture],
            'N': [len(req.origin_input_ids) for req in batch.reqs],
            'L': [len(req.prefix_indices) for req in batch.reqs]}


def install_server_observers():
    """Patch function seams in each owned scheduler process, retaining no tensors."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    verify_runtime_imports()
    import torch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
    from sglang.srt.spec_capture_sink import SpecCaptureSink

    original_run = Scheduler.run_batch
    @functools.wraps(original_run)
    def run_batch(self, batch, *args, **kwargs):
        previous = getattr(_CONTEXT, 'capture', None)
        context = _scalar_context(batch)
        _CONTEXT.capture = context if self.ps.attn_tp_rank == 0 and context['sample_ids'] else None
        try:
            return original_run(self, batch, *args, **kwargs)
        finally:
            _CONTEXT.capture = previous
    Scheduler.run_batch = run_batch

    def cuda_wrapper(original, stage):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            context = getattr(_CONTEXT, 'capture', None)
            if not context or not torch.cuda.is_available():
                return original(self, *args, **kwargs)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
            result = original(self, *args, **kwargs)
            end.record(torch.cuda.current_stream())
            _deferred(stage, start, end, context)
            return result
        return wrapped
    TpModelWorker.forward_batch_generation = cuda_wrapper(TpModelWorker.forward_batch_generation, 'official_forward_cuda_ms')
    GenerationBatchResult.copy_to_cpu = cuda_wrapper(GenerationBatchResult.copy_to_cpu, 'copy_result_combined_cuda_ms')

    original_result = Scheduler.process_batch_result
    @functools.wraps(original_result)
    def process_result(self, *args, **kwargs):
        result = original_result(self, *args, **kwargs)
        _poll_events()
        return result
    Scheduler.process_batch_result = process_result
    original_state = Scheduler.get_internal_state
    @functools.wraps(original_state)
    def internal_state(self, *args, **kwargs):
        result = original_state(self, *args, **kwargs)
        _poll_events()
        return result
    Scheduler.get_internal_state = internal_state

    for method, stage in (('_append_spec_capture_states', 'capture_slice_host_wall_ms'),
                          ('_sink_spec_capture', 'capture_coalesce_host_wall_ms')):
        original = getattr(SchedulerBatchResultProcessor, method)
        def make_wrapper(call, stage_name):
            @functools.wraps(call)
            def wrapped(self, *args, **kwargs):
                req = kwargs.get('req') or (args[0] if args else None)
                started = time.perf_counter()
                result = call(self, *args, **kwargs)
                if self.output_streamer.ps.attn_tp_rank == 0 and req is not None and req.spec_capture:
                    _emit('server-stages', {'stage': stage_name, 'milliseconds': (time.perf_counter()-started)*1000,
                                           'sample_ids': [req.spec_capture['sample_id']]})
                return result
            return wrapped
        setattr(SchedulerBatchResultProcessor, method, make_wrapper(original, stage))

    original_put = SpecCaptureSink.put_samples
    @functools.wraps(original_put)
    def put_samples(self, samples):
        # GPU-source proof concerns actual model captures, not merely flags.
        cuda_sources = all(t is not None and t.is_cuda for _, aux, hidden in samples for t in (aux, hidden))
        started = time.perf_counter()
        result = original_put(self, samples)
        _emit('server-stages', {'stage': 'publication_register_put_unregister_wall_ms',
            'milliseconds': (time.perf_counter()-started)*1000,
            'sample_ids': [spec['sample_id'] for spec, _, _ in samples], 'cuda_sources': cuda_sources})
        return result
    SpecCaptureSink.put_samples = put_samples
    original_submit = SpecCaptureSink.submit_samples
    @functools.wraps(original_submit)
    def submit_samples(self, samples):
        ids = [spec['sample_id'] for spec, _, _ in samples]
        started = time.perf_counter()
        future = original_submit(self, samples)
        def finished(completed):
            _emit('server-stages', {'stage': 'publication_submit_done_wall_ms',
                'milliseconds': (time.perf_counter()-started)*1000, 'sample_ids': ids,
                'successful': completed.exception() is None})
        future.add_done_callback(finished)
        return future
    SpecCaptureSink.submit_samples = submit_samples
    atexit.register(_poll_events, final=True)


def _validate_sample(ref, features, workload, schema):
    import torch
    ids = features['input_ids'].detach().cpu()
    mask = features['loss_mask'].detach().cpu()
    N = int(ref.num_tokens)
    if tuple(ids.shape) != (1, N) or tuple(mask.shape) != (1, N) or ids.dtype != torch.int64 or mask.dtype != torch.int64:
        raise AssertionError('full sample IDs/mask shape or dtype mismatch')
    fingerprint = content_fingerprint(ids.reshape(-1).tolist(), mask.reshape(-1).tolist())
    allowed = workload['fingerprints']
    if fingerprint not in allowed:
        raise AssertionError('full sample token/mask content differs from this exact workload')
    if set(features)!=set(ref.feature_specs):
        raise AssertionError('full returned feature set differs from SampleRef')
    for name, spec in ref.feature_specs.items():
        tensor=features[name]
        if tuple(tensor.shape)!=tuple(spec.shape) or str(tensor.dtype).removeprefix('torch.')!=spec.dtype:
            raise AssertionError(f'full SampleRef tensor metadata mismatch: {name}')
    if 'attention_mask' in features and not bool((features['attention_mask']==1).all()):
        raise AssertionError('unexpected attention mask in standard unpadded text sample')
    required = {'hidden_states': (1, N, schema['hidden_size'] * len(schema['layers'])),
                'target_last_hidden_states': (1, N, schema['hidden_size'])}
    for name, shape in required.items():
        tensor = features[name]
        if tuple(tensor.shape) != shape or tensor.dtype != getattr(torch, schema['dtype']):
            raise AssertionError(f'full feature shape/dtype mismatch: {name}')
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError(f'nonfinite full feature: {name}')
    if ref.metadata.get('generation') != 1:
        raise AssertionError('performance sample generation changed')
    return fingerprint


def install_polling_policy():
    """Official uses its existing callable APIs; candidate must use its config."""
    import specforge.launch as launch
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue
    from specforge.runtime.data_plane.ref_distributor import RefDistributor
    policy = validate_polling_policy(json.loads(os.environ['PLANA_PERF_POLL_POLICY']))
    mode = os.environ['PLANA_PERF_POLL_MODE']
    for owner, name, argument, field in (
        (launch, 'build_disagg_online_producer', 'backpressure_poll_s', 'producer_backpressure_poll_s'),
        (StreamingRefQueue, '__init__', 'poll_s', 'consumer_ref_poll_s'),
        (RefDistributor, '__init__', 'poll_s', 'ref_distributor_poll_s')):
        setattr(owner, name, polling_policy_call(getattr(owner, name), argument=argument,
            field=field, value=policy[field], mode=mode, emit=lambda row: _emit('polling-policy', row)))


def install_worker_observers():
    """Observe actual optimizer, data loader and SampleRef path in CLI workers."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    verify_runtime_imports()
    install_polling_policy()
    import torch
    from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter, _SessionPost
    from specforge.runtime.contracts import SampleRef
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
    from specforge.training.controller import TrainerController, TrainerCore

    workload = json.loads(Path(os.environ['PLANA_PERF_WORKLOAD_JSON']).read_text())
    schema = json.loads(os.environ['PLANA_PERF_SCHEMA_JSON'])
    isolation = {}
    seen_extra_keys = set()
    original_post = _SessionPost.__call__
    @functools.wraps(original_post)
    def post(self, url, json_body, timeout):
        if url.endswith('/generate'):
            specs = json_body.get('spec_capture') or []
            extras = json_body.get('extra_key') or []
            for spec, extra in zip(specs, extras, strict=True):
                unique_uuid = isinstance(extra, str) and len(extra) == 32 and extra not in seen_extra_keys
                isolation[spec['sample_id']] = unique_uuid
                seen_extra_keys.add(extra)
        return original_post(self, url, json_body, timeout)
    _SessionPost.__call__ = post

    original_produce = SGLangServerCaptureAdapter.produce_refs
    @functools.wraps(original_produce)
    def produce(self, tasks, *, capture):
        started = time.perf_counter()
        refs = original_produce(self, tasks, capture=capture)
        elapsed = time.perf_counter()-started
        for task, ref in zip(tasks, refs, strict=True):
            if not isinstance(ref, SampleRef):
                _emit('capture-failures', {'failure': repr(ref), 'task_id': task.task_id})
                continue
            diagnostic = dict(ref.metadata.get('capture_diagnostics') or {})
            if not isinstance(diagnostic.get('L'), int):
                if not isolation.get(ref.sample_id):
                    raise AssertionError('legacy full-prefill accounting lacks fresh UUID isolation proof')
                diagnostic.update(N=len(task.payload['input_ids']), L=0, span=[0,len(task.payload['input_ids'])],
                                  L_source='fixed_official_full_prefill_contract')
            _emit('captures', {**diagnostic, 'sample_id': ref.sample_id,
                'attempt_id': ref.metadata.get('capture_fence', {}).get('attempt_id', ref.source_task_id),
                'produce_batch_wall_s': elapsed, 'produce_batch_size': len(tasks),
                'feature_specs': {name: dataclasses.asdict(value) for name,value in ref.feature_specs.items()},
                'ref': dataclasses.asdict(ref), 'timings': diagnostic.get('timings') or {}})
        return refs
    SGLangServerCaptureAdapter.produce_refs = produce

    original_get = MooncakeFeatureStore.get
    @functools.wraps(original_get)
    def get(self, ref, *args, **kwargs):
        started = time.perf_counter()
        features, handle = original_get(self, ref, *args, **kwargs)
        fetched = time.perf_counter()
        try:
            fingerprint = _validate_sample(ref, features, workload, schema)
        except BaseException:
            self.release(handle)
            raise
        _emit('fetch-validation', {'sample_id': ref.sample_id, 'fetch_wall_s': fetched-started,
            'validation_wall_s': time.perf_counter()-fetched, 'content_fingerprint': fingerprint,
            'n_tokens': ref.num_tokens, 'feature_bytes': ref.estimated_bytes})
        return features, handle
    MooncakeFeatureStore.get = get

    original_close = MooncakeFeatureStore.close
    @functools.wraps(original_close)
    def close(self, *args, **kwargs):
        _emit('store-close', {'health_before_close': self.health(), 'store_id': self.store_id})
        return original_close(self, *args, **kwargs)
    MooncakeFeatureStore.close = close

    original_step = TrainerCore.train_step
    @functools.wraps(original_step)
    def train_step(self, batch, *args, **kwargs):
        result = original_step(self, batch, *args, **kwargs)
        _emit('consumed', {'step': optimizer_step_number(args, kwargs),
            'optimizer_stepped': bool(result.optimizer_stepped), 'sample_ids': list(batch.sample_ids)})
        return result
    TrainerCore.train_step = train_step

    original_fit = TrainerController.fit
    @functools.wraps(original_fit)
    def fit(self, data):
        previous = self.logger
        if previous is None:
            raise ValueError('actual training logger must already be enabled; observers do not enable new math/metrics paths')
        initial = hashlib.sha256()
        for name, tensor in self.core.strategy.trainable_module().state_dict().items():
            initial.update(name.encode())
            values = tensor.detach().cpu().contiguous()
            initial.update(values.reshape(-1).view(torch.uint8).numpy().tobytes())
        _emit('draft-initial', {'sha256': initial.hexdigest()})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        boundaries = {}
        def logger(metrics, step):
            if previous is not None:
                previous(metrics, step)
            now = time.perf_counter()
            boundaries[int(step)] = now
            scalar = {name: float(value) for name,value in metrics.items()
                      if name.startswith('perf/') and isinstance(value, (int,float))}
            _emit('training-steps', {'step': int(step), 'metrics': scalar,
                                  'fit_elapsed_s': now-started})
        self.logger = logger
        try:
            result = original_fit(self, data)
            torch.cuda.synchronize()  # Once at the real fit boundary.
            warmup = int(workload['training_warmup_steps'])
            end_step = warmup + int(workload['measured_steps'])
            if warmup not in boundaries or end_step not in boundaries:
                raise AssertionError('actual optimizer-step timing boundaries missing')
            _emit('fit', {'completed_steps': int(result), 'fit_wall_s': time.perf_counter()-started,
                'measured_fit_boundary_wall_s': boundaries[end_step]-boundaries[warmup],
                'warmup_steps': warmup, 'measured_steps': workload['measured_steps'],
                'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'boundary': 'actual CLI TrainerController.fit with producer+loader+optimizer; validation and logging included'})
            return result
        finally:
            self.logger = previous
    TrainerController.fit = fit


def git_identity(path, expected):
    path = Path(path).resolve()
    def git(*argv):
        return subprocess.check_output(['git','-C',str(path),*argv], text=True).strip()
    commit = git('rev-parse','HEAD')
    resolved = git('rev-parse',expected)
    if commit != resolved or git('status','--porcelain'):
        raise ValueError(f'source must be clean exact commit {expected}: {path} {commit}')
    if git('rev-parse','--show-toplevel') != str(path):
        raise ValueError('source path is not its verified git root')
    return {'path': str(path), 'commit': commit, 'branch': git('branch','--show-current')}


def _model_info(cfg, resolved):
    from specforge.training.model_loading import resolve_draft_config
    from specforge.modeling.target.target_utils import load_target_config, target_hidden_size
    draft = resolve_draft_config(cfg, provider=resolved.algorithm.providers.model.draft_config)
    target = load_target_config(cfg.model.target_model_path, trust_remote_code=cfg.model.trust_remote_code)
    layers = list(resolved.algorithm.providers.model.resolve_capture_layers(cfg,draft,target))
    width = target_hidden_size(target)
    return {'hidden_size': width, 'layers': layers, 'dtype': 'bfloat16', 'row_bytes': (len(layers)+1)*width*2,
            'target_config': target.to_dict(), 'draft_config': draft.to_dict()}


def _load_events(run_dir, kind):
    values=[]
    for path in sorted((Path(run_dir)/'events').glob(f'{kind}-*.jsonl')):
        values.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return values


def _setup_from_env(env):
    return {'local_hostname': env['MOONCAKE_LOCAL_HOSTNAME'], 'metadata_server': env['MOONCAKE_METADATA_SERVER'],
        'global_segment_size': int(env['MOONCAKE_GLOBAL_SEGMENT_SIZE']), 'local_buffer_size': int(env['MOONCAKE_LOCAL_BUFFER_SIZE']),
        'protocol': env['MOONCAKE_PROTOCOL'], 'rdma_devices': env.get('MOONCAKE_RDMA_DEVICES',''),
        'master_server_addr': env['MOONCAKE_MASTER_SERVER_ADDR']}


def warm_server(run_dir, cfg, resolved, server, workload):
    """Real capture/fetch/release warmup, before training workers begin."""
    if not workload['prewarm']:
        return
    import torch
    from specforge.algorithms.common.providers import resolve_server_capture_layout
    from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter, ServerCaptureSchema
    from specforge.inference.capture import CaptureConfig
    from specforge.runtime.contracts import PromptTask, SampleRef
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
    schema = json.loads(os.environ['PLANA_PERF_SCHEMA_JSON'])
    layout = resolve_server_capture_layout(resolved.algorithm,cfg,modality='text')
    capture_schema = ServerCaptureSchema(aux_feature=layout.aux_feature,last_hidden_feature=layout.last_hidden_feature,
        passthrough=layout.passthrough,attention_mask_feature=layout.attention_mask_feature)
    prewarm_id = performance_run_id(run_dir)+'-prewarm'
    store = MooncakeFeatureStore(store_id=prewarm_id,setup_kwargs=_setup_from_env(server.command.env))
    adapter = SGLangServerCaptureAdapter(server.command.env['DISAGG_SERVER_URLS'],store,run_id=prewarm_id,
        algorithm='dflash',schema=capture_schema,server_capture='patch')
    contract=CaptureConfig.from_strategy(required_features={'input_ids','loss_mask','hidden_states','target_last_hidden_states'},
        aux_hidden_state_layer_ids=tuple(schema['layers']),target_repr='hidden_state',target_hidden_size=schema['hidden_size'])
    try:
        for index,row in enumerate(workload['prewarm']):
            task=PromptTask(task_id=f'prewarm-{index:05d}',run_id=prewarm_id,source_id='performance-warmup',
                payload={key:row[key] for key in ('input_ids','loss_mask')},max_length=len(row['input_ids']))
            ref,=adapter.produce_refs([task],capture=contract)
            if not isinstance(ref,SampleRef):
                raise AssertionError(f'prewarm failed: {ref}')
            features,handle=store.get(ref)
            _validate_sample(ref,features,workload,schema)
            store.release(handle)
            store.drain_sample_removals([ref.sample_id])
            append_json(Path(run_dir)/'prewarm.jsonl',{'sample_id':ref.sample_id,'N':ref.num_tokens,'metadata':ref.metadata})
        store.discard_external_attempts(reason='performance-prewarm-finally')
        store.drain_pending_removals()
        health=store.health()
        for name in ('resident_samples','provisional_external','active_leases','release_pending'):
            if health[name]:
                raise AssertionError(f'prewarm retained {name}')
        protocol=getattr(adapter,'_capture_protocol',None)
        if protocol is not None:
            capability=protocol.capability()
            if capability is not None:
                stats=protocol.call({'action':'stats','server_epoch':capability['server_epoch']})
                write_json(Path(run_dir)/'prewarm-server-stats.json',stats)
                if stats['active_attempts'] or stats['active_readers'] or stats['pending_deliveries']:
                    raise AssertionError('prewarm delivery fence not drained')
    finally:
        store.close()


def _cleanup_probe_resources(sink, store, handle, sid, names, *, writer_finished=False,
                             read_started=False, read_finished=False, report_path=None, store_id=None):
    """Remove probe keys only after proven writer/reader drain; always close clients."""
    errors=[]
    def attempt(label,operation):
        try:
            operation()
            return True
        except BaseException as exc:
            errors.append(f'{label}: {type(exc).__name__}: {exc}')
            return False
    writer_safe=sink is None
    if sink is not None:
        executor_drained=attempt('sink drain',lambda:sink._executor.shutdown(wait=True))
        raw=getattr(sink,'_store',None)
        close=getattr(raw,'close',None)
        def close_transport():
            rc=close()
            if rc is not None and int(rc)!=0:
                raise RuntimeError(f'native writer transport close failed ({rc})')
        transport_stopped=attempt('sink transport close',close_transport) if close is not None else False
        writer_safe=executor_drained and (writer_finished or transport_stopped or raw is None)
    reader_safe=not read_started or read_finished
    if store is not None:
        # A failed native get may still own DMA receive storage with no returned handle.
        # Closing its owned transport preserves the store's quarantine protocol; no
        # removal RPC is issued through an uncertain receiver or before writer drain.
        if writer_safe and reader_safe:
            released=handle is None or attempt('read release',lambda:store.release(handle))
            if released:
                attempt('sample removal drain',lambda:store.drain_sample_removals([sid]))
                for name in names:
                    def remove(name=name):
                        key=store._tkey(sid,1,name)
                        if not store._store_remove(key) or store._store_exists(key):
                            raise RuntimeError(f'probe key not removed: {key}')
                    attempt('probe key cleanup',remove)
        else:
            errors.append('probe keys retained: writer/receiver drain unproven; owned transport/process teardown required')
        health=getattr(store,'health',None)
        before_close=None
        if health is not None:
            try:before_close=health()
            except BaseException as exc:errors.append(f'health snapshot: {type(exc).__name__}: {exc}')
        attempt('store close',store.close)
    else:
        before_close=None
    if store_id is None:store_id=getattr(store,'store_id',None)
    report={'store_id':store_id,'sample_id':sid,'generation':1,'feature_names':list(names),
            'physical_keys':[f'{store_id}/{sid}/g1/{name}' for name in names] if store_id is not None else None,
            'writer_safe':writer_safe,'reader_safe':reader_safe,
            'key_removal_authorized':writer_safe and reader_safe,'health_before_close':before_close,'errors':errors}
    if report_path is not None:
        attempt('cleanup report',lambda:write_json(report_path,report))
    if errors:raise RuntimeError('; '.join(errors))


def rdma_cuda_probe(env, output):
    """Actual CUDA-pointer registration/publication/fetch/removal proof."""
    import torch
    from sglang.srt.spec_capture_sink import SpecCaptureSink, check_rdma_environment, gpu_put_enabled
    from specforge.runtime.contracts import FeatureSpec, SampleRef
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
    check_rdma_environment()
    if env['MOONCAKE_PROTOCOL']!='rdma' or env['SGLANG_SPEC_CAPTURE_GPU_PUT']!='1' or not gpu_put_enabled():
        raise ValueError('probe requires explicit RDMA GPU publication')
    aux=torch.arange(128,device='cuda',dtype=torch.float32).reshape(32,4).to(torch.bfloat16)
    hidden=aux[:,:2].contiguous()
    cuda_ptrs={aux.untyped_storage().data_ptr(),hidden.untyped_storage().data_ptr()}
    store_id=performance_run_id(Path(output).parent)+'-rdma-proof'
    sid=store_id+':cuda-proof'
    names=('hidden_states','target_last_hidden_states')
    sink=store=handle=None
    writer_finished=read_started=read_finished=False
    try:
        # Connect the cleanup client before any write can create a partial sample.
        store=MooncakeFeatureStore(store_id=store_id,setup_kwargs=_setup_from_env(env))
        sink=SpecCaptureSink(aux_layer_ids=[0]);raw=sink._connect();registrations=[];puts=[];registration_failed=[False]
        class Proxy:
            def __getattr__(self,name):return getattr(raw,name)
            def register_buffer(self,ptr,size):
                try:
                    rc=raw.register_buffer(ptr,size)
                except Exception:
                    if ptr in cuda_ptrs:registration_failed[0]=True
                    raise
                if ptr in cuda_ptrs:
                    registrations.append(0 if rc is None else int(rc))
                    registration_failed[0] = registration_failed[0] or (rc is not None and int(rc)!=0)
                return rc
            def batch_put_from(self,*args):
                result=raw.batch_put_from(*args);puts.extend([0]*len(args[0]) if result is None else result);return result
        sink._store=Proxy()
        try:
            result=sink.put_samples([({'store_id':store_id,'sample_id':sid,'gen':1,'features':{'aux':names[0],'last_hidden':names[1]}},aux,hidden)])[0]
            writer_finished=True
        except Exception as exc:
            if registration_failed[0] or any(value is not None and int(value)<0 for value in puts):
                raise DevicePublicationUnavailable('actual CUDA registration/RDMA transfer failed: '+str(exc)) from None
            raise
        ref=SampleRef(sample_id=sid,run_id=store_id,source_task_id='proof',feature_store_uri=f'mooncake://{store_id}/{sid}',
            feature_keys={name:f'{sid}/{name}' for name in result['features']},
            feature_specs={name:FeatureSpec(name,tuple(meta['shape']),meta['dtype']) for name,meta in result['features'].items()},
            strategy='dflash',num_tokens=32,metadata={'generation':1})
        store.adopt(ref)
        read_started=True
        features,handle=store.get(ref)
        read_finished=True
        equal=torch.equal(features[names[0]],aux.cpu().unsqueeze(0)) and torch.equal(features[names[1]],hidden.cpu().unsqueeze(0))
        store.release(handle);handle=None;store.drain_sample_removals([sid])
        removed=all(not store._store_exists(store._tkey(sid,1,name)) for name in names)
        proof={'protocol':'rdma','gpu_put':True,'cuda_source':aux.is_cuda and hidden.is_cuda,
            'run_id':store_id,'store_id':store_id,'sample_id':sid,
            'cuda_register_status':0 if len(registrations)==2 and all(value==0 for value in registrations) else -1,
            'put_status':0 if puts and all(value is None or int(value)==0 for value in puts) else -1,
            'fetch_equal':equal,'keys_removed':removed,'rdma_devices':env['MOONCAKE_RDMA_DEVICES'],
            'local_memcpy_setting':os.environ.get('MC_STORE_MEMCPY'),
            'forbidden_overrides':{key:os.environ[key] for key in ('MC_FORCE_TCP','MC_MS_AUTO_DISC','MC_USE_TENT','MC_USE_TEV1') if key in os.environ},
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'cuda_device':torch.cuda.get_device_name(),'peermem_present':Path('/sys/module/nvidia_peermem').exists()}
        write_json(output,proof);validate_rdma_proof(proof,expected_devices=env['MOONCAKE_RDMA_DEVICES'])
    finally:
        _cleanup_probe_resources(sink,store,handle,sid,names,writer_finished=writer_finished,
            read_started=read_started,read_finished=read_finished,report_path=Path(output).with_name('probe-cleanup.json'),store_id=store_id)


@contextmanager
def scoped_environment(values):
    previous={key:os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def run_owned_probe(popen, argv, *, env, terminate, timeout_s, grace_s):
    """A probe is a detached, bounded child and is drained on every exit path."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('positive native probe timeout required')
    probe=popen(argv,env=env,start_new_session=True)
    if probe.pid == os.getpid() or probe.pid <= 0:
        raise RuntimeError('probe must be a distinct owned child session')
    try:
        return probe.wait(timeout=timeout_s)
    finally:
        terminate([probe],grace_s=grace_s,
                  exited_group_leaders=[probe] if probe.poll() is not None else [])


def finish_owned_run(launch_module, real_stop, started_services, processes, sampler, run_dir, *, grace_s,
                     result=None,workload=None):
    """Attempt every cleanup phase even if an earlier phase raises/interruption repeats."""
    errors=[]
    def attempt(label, operation):
        try:return operation()
        except BaseException as exc:errors.append(f'{label}: {type(exc).__name__}: {exc}')
    attempt('services',lambda:real_stop(started_services,grace_s=grace_s))
    # Every spawn wrapper enforces start_new_session=True; the orchestrator is never tracked here.
    def drain_children():
        live=[process for process in processes if process.pid != os.getpid() and process.poll() is None]
        launch_module._terminate_processes(live,grace_s=grace_s)
    attempt('owned children',drain_children)
    sampler_status=attempt('sampler',sampler.stop)
    attempt('exit records',lambda:write_json(Path(run_dir)/'owned-processes-final.json',
        [{'pid':p.pid,'exit':p.poll()} for p in processes]))
    if result is not None:
        result['resource_sampler']=sampler_status or {}
        attempt('final resource observations',lambda:finalize_resource_observations(run_dir,result,workload))
    if errors:raise RuntimeError('owned run cleanup failed: '+'; '.join(errors))
    if any(process.poll() is None for process in processes):
        raise RuntimeError('owned child remained live after cleanup')
    return sampler_status


def final_server_and_keys(plan, run_dir):
    """Runs after workers exit, while owned capture/Mooncake services live."""
    import requests
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
    server=next(service for service in plan.services if service.command.label.startswith('capture-server-'))
    url=server.command.env['DISAGG_SERVER_URLS'].rstrip('/')+'/spec_capture/control'
    requests.get(server.command.env['DISAGG_SERVER_URLS'].rstrip('/')+'/server_info',timeout=10).raise_for_status()
    response=requests.post(url,json={'protocol_version':1,'action':'capability'},timeout=10)
    final={'control_status':response.status_code}
    if response.status_code!=404:
        response.raise_for_status();cap=response.json()
        response=requests.post(url,json={'protocol_version':1,'action':'stats','server_epoch':cap['server_epoch']},timeout=10)
        response.raise_for_status();stats=response.json();final['stats']=stats
        for name in ('active_attempts','active_readers','pending_deliveries'):
            if stats[name]:raise AssertionError(f'final server retained {name}')
        if cap['cache_enabled']:
            for name in ('active_leases','fill_reservations','writing_blocks'):
                if stats['cache'][name]:raise AssertionError(f'final cache retained {name}')
            for name in ('pinned_bytes','output_bytes','device_bytes','output_reservations','pending_executions','pending_copies','quarantined_executions'):
                if stats['controller'][name]:raise AssertionError(f'final controller retained {name}')
            limits=cap['limits'];controller=stats['controller'];cache=stats['cache']
            for value,limit in ((cache['cache_peak_bytes'],limits['cache_bytes']),
                (controller['pinned_peak_bytes'],limits['pinned_capacity_bytes']),
                (controller['pinned_arena_bytes'],limits['pinned_capacity_bytes']),
                (controller['output_peak_bytes'],limits['output_bytes']),
                (controller['device_peak_bytes'],limits['device_bytes'])):
                if value>limit:raise AssertionError('actual cache/controller hard budget exceeded')
    run_id=performance_run_id(run_dir)
    store=MooncakeFeatureStore(store_id=run_id+'-inspection',setup_kwargs=_setup_from_env(server.command.env))
    remaining=[];key_count=0
    try:
        for row in _load_events(run_dir,'captures'):
            ref=row['ref'];store_id=ref['feature_store_uri'].split('://',1)[1].split('/',1)[0]
            if ref['run_id']!=run_id or store_id!=run_id or not ref['sample_id'].startswith(run_id+':'):
                raise AssertionError('actual sample escaped the isolated run/store namespace')
            for name in ref['feature_specs']:
                key=f"{store_id}/{ref['sample_id']}/g1/{name}";key_count+=1
                if store._store_exists(key):remaining.append(key)
    finally:
        store.close()
    final.update(inspected_keys=key_count,remaining_keys=remaining)
    write_json(Path(run_dir)/'cleanup-final.json',final)
    if remaining:raise AssertionError('actual training left published sample keys')


def summarize(run_dir, workload, group):
    consumed=_load_events(run_dir,'consumed');steps=_load_events(run_dir,'training-steps')
    consumed_order=audit_consumed_order(consumed,_load_events(run_dir,'fetch-validation'),workload)
    warmup=workload['training_warmup_steps'];end=warmup+workload['measured_steps']
    used={sid for row in consumed if row['optimizer_stepped'] and warmup<row['step']<=end for sid in row['sample_ids']}
    captures=[row for row in _load_events(run_dir,'captures') if row['sample_id'] in used]
    if len(used)!=workload['measured_steps'] or len(captures)!=len(used):raise AssertionError('measured optimizer samples lack complete unique capture diagnostics')
    fits=_load_events(run_dir,'fit')
    if len(fits)!=1:raise AssertionError('exactly one actual training rank fit record required')
    manifest=json.loads((Path(run_dir)/'manifest.json').read_text())
    polling=audit_polling_receipts(_load_events(run_dir,'polling-policy'),manifest['polling_policy'],manifest['polling_mode'],
                                 source_root=manifest['specforge']['root'])
    result={'polling':polling,'consumed_order':consumed_order,'training':aggregate_training(steps,warmup_steps=warmup,measured_steps=workload['measured_steps']),
            'capture':aggregate_capture(captures),'actual_fit':fits[0],
            'initial_draft':_load_events(run_dir,'draft-initial'),
            'validation_wall_s':sum(row['validation_wall_s'] for row in _load_events(run_dir,'fetch-validation') if row['sample_id'] in used),
            'owned_resource_peaks_boundary':'owned process-tree PSS/per-process GPU sampling through sampler drain; assigned-device totals include unattributed/platform memory; sampled bounds, not exact peaks',
            'server_stage_attribution':{}}
    for row in _load_events(run_dir,'server-stages'):
        ids=row.get('sample_ids',[])
        if len(ids)==1 and ids[0] in used:
            stage=result['server_stage_attribution'].setdefault(row['stage'],{'milliseconds':0.,'events':0})
            stage['milliseconds']+=row['milliseconds'];stage['events']+=1
    for stage in ('copy_result_combined_cuda_ms','capture_slice_host_wall_ms','capture_coalesce_host_wall_ms'):
        result['server_stage_attribution'].setdefault(stage,{'milliseconds':None,'events':0})
    if group=='device-off':
        actual=[row for row in _load_events(run_dir,'server-stages') if row['stage']=='publication_register_put_unregister_wall_ms' and any(sid in used for sid in row['sample_ids'])]
        if not actual or not all(row['cuda_sources'] for row in actual):
            raise AssertionError('device group lacks successful actual model CUDA publication source proof')
    result['training']['optimizer_step_wall_s'] = fits[0]['measured_fit_boundary_wall_s']/workload['measured_steps']
    result['training']['primary_boundary'] = 'actual optimizer callback boundaries after warmup; validation and production logging included'
    if workload['kind']=='cold':
        if any(row['L']!=0 for row in captures):raise AssertionError('cold-prefix stream reused a supposedly unique prefix')
        first=next((row for row in steps if row['step']==1),None)
        result['cold_profile']={'measured_requests_all_prefix_miss':True,
            'first_optimizer_boundary_wall_s': first['fit_elapsed_s'] if first else None,
            'boundary':'primary = distinct cold-prefix stream after trainer warmup; whole-command and first optimizer boundary separately include startup/JIT'}
    return result


def finalize_resource_observations(run_dir, result, workload):
    """Read the immutable sampler log only after its writer has drained."""
    if result.get('resource_sampler',{}).get('thread_drained') is not True:
        result['resource_observation_error']='sampler drain unproven; final resource aggregation unavailable'
        return
    path=Path(run_dir)/'resources.jsonl'
    result['observed_resources']=_load_resource_samples(path)
    # Failed startup/worker runs retain whole-lifetime observations without
    # inventing a completed optimizer measurement window.
    if 'actual_fit' not in result:
        return
    steps=_load_events(run_dir,'training-steps')
    warmup=workload['training_warmup_steps'];end=warmup+workload['measured_steps']
    first=next(row['monotonic'] for row in steps if row['step']==warmup)
    last=next(row['monotonic'] for row in steps if row['step']==end)
    result['measured_window_observed_resources']=_load_resource_samples(path,window=(first,last))


def _load_resource_samples(path,window=None):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if window is not None:
        rows=[row for row in rows if window[0]<=row.get('monotonic',-1)<=window[1]]
    peaks,coverage={},{}
    for field,key in (('pss_bytes','host_pss_peak_bytes'),('rss_bytes','host_rss_sum_peak_bytes'),
                      ('locked_bytes','host_locked_sum_peak_bytes')):
        complete,partial,unavailable=[],[],0
        for row in rows:
            host=row['host'];expected=row.get('owned_process_count',len(host))
            values=[p[field] for p in host if isinstance(p.get(field),(int,float)) and p[field]>=0]
            if expected and len(host)==expected and len(values)==expected and row.get('owned_tree_complete',True):
                complete.append(sum(values))
            elif values:
                partial.append(sum(values))
            else:
                unavailable+=1
        peaks[key]=max(complete,default=None)
        coverage[field]={'complete_samples':len(complete),'partial_samples':len(partial),
            'unavailable_samples':unavailable,'complete_fraction':len(complete)/len(rows) if rows else 0.,
            'partial_known_sum_peak_bytes':max(partial,default=None)}
    gpu_values=[sum(p['physical_used_bytes'] for p in row['gpu']) for row in rows
        if row['gpu'] and row.get('gpu_query_complete',False) and row.get('owned_tree_complete',True)]
    gpu_partial=[sum(p['physical_used_bytes'] for p in row['gpu']) for row in rows
        if row['gpu'] and not (row.get('gpu_query_complete',False) and row.get('owned_tree_complete',True))]
    gpu_complete=sum(row.get('gpu_query_complete',False) and row.get('owned_tree_complete',True) for row in rows)
    gpu_unavailable={}
    for row in rows:
        if not row['gpu']:
            reason=row.get('gpu_query',{}).get('availability','no_raw_query_diagnostic')
            gpu_unavailable[reason]=gpu_unavailable.get(reason,0)+1
    device_totals=[];device_peaks={};device_complete=0;assignment=None
    for row in rows:
        devices=row.get('assigned_gpu_devices',[])
        expected=row.get('assigned_gpu_indices',[])
        if expected:
            if assignment is None:assignment=set(expected)
            elif set(expected)!=assignment:raise AssertionError('assigned GPU set changed during resource sampling')
        for device in devices:
            value=device.get('physical_total_used_bytes')
            if device['index'] not in expected or not isinstance(value,int) or isinstance(value,bool) or value<0:
                continue
            previous=device_peaks.setdefault(device['index'],{'gpu_uuid':device['gpu_uuid'],'peak_bytes':None,'complete_samples':0})
            if previous['gpu_uuid']!=device['gpu_uuid']:
                raise AssertionError('assigned GPU UUID changed during resource sampling')
            previous['peak_bytes']=max(previous['peak_bytes'] or 0,value)
            previous['complete_samples']+=1
        valid=(bool(expected) and row.get('gpu_device_query_complete',False)
               and {device['index'] for device in devices}==set(expected)
               and all(isinstance(device.get('physical_total_used_bytes'),int)
                       and not isinstance(device['physical_total_used_bytes'],bool)
                       and device['physical_total_used_bytes']>=0 for device in devices))
        if not valid:
            continue
        device_complete+=1
        device_totals.append(sum(device['physical_total_used_bytes'] for device in devices))
    errors=[error for row in rows for error in row.get('errors',[])]
    intervals=[b['monotonic']-a['monotonic'] for a,b in zip(rows,rows[1:])]
    return {**peaks,'host_metric_coverage':coverage,
        'gpu_physical_used_sum_peak_bytes':max(gpu_values,default=None),
        'sample_count':len(rows),'gpu_samples_with_values':len(gpu_values)+len(gpu_partial),'gpu_complete_samples_with_values':len(gpu_values),
        'gpu_partial_known_sum_peak_bytes':max(gpu_partial,default=None),'gpu_query_complete_samples':gpu_complete,
        'gpu_unavailable_reason_counts':gpu_unavailable,
        'assigned_device_physical_total_sum_peak_bytes':max(device_totals,default=None),
        'assigned_device_physical_total_peak_bytes':device_peaks,
        'assigned_device_total_complete_samples':device_complete,
        'assigned_device_total_unavailable_samples':len(rows)-device_complete,
        'assigned_device_total_complete_fraction':device_complete/len(rows) if rows else 0.,
        'assigned_device_total_observation_complete':len(rows)>=2 and device_complete==len(rows),
        'assigned_device_total_boundary':'physical memory.used over assigned target/draft GPUs, including unattributed/platform memory; not owned process memory',
        'assigned_device_attributable_owned_peak_bytes':None,
        'assigned_device_ownership_proven':False,
        'host_pss_complete_samples':coverage['pss_bytes']['complete_samples'],
        'host_memory_comparison_eligible':len(rows)>=2 and all(item['complete_samples']==len(rows) for item in coverage.values()) and not errors,
        'gpu_memory_comparison_eligible':len(rows)>=2 and len(gpu_values)==len(rows) and gpu_complete==len(rows) and not errors,
        'errors':errors,'fatal_sampler_errors':sum(bool(row.get('fatal_sampler_error')) for row in rows),
        'sampling_interval_s':rows[0]['sample_interval_s'] if rows else None,
        'observed_interval_s':{'min':min(intervals,default=None),'max':max(intervals,default=None)},
        'window_monotonic':window,
        'boundary':'complete simultaneous host observations only; partial known sums and missing coverage separate; sampled bounds, not exact peaks'}


def instrumented_child(kind, original):
    if original and original[0]=='--':original=original[1:]
    if kind=='server':
        os.environ['PLANA_PERF_ROLE']='capture'
        install_server_observers()
        from sglang.launch_server import load_plugins,prepare_server_args,run_server,kill_process_tree
        load_plugins();args=prepare_server_args(original)
        try:run_server(args)
        finally:
            _poll_events(final=True)
            kill_process_tree(os.getpid(),include_parent=False)
        return 0
    install_worker_observers()
    from specforge.cli import main
    return main(original)


def main(argv=None):
    raw=list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in ('--instrumented-worker','--instrumented-server'):
        return instrumented_child('server' if raw[0].endswith('server') else 'worker',raw[1:])
    if raw and raw[0]=='--compare-run-dirs':
        print(json.dumps(compare_runs(raw[1:]),indent=2,sort_keys=True));return 0
    if raw and raw[0]=='--rdma-probe':
        try:
            rdma_cuda_probe(dict(os.environ),raw[1]);return 0
        except DevicePublicationUnavailable as exc:
            write_json(raw[1],{'status':'unavailable','error':str(exc)});return 3
        except Exception as exc:
            write_json(raw[1],{'status':'failed','error':f'{type(exc).__name__}: {exc}'});traceback.print_exc();return 2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--native-cases',required=True);parser.add_argument('--real-cases',required=True)
    parser.add_argument('--run-dir',required=True);parser.add_argument('--group',choices=GROUPS,required=True);parser.add_argument('--workload',choices=WORKLOADS,required=True)
    parser.add_argument('--transport',choices=('tcp','rdma'),required=True);parser.add_argument('--rdma-devices');parser.add_argument('--rdma-proof')
    parser.add_argument('--specforge-source',required=True);parser.add_argument('--sglang-source',required=True)
    parser.add_argument('--expected-specforge',required=True);parser.add_argument('--expected-sglang',required=True)
    parser.add_argument('--warmup-steps',type=int,default=8);parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--cache-bytes',type=int,default=64<<20);parser.add_argument('--chunk-tokens',type=int,default=256)
    parser.add_argument('--rdma-probe-timeout-s',type=float,default=120.,help='bounded native CUDA/RDMA probe wait')
    for field, default in POLL_DEFAULTS.items():
        parser.add_argument('--'+field.replace('_','-'),type=float,default=default)
    parser.add_argument('--execute',action='store_true',help='requires root-assigned GPUs/ports/HCAs; default only plans')
    args=parser.parse_args(raw)
    run_dir=Path(args.run_dir).resolve();run_dir.mkdir(parents=True,exist_ok=False)
    result={'status':'planning','passed':False,'group':args.group,'workload':args.workload,'transport':args.transport}
    try:
        polling_policy=validate_polling_policy({field:getattr(args,field) for field in POLL_DEFAULTS})
        polling_mode='official-api' if args.group in ('device-off','host-off') else 'candidate-config'
        if not math.isfinite(args.rdma_probe_timeout_s) or args.rdma_probe_timeout_s <= 0:
            raise ValueError('positive finite RDMA probe timeout required')
        sf=git_identity(args.specforge_source,args.expected_specforge);sg=git_identity(args.sglang_source,args.expected_sglang)
        if args.group in ('host-off','device-off') and (sf['commit']!=OFFICIAL_SF or sg['commit']!=OFFICIAL_SG):
            raise ValueError('official comparison groups require the exact immutable official baselines')
        for root,ancestor in ((args.specforge_source,OFFICIAL_SF),(args.sglang_source,OFFICIAL_SG)):
            if subprocess.run(['git','-C',root,'merge-base','--is-ancestor',ancestor,'HEAD']).returncode:
                raise ValueError('candidate is not descended from the fixed official baseline')
        patch=Path(args.specforge_source)/'patches/sglang/v0.5.18/spec-capture.patch'
        if hashlib.sha256(patch.read_bytes()).hexdigest()!=OFFICIAL_PATCH_SHA256:raise ValueError('official patch bytes differ')
        sys.path[:0]=[str(Path(args.specforge_source).resolve()),str(Path(args.sglang_source).resolve()/'python')]
        import yaml
        from specforge.config import load_config
        from specforge.application import resolve_run
        from specforge.launch_plan import build_launch_plan,run_commands,_wait_for_service
        from specforge.launch import _epoch_prompt_indices
        from specforge.training.disaggregated import _online_prompt_seed
        original=yaml.safe_load(Path(args.config).read_text());original_cfg=load_config(args.config);resolved=resolve_run(original_cfg)
        schema=_model_info(original_cfg,resolved)
        native,real=load_rows(args.native_cases),load_rows(args.real_cases)
        workload=build_workload(args.workload,native,real,warmup_steps=args.warmup_steps,measured_steps=args.steps,
            chunk_tokens=args.chunk_tokens,row_bytes=schema['row_bytes'],cache_bytes=args.cache_bytes)
        if max(len(row['input_ids']) for row in workload['training'])>original_cfg.data.max_length:
            raise ValueError('workload would be silently truncated by native reader; refusing entire run')
        cfg_dict=isolate_config(original,run_dir=run_dir,workload=workload,group=args.group,transport=args.transport,
            cache_bytes=args.cache_bytes,chunk_tokens=args.chunk_tokens,rdma_devices=args.rdma_devices)
        if polling_mode=='candidate-config':
            cfg_dict['runtime'].update(polling_policy)
        config_path=run_dir/'config.yaml';config_path.write_text(yaml.safe_dump(cfg_dict,sort_keys=False))
        cfg=load_config(str(config_path));resolved=resolve_run(cfg)
        if cfg.runtime.producer_concurrency != 1:
            raise ValueError('resolved runtime producer_concurrency must be exactly one')
        serialize_for_epoch_order(workload,_epoch_prompt_indices,prompt_seed=_online_prompt_seed(cfg))
        workload['fingerprints']=sorted({content_fingerprint(row['input_ids'],row['loss_mask']) for row in workload['training']+workload['prewarm']})
        write_json(run_dir/'workload.json',workload)
        (run_dir/'training.jsonl').write_text(''.join(json.dumps({'input_ids':row['input_ids'],'loss_mask':row['loss_mask']})+'\n' for row in workload['training']))
        performance_env={'PYTHONPATH':os.pathsep.join([str(Path(args.specforge_source).resolve()),str(Path(args.sglang_source).resolve()/'python')]),
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PLANA_PERF_RUN_DIR':str(run_dir),
            'PLANA_PERF_SPECFORGE_SOURCE':str(Path(args.specforge_source).resolve()),
            'PLANA_PERF_SGLANG_SOURCE':str(Path(args.sglang_source).resolve()),
            'PLANA_PERF_WORKLOAD_JSON':str(run_dir/'workload.json'),'PLANA_PERF_SCHEMA_JSON':json.dumps(schema),
            'PLANA_PERF_POLL_POLICY':json.dumps(polling_policy),'PLANA_PERF_POLL_MODE':polling_mode}
        if args.transport=='rdma':performance_env['MC_STORE_MEMCPY']='0'
        base_env={**os.environ,**performance_env}
        plan=build_launch_plan(resolved.config,algorithm=resolved.algorithm,config_path=str(config_path),env=base_env)
        services=[]
        for service in plan.services:
            command=service.command
            env={**performance_env,**command.env}
            if command.label.startswith('capture-server-'):
                env.update(PLANA_PERF_ROLE='capture',SGLANG_SPEC_CAPTURE_TIMING='1')
                command=dataclasses.replace(command,argv=rewrite_capture_argv(command.argv,SCRIPT),env=env)
            else:command=dataclasses.replace(command,env=env)
            services.append(dataclasses.replace(service,command=command))
        commands=tuple(dataclasses.replace(command,argv=rewrite_worker_argv(command.argv,SCRIPT),
            env={**performance_env,**command.env,'PLANA_PERF_ROLE':command.label}) for command in plan.commands)
        plan=dataclasses.replace(plan,commands=commands,services=tuple(services))
        write_json(run_dir/'manifest.json',{'comparison_contract': comparison_contract(cfg_dict,workload['data_sha256'],schema,args.cache_bytes,args.chunk_tokens,polling_policy=polling_policy), 'specforge':sf,'sglang':sg,'schema':schema,'data_sha256':workload['data_sha256'],
            'polling_policy':polling_policy,'polling_mode':polling_mode,
            'polling_boundary':'same three effective intervals; official existing Python APIs, candidate real runtime config; actual application receipts required',
            'intended_data_sha256':workload['intended_data_sha256'],'prompt_seed':workload['prompt_seed'],
            'producer_concurrency':cfg.runtime.producer_concurrency,
            'run_identity':{'run_dir':str(run_dir),'run_id':cfg.run_id,
                'store_id':cfg_dict['deployment']['disaggregated']['store_id'],
                'prewarm_id':performance_run_id(run_dir)+'-prewarm',
                'rdma_proof_id':performance_run_id(run_dir)+'-rdma-proof',
                'inspection_id':performance_run_id(run_dir)+'-inspection'},
            'template_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),'argv':raw,
            'python':sys.executable,'harness_sha256':hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            'helpers_sha256':hashlib.sha256(SCRIPT.with_name('performance_helpers.py').read_bytes()).hexdigest(),
            'harness_git_commit':subprocess.check_output(['git','-C',str(SCRIPT.parents[2]),'rev-parse','HEAD'],text=True).strip(),
            'physical_sampler':'owned process trees and separate unattributed assigned-device totals; 100ms requested interval, observed intervals/coverage reported',
            'measurement_boundary':'actual CLI fit+loader+optimizer, full per-get validation overhead included; isolated capture never substituted'})
        (run_dir/'launch-plan.json').write_text(plan.render())
        if not args.execute:
            result.update(status='planned',logical_footprint=workload['logical_footprint'])
            return 0
        if args.transport=='rdma' and args.group!='device-off':
            if not args.rdma_proof:raise ValueError('host RDMA comparison requires same-boot actual device proof file')
            proof=json.loads(Path(args.rdma_proof).read_text());validate_rdma_proof(proof,expected_devices=args.rdma_devices)
            if proof.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise ValueError('stale-boot device proof')
        from types import SimpleNamespace
        sampler=ProcessSampler(run_dir/'resources.jsonl',assigned_gpus=assigned_gpu_indices(cfg_dict))
        sampler.add('orchestrator',SimpleNamespace(pid=os.getpid(),poll=lambda:None))
        processes=[];warmed=False
        parent_env={key:base_env[key] for key in ('PLANA_PERF_RUN_DIR','PLANA_PERF_WORKLOAD_JSON','PLANA_PERF_SCHEMA_JSON')}
        if args.transport=='rdma':parent_env['MC_STORE_MEMCPY']='0'
        labels={tuple(command.argv):command.label for command in (*plan.commands,*(service.command for service in plan.services))}
        def popen(*a,**kw):
            if kw.get('start_new_session',True) is not True:
                raise ValueError('all owned performance children require detached sessions')
            kw['start_new_session']=True
            process=subprocess.Popen(*a,**kw);processes.append(process)
            label=labels.get(tuple(a[0]),'rdma-proof' if '--rdma-probe' in a[0] else 'owned-extra')
            sampler.add(label,process)
            append_json(run_dir/'owned-processes.jsonl',{'pid':process.pid,'label':label,'argv_sha256':canonical_digest(list(a[0])),'spawned':time.time()})
            return process
        def ready(service,process,started):
            nonlocal warmed
            _wait_for_service(service,process,started)
            if service.command.label.startswith('capture-server-') and not warmed:
                if args.group=='device-off':
                    proof_env={**os.environ,**service.command.env,'CUDA_VISIBLE_DEVICES':service.command.env['CUDA_VISIBLE_DEVICES']}
                    import specforge.launch_plan as launch_module
                    probe_status=run_owned_probe(popen,[sys.executable,str(SCRIPT),'--rdma-probe',str(run_dir/'device-proof.json')],
                        env=proof_env,terminate=launch_module._terminate_processes,
                        timeout_s=args.rdma_probe_timeout_s,grace_s=plan.shutdown_grace_s)
                    if probe_status==3:raise DevicePublicationUnavailable('actual native CUDA registration/RDMA transfer unavailable; see device-proof.json')
                    if probe_status!=0:raise RuntimeError('device proof harness failed; see device-proof.json')
                    result['device_publication_available']=True
                with scoped_environment(parent_env):
                    warm_server(run_dir,cfg,resolved,service,workload)
                warmed=True
        # Defer service teardown until worker validation and final control/key audit.
        import specforge.launch_plan as launch_module
        real_stop=launch_module._stop_services
        started_services=[]
        def remember_stop(started,**kwargs):started_services.extend(item for item in started if item not in started_services)
        launch_module._stop_services=remember_stop
        started=time.perf_counter()
        try:
            sampler.start()
            status=run_commands(plan,popen=popen,readiness_waiter=ready)
            result['command_exit']=status;result['whole_command_wall_s']=time.perf_counter()-started
            if status:raise RuntimeError(f'actual CLI workers failed exit={status}')
            with scoped_environment(parent_env):
                final_server_and_keys(plan,run_dir)
            result.update(summarize(run_dir,workload,args.group),passed=True,status='passed')
        finally:
            launch_module._stop_services=real_stop
            result['resource_sampler']=finish_owned_run(launch_module,real_stop,started_services,processes,sampler,run_dir,
                grace_s=plan.shutdown_grace_s,result=result,workload=workload)
        return 0
    except BaseException as exc:
        result.update(passed=False,status='unavailable' if isinstance(exc,DevicePublicationUnavailable) else 'failed',error=f'{type(exc).__name__}: {exc}')
        (run_dir/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
        return 2
    finally:
        if 'sf' in locals() and 'sg' in locals():
            try:
                source_after={'specforge':git_identity(args.specforge_source,args.expected_specforge),
                              'sglang':git_identity(args.sglang_source,args.expected_sglang)}
                before=json.loads((run_dir/'manifest.json').read_text()) if (run_dir/'manifest.json').exists() else None
                if before is not None and (hashlib.sha256(SCRIPT.read_bytes()).hexdigest()!=before['harness_sha256']
                    or hashlib.sha256(SCRIPT.with_name('performance_helpers.py').read_bytes()).hexdigest()!=before['helpers_sha256']):
                    raise AssertionError('performance harness source changed during run')
                write_json(run_dir/'source-after.json',source_after)
                result['source_unchanged']=True
            except Exception as changed:
                result.update(source_unchanged=False,passed=False,status='failed',source_error=str(changed))
        result['ended_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(run_dir/'result.json',result)
        if result.get('source_unchanged') is False:return 2


# The parent script stays __main__ when importing/calling sglang.run_server,
# so multiprocessing spawn executes this file before unpickling scheduler code.
if __name__=='__mp_main__' and os.environ.get('PLANA_PERF_ROLE')=='capture':
    install_server_observers()

if __name__=='__main__':
    raise SystemExit(main())
