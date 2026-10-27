"""Import-light, CPU-testable planning and accounting for formal Plan A runs."""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import threading
import time

GROUPS = ('device-off', 'host-off', 'host-on', 'candidate-host-off')
WORKLOADS = ('cold', 'hot', 'over-capacity', 'real64')
STAGES = ('forward_ms', 'd2h_wait_ms', 'assembly_ms', 'cache_fill_ms', 'publication_queue_ms')
OFFICIAL_PATCH_SHA256 = '8182e4eac902b4009f4d0314c3b6a9e71543faeac5412e3032b8741ec35537a8'


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def append_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, default=str) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def load_rows(path):
    rows = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        tokens, mask = row['input_ids'], row['loss_mask']
        if not tokens or len(tokens) != len(mask):
            raise ValueError(f'invalid tokens/mask in row {index}')
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens):
            raise ValueError(f'invalid token ID in row {index}')
        if not any(a > .5 and b > .5 for a, b in zip(mask, mask[1:])):
            raise ValueError(f'row {index} cannot supervise training anchors')
        rows.append(dict(row, id=str(row.get('id', f'row-{index:04d}'))))
    if not rows:
        raise ValueError('empty dataset')
    return rows


def content_fingerprint(tokens, mask):
    if any(value not in (0, 1) for value in mask):
        raise ValueError('binary loss_mask required by int64 capture schema')
    return canonical_digest([list(tokens), [int(value) for value in mask]])


def complete_prefix_footprint(rows, chunk_tokens, row_bytes):
    """Logical unique complete causal chunks, independent of sample names/masks."""
    if chunk_tokens < 1 or row_bytes < 1:
        raise ValueError('positive chunk and feature row size required')
    blocks = set()
    for row in rows:
        tokens = row['input_ids']
        parent = hashlib.sha256(b'plana-performance-causal-prefix-v1').digest()
        for start in range(0, len(tokens) - chunk_tokens + 1, chunk_tokens):
            end = start + chunk_tokens
            payload = b''.join(struct.pack('<q', int(token)) for token in tokens[start:end])
            parent = hashlib.sha256(parent + struct.pack('<QQ', start, end) + payload).digest()
            blocks.add((parent, start, end))
    return {'unique_complete_chunks': len(blocks), 'feature_bytes': len(blocks) * chunk_tokens * row_bytes,
            'chunk_tokens': chunk_tokens, 'row_bytes': row_bytes}


def build_workload(kind, native_rows, real_rows, *, warmup_steps, measured_steps, chunk_tokens, row_bytes, cache_bytes):
    if kind not in WORKLOADS or warmup_steps < 2 or measured_steps < 8:
        raise ValueError('workload requires >=2 training warmup and >=8 measured optimizer steps')
    base = next((row for row in native_rows if len(row['input_ids']) >= 4 * chunk_tokens), None)
    if base is None:
        raise ValueError('native data lacks a sequence with four complete chunks')
    token_pool = sorted({token for row in native_rows for token in row['input_ids']})
    total = warmup_steps + measured_steps
    if kind == 'real64':
        if len(real_rows) != 64 or measured_steps < 64:
            raise ValueError('real64 requires exactly64 original rows and >=64 measured optimizer steps')
        catalog = copy.deepcopy(real_rows)
    elif kind == 'hot':
        catalog = [copy.deepcopy(base)]
    else:
        sample_bytes = complete_prefix_footprint([base], chunk_tokens, row_bytes)['feature_bytes']
        count = total if kind == 'cold' else max(4, math.ceil(cache_bytes / sample_bytes) + 2)
        if count > len(token_pool):
            raise ValueError('not enough known legal token IDs for distinct first causal chunks')
        catalog = []
        for index in range(count):
            row = copy.deepcopy(base)
            row['id'] = f'{kind}-{index:04d}'
            row['input_ids'][0] = token_pool[index]
            catalog.append(row)
    footprint = complete_prefix_footprint(catalog, chunk_tokens, row_bytes)
    if kind == 'over-capacity' and footprint['feature_bytes'] <= cache_bytes:
        raise AssertionError('over-capacity label lacks a proven >capacity working set')
    if kind == 'hot' and footprint['feature_bytes'] > cache_bytes:
        raise ValueError('hot working set does not fit the fixed cache budget')
    training = [dict(copy.deepcopy(catalog[index % len(catalog)]), id=f'train-{index:05d}') for index in range(total)]
    # Cold measured requests have distinct first chunks, including from warmup.
    warming = [] if kind == 'cold' else [dict(copy.deepcopy(row), id=f'prewarm-{i:05d}') for i, row in enumerate(catalog)]
    return {'kind': kind, 'catalog': catalog, 'training': training, 'prewarm': warming,
            'training_warmup_steps': warmup_steps, 'measured_steps': measured_steps,
            'logical_footprint': footprint, 'data_sha256': canonical_digest(training)}


def serialize_for_epoch_order(workload, epoch_indices, *, prompt_seed):
    """Invert the actual producer shuffle so its chronological stream is fixed."""
    if 'intended_training' in workload:
        raise ValueError('workload epoch permutation was already serialized')
    desired = copy.deepcopy(workload['training'])
    order = list(epoch_indices(desired, 0, seed=prompt_seed))
    if sorted(order) != list(range(len(desired))):
        raise ValueError('actual producer order is not a complete permutation')
    serialized = [None] * len(desired)
    for index, row in zip(order, desired, strict=True):
        serialized[index] = copy.deepcopy(row)
    workload.update(training=serialized, intended_training=desired,
                    prompt_seed=prompt_seed, producer_epoch_indices=order,
                    data_sha256=canonical_digest(serialized),
                    intended_data_sha256=canonical_digest(desired))


def audit_consumed_order(consumed, fetches, workload):
    """Verify real optimizer inputs, including warmup and duplicate originals."""
    fingerprints = {}
    for row in fetches:
        sid, fingerprint = row['sample_id'], row['content_fingerprint']
        if sid in fingerprints and fingerprints[sid] != fingerprint:
            raise ValueError('a sample changed content across actual fetches')
        fingerprints[sid] = fingerprint
    selected = sorted((row for row in consumed if row['optimizer_stepped']), key=lambda row: row['step'])
    expected = [content_fingerprint(row['input_ids'], row['loss_mask']) for row in workload['intended_training']]
    if [row['step'] for row in selected] != list(range(1, len(expected)+1)):
        raise ValueError('actual optimizer sequence is missing or duplicated')
    actual, observed_steps = [], []
    for row in selected:
        if len(row['sample_ids']) != 1 or row['sample_ids'][0] not in fingerprints:
            raise ValueError('actual optimizer input lacks exactly one validated full sample')
        fingerprint = fingerprints[row['sample_ids'][0]]
        actual.append(fingerprint)
        observed_steps.append({'step': row['step'], 'sample_id': row['sample_ids'][0],
                               'content_fingerprint': fingerprint})
    warmup = workload['training_warmup_steps']
    if Counter(actual[warmup:]) != Counter(expected[warmup:]):
        raise ValueError('measured actual sample multiset differs from the intended workload')
    if actual != expected:
        raise ValueError('actual optimizer sample order differs from the intended workload')
    return {'steps': observed_steps, 'ordered_fingerprints': actual, 'ordered_sha256': canonical_digest(actual),
            'measured_fingerprints': actual[warmup:],
            'measured_counter': dict(Counter(actual[warmup:])),
            'measured_counter_matches': True, 'full_order_matches': True}


def validate_rdma_proof(proof, *, expected_devices=None):
    required = ('protocol', 'gpu_put', 'cuda_source', 'cuda_register_status', 'put_status',
                'fetch_equal', 'keys_removed', 'rdma_devices', 'forbidden_overrides', 'local_memcpy_setting')
    if any(name not in proof for name in required):
        raise ValueError('incomplete actual CUDA publication proof')
    if proof['protocol'] != 'rdma' or proof['gpu_put'] is not True or proof['cuda_source'] is not True:
        raise ValueError('device group requires actual RDMA CUDA publication')
    if proof['cuda_register_status'] != 0 or proof['put_status'] != 0 or not proof['fetch_equal'] or not proof['keys_removed']:
        raise ValueError('actual CUDA register/put/fetch/remove roundtrip did not pass')
    if not proof['rdma_devices'] or proof['forbidden_overrides'] or proof['local_memcpy_setting']!='0':
        raise ValueError('RDMA devices absent or transport overrides detected')
    if expected_devices is not None and proof['rdma_devices'] != expected_devices:
        raise ValueError('proof HCA list differs from actual run')
    return True


def isolate_config(original, *, run_dir, workload, group, transport, cache_bytes, chunk_tokens, rdma_devices):
    if group not in GROUPS or transport not in ('tcp', 'rdma'):
        raise ValueError('unsupported comparison group/transport')
    if group == 'device-off' and transport != 'rdma':
        raise ValueError('TCP can never be labelled device publication')
    cfg = copy.deepcopy(original)
    root = Path(run_dir)
    cfg['run_id'], cfg['output_dir'] = root.name, str(root / 'output')
    cfg['data']['prompts_path'], cfg['data']['cache_dir'] = str(root / 'training.jsonl'), str(root / 'data-cache')
    cfg['training'].update(seed=42, batch_size=1, accumulation_steps=1, num_anchors=8,
                           dflash_teacher_metrics=True, num_epochs=1, log_interval=1,
                           max_steps=len(workload['training']), total_steps=len(workload['training']))
    if cfg['training'].get('eval_interval', 0):
        raise ValueError('performance template must disable evaluation')
    cfg['tracking']['report_to'] = 'none'
    disagg = cfg['deployment']['disaggregated']
    cfg.setdefault('runtime', {})['producer_concurrency'] = 1
    disagg['control_dir'], disagg['consumer_state_dir'] = str(root / 'control'), str(root / 'consumer-state')
    managed = disagg['managed_local']
    if len(managed['capture_servers']) != 1:
        raise ValueError('exactly one capture server required')
    mooncake = managed['mooncake']
    mooncake['protocol'] = transport
    if transport == 'rdma':
        if not rdma_devices:
            raise ValueError('explicit root-assigned HCA list required for RDMA')
        mooncake['rdma_devices'] = rdma_devices
    else:
        mooncake.pop('rdma_devices', None)
    server = managed['capture_servers'][0]
    server['gpu_put'] = group == 'device-off'
    for owner, key in ((cfg['model'], 'sglang_extra_args'), (server, 'extra_args')):
        if any(str(item).startswith('--spec-feature-') for item in owner.get(key, ())):
            raise ValueError('template must not carry stale cache mode/budget flags')
    if group == 'host-on':
        server.setdefault('extra_args', []).extend(['--spec-feature-cache', '--spec-feature-chunk-tokens', str(chunk_tokens),
            '--spec-feature-cache-bytes', str(cache_bytes), '--spec-feature-pinned-bytes', str(256 << 20),
            '--spec-feature-output-bytes', str(512 << 20), '--spec-feature-device-bytes', str(256 << 20)])
    return cfg


def aggregate_capture(records):
    total = reused = 0
    stages = {name: {'total_ms': 0.0, 'requests': 0} for name in STAGES}
    attempts = set()
    for row in records:
        N, L = row['N'], row['L']
        if not isinstance(N, int) or not isinstance(L, int) or not 0 <= L < N or row.get('span') != [L, N]:
            raise ValueError('invalid absolute captured span')
        identity = (row.get('sample_id'), row.get('attempt_id'))
        if identity in attempts:
            raise ValueError('duplicate capture result in aggregate')
        attempts.add(identity)
        total += N
        reused += L
        for name, value in row.get('timings', {}).items():
            if name in stages and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                stages[name]['total_ms'] += value
                stages[name]['requests'] += 1
    count = len(records)
    for entry in stages.values():
        entry['coverage'] = entry['requests'] / count if count else 0.0
        if not entry['requests']:
            entry['total_ms'] = None  # Missing never means zero.
    return {'requests': count, 'tokens': total, 'reused_tokens': reused, 'computed_tokens': total - reused,
            'token_weighted_L_over_N': reused / total if total else None, 'stages': stages}


def aggregate_training(records, *, warmup_steps, measured_steps):
    selected = [row for row in records if warmup_steps < row['step'] <= warmup_steps + measured_steps]
    if len(selected) != measured_steps or len({row['step'] for row in selected}) != measured_steps:
        raise ValueError('actual optimizer-step metrics missing/duplicated')
    times = [float(row['metrics']['perf/optimizer_step_time_s']) for row in selected]
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError('nonpositive actual optimizer-step window')
    return {'measured_optimizer_steps': measured_steps, 'controller_window_optimizer_step_wall_s': sum(times) / measured_steps,
            'measured_controller_window_s': sum(times), 'warmup_optimizer_steps': warmup_steps,
            'boundary': 'real TrainerController loader+optimizer+durableACK window; logging/validation overhead included'}


def rewrite_worker_argv(argv, script):
    """Keep torchrun and every original argument, replacing only its CLI module."""
    argv = list(argv)
    for index in range(len(argv) - 1):
        if argv[index:index+2] == ['-m', 'specforge.cli']:
            return tuple(argv[:index] + [str(script), '--instrumented-worker', '--'] + argv[index+2:])
    raise ValueError('unsupported worker command: expected exact -m specforge.cli')


def rewrite_capture_argv(argv, script):
    argv = list(argv)
    for index in range(len(argv) - 1):
        if argv[index:index+2] == ['-m', 'sglang.launch_server']:
            return tuple(argv[:index] + [str(script), '--instrumented-server', '--'] + argv[index+2:])
    raise ValueError('unsupported capture command: expected exact -m sglang.launch_server')


def _resource_error(phase, exc, **context):
    return {'phase': phase, 'error': f'{type(exc).__name__}: {exc}', **context}


def read_process_sample(pid):
    base = Path('/proc') / str(pid)
    stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
    start_time = int(stat[19])
    fields, errors, sources = {}, [], {}
    for name in ('status', 'smaps_rollup'):
        try:
            for line in (base / name).read_text().splitlines():
                key, _, value = line.partition(':')
                parts = value.split()
                if parts and parts[0].isdigit():
                    fields[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == 'kB' else 1)
            sources[name] = True
        except (OSError, ProcessLookupError) as exc:
            sources[name] = False
            errors.append(_resource_error(name, exc, pid=pid))
    return {'pid': pid, 'start_time': start_time, 'rss_bytes': fields.get('VmRSS'),
            'pss_bytes': fields.get('Pss'), 'locked_bytes': fields.get('VmLck'), 'hwm_bytes': fields.get('VmHWM'),
            'sources': sources, 'read_errors': errors}


class ProcessSampler:
    """Sample owned trees, with explicit unavailable values and failure coverage."""
    def __init__(self, output, interval=.1):
        self.output, self.interval = Path(output), interval
        self.roots, self.identities = {}, {}
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.fatal_errors, self.sample_count = [], 0
        self.thread = threading.Thread(target=self._run, name='plana-owned-resource-sampler', daemon=True)

    def add(self, label, process):
        with self.lock:
            self.roots[process.pid] = (label, process)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError('owned resource sampler did not drain')
        if self.fatal_errors:
            raise RuntimeError('owned resource sampler failed: ' + json.dumps(self.fatal_errors))
        return {'samples': self.sample_count, 'fatal_errors': [], 'thread_drained': True}

    def _owned(self, errors):
        with self.lock:
            roots = sorted(self.roots.items(), key=lambda item:item[0]==os.getpid())
        found = {}
        for pid, (label, process) in roots:
            if process.poll() is not None:
                continue
            pending = [pid]
            while pending:
                current = pending.pop()
                if current in found:
                    continue
                found[current] = label
                try:
                    pending.extend(int(value) for value in Path(f'/proc/{current}/task/{current}/children').read_text().split())
                except (OSError, ProcessLookupError) as exc:
                    errors.append(_resource_error('children', exc, pid=current))
        return found

    def _sample_once(self):
        errors = []
        owned = self._owned(errors)
        tree_complete = not errors
        records = []
        for pid, label in owned.items():
            try:
                row = read_process_sample(pid)
                previous = self.identities.setdefault(pid, row['start_time'])
                if previous != row['start_time']:
                    errors.append({'phase': 'identity', 'pid': pid, 'error': 'PID start time changed'})
                    continue
                records.append(dict(row, label=label))
                errors.extend(row.get('read_errors', []))
            except (OSError, ProcessLookupError) as exc:
                errors.append(_resource_error('host', exc, pid=pid))
        gpu, gpu_query_complete = [], False
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=2)
            if result.returncode == 0:
                gpu_query_complete = True
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) in owned:
                        if parts[1].isdigit():
                            gpu.append({'pid': int(parts[0]), 'physical_used_bytes': int(parts[1]) << 20, 'gpu_uuid': parts[2]})
                        else:
                            gpu_query_complete = False
                            errors.append({'phase': 'gpu', 'pid': int(parts[0]), 'error': 'physical used GPU memory unavailable'})
            else:
                errors.append({'phase': 'gpu', 'error': f'nvidia-smi exit {result.returncode}'})
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(_resource_error('gpu', exc))
        append_json(self.output, {'time': time.time(), 'monotonic': time.perf_counter(), 'host': records, 'gpu': gpu,
            'owned_process_count': len(owned), 'owned_tree_complete': tree_complete,
            'gpu_query_complete': gpu_query_complete, 'errors': errors, 'sample_interval_s': self.interval})
        self.sample_count += 1

    def _run(self):
        try:
            while not self.stop_event.is_set():
                self._sample_once()
                self.stop_event.wait(self.interval)
        except Exception as exc:
            failure = _resource_error('sampler_thread', exc)
            self.fatal_errors.append(failure)
            try:
                append_json(self.output, {'time': time.time(), 'monotonic': time.perf_counter(),
                    'host': [], 'gpu': [], 'owned_tree_complete': False, 'gpu_query_complete': False,
                    'errors': [failure], 'fatal_sampler_error': True, 'sample_interval_s': self.interval})
            except Exception:
                pass  # stop() still exposes the original failure if the output itself is unavailable.
            self.stop_event.set()


def comparison_contract(config, workload_sha, schema, cache_bytes, chunk_tokens):
    value = copy.deepcopy(config)
    value.pop('run_id', None)
    value.pop('output_dir', None)
    value['data']['prompts_path'] = workload_sha
    value['data']['cache_dir'] = '<isolated>'
    disagg = value['deployment']['disaggregated']
    for key in ('control_dir', 'consumer_state_dir'):
        disagg[key] = '<isolated>'
    for owner, key in [(value['model'], 'sglang_extra_args')] + [(s, 'extra_args') for s in disagg['managed_local']['capture_servers']]:
        original = owner.get(key, [])
        clean = []
        index = 0
        while index < len(original):
            token = original[index]
            if token == '--spec-feature-cache':
                index += 1
            elif token.startswith('--spec-feature-'):
                index += 1 if '=' in token else 2
            else:
                clean.append(token)
                index += 1
        owner[key] = clean
        owner.pop('gpu_put', None)
    return {'configuration': value, 'schema': schema, 'cache_bytes': cache_bytes, 'chunk_tokens': chunk_tokens}


def compare_runs(run_dirs):
    rows = []
    contracts = set()
    initial = set()
    consumed = set()
    for directory in run_dirs:
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        contracts.add(canonical_digest(manifest['comparison_contract']))
        if result.get('status') == 'passed':
            states = result['initial_draft']
            if len(states) != 1:
                raise ValueError('one initial actual draft fingerprint required')
            initial.add(states[0]['sha256'])
            audit = result['consumed_order']
            if not audit['full_order_matches'] or not audit['measured_counter_matches']:
                raise ValueError('actual optimizer input audit did not pass')
            consumed.add(audit['ordered_sha256'])
        rows.append({'group': result['group'], 'status': result['status'], 'run_dir': str(directory),
                     'optimizer_step_wall_s': result.get('training', {}).get('optimizer_step_wall_s')})
    if len(contracts) != 1 or len(initial) > 1 or len(consumed) > 1:
        raise ValueError('workload/configuration/initial draft/actual sample order differs across comparison groups')
    passing = {row['group']: row for row in rows if row['status'] == 'passed'}
    if 'host-on' not in passing or 'host-off' not in passing:
        raise ValueError('passing official host-off and candidate host-on required')
    off, on = passing['host-off']['optimizer_step_wall_s'], passing['host-on']['optimizer_step_wall_s']
    return {'runs': rows, 'host_off_over_host_on_step_ratio': off/on,
            'measured_lower_step_time': on < off, 'acceleration_claim_evaluated': False,
            'claim_boundary': 'single matched run pair; independent repetitions and variance analysis still required',
            'device_group_available': 'device-off' in passing}


def optimizer_step_number(args, kwargs):
    ctx = kwargs.get('ctx') or (args[0] if args else None)
    if ctx is None or not hasattr(ctx, 'global_step'):
        raise ValueError('actual TrainerCore StepContext is required for sample/step accounting')
    return int(ctx.global_step) + 1
