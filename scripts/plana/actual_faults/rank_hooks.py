"""Scoped instrumentation installed at the official scheduler entry on every rank.

Never drives a collective. All communication remains the engine's ordinary
request broadcast / bridge admission sequence. No tensor data is fabricated.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import inspect
import os
from pathlib import Path
import threading
import time
import weakref
from actual_faults import append_json, atomic_json, read_json, source_hash

_LOCAL = threading.local()
_STATES = []


@contextmanager
def patched_attribute(owner, name, replacement):
    own = name in vars(owner)
    previous_own = vars(owner).get(name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if own:
            setattr(owner, name, previous_own)
        else:
            delattr(owner, name)


def scalar(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        return value.detach().cpu().reshape(-1).tolist()
    return value


def index_list(value):
    value = scalar(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for part in value for item in index_list(part)]
    return [int(value)]


def owned_free_membership(allocator, indices, *, mamba=False):
    """Read static allocator free lists; virtual unified IDs use count proof.

    A unified allocator's public free list contains physical rather than
    virtual IDs. Comparing those to request virtual IDs would be false proof.
    """
    if hasattr(allocator, '_multi_ended_allocator'):
        return None
    names = ('free_slots',) if mamba else ('free_pages', 'release_pages')
    lists = [getattr(allocator, name, None) for name in names]
    if not any(value is not None for value in lists):
        return None
    owned = set(indices)
    # Preserve multiplicity: duplicate free-list entries would turn a double
    # free into apparently larger available capacity if reduced to a set.
    return sorted(index for value in lists for index in index_list(value) if index in owned)


def assert_retract_allocation_return(receipt, before, after):
    """Validate exact request-owned frees and conservation, independently of Req reset."""
    owned = receipt['owned']
    kv, mamba = owned['kv_indices'], owned['mamba_slots']
    if (not kv or len(set(kv)) != len(kv) or any(index <= 0 for index in kv)
            or len(kv) != owned['kv_end'] - owned['kv_start']
            or owned['kv_start'] != owned['prefix_len']):
        raise AssertionError('retract lacks exact exclusively owned suffix KV indices')
    if (len(set(mamba)) != len(mamba) or any(index <= 0 for index in mamba)
            or set(mamba) & set(owned['cache_owned_cow_sources'])
            or set(mamba) != set(owned['fresh_cow_slots'] + owned['ping_pong_slots'])):
        raise AssertionError('retract Mamba ownership includes duplicate or cache-owned checkpoint slots')
    if owned['hybrid'] and (len(owned['fresh_cow_slots']) != 1 or not owned['cache_owned_cow_sources']):
        raise AssertionError('retract lacks actual fresh COW ownership before pool release')
    for kind, indices, metric, calls in (
            ('kv', kv, 'kv_pool_free', receipt['kv_free_calls']),
            ('mamba', mamba, 'mamba_pool_free', receipt['mamba_free_calls'])):
        if not indices:
            continue
        observed = {index for call in calls for index in call['indices']}
        if not set(indices) <= observed:
            raise AssertionError('retract did not call the actual allocator for every owned '+kind+' index')
        if after[metric] - before[metric] < len(indices):
            raise AssertionError('retract allocator did not return the owned '+kind+' capacity')
        if receipt[kind+'_free_before'] not in (None, []):
            raise AssertionError('retract counted already free '+kind+' indices as request-owned')
        membership = receipt[kind+'_free_after']
        if membership is not None and sorted(indices) != membership:
            raise AssertionError('retract owned '+kind+' indices are absent from the actual allocator free list')


def engine_snapshot(scheduler):
    tree = scheduler.tree_cache
    allocator = getattr(scheduler.req_to_token_pool, 'mamba_allocator', None)
    result = {'req_pool_free': scheduler.req_to_token_pool.available_size(),
              'kv_pool_free': scheduler.token_to_kv_pool_allocator.available_size(),
              'waiting': [r.rid for r in scheduler.waiting_queue],
              'cache_backend': type(tree).__module__+'.'+type(tree).__name__,
              'cache_components': [str(kind) for kind in getattr(tree, 'components', {})]}
    if allocator is not None:
        result['mamba_pool_free'] = allocator.available_size()
    for name in ('full_protected_size', 'mamba_protected_size', 'mamba_evictable_size'):
        method = getattr(tree, name, None)
        if callable(method):
            result[name] = method()
    locks = {}
    stack = list(tree.root_node.children.values())
    while stack:
        node = stack.pop()
        if hasattr(node, 'component_data'):
            value = sum(data.lock_ref for data in node.component_data)
        else:
            value = sum(getattr(node, name, 0) for name in
                        ('full_lock_ref', 'swa_lock_ref', 'mamba_lock_ref', 'lock_ref'))
        if value:
            locks[str(node.id)] = value
        stack.extend(node.children.values())
    result['locked_nodes'] = locks
    return result


class RankState:
    def __init__(self, scheduler, directory, bridge=None):
        self.scheduler = scheduler
        self.rank = scheduler.ps.tp_rank
        self.directory = Path(directory)
        self.path = self.directory / f'rank-{self.rank}.jsonl'
        self.seen = set()
        self.sequence = 0
        self._event_lock = threading.Lock()
        self.gates = {}
        self.delay_events = {}
        self.cuda_gates = {}
        self.request_arms = {}
        self.last_command = None
        self.bridge = bridge if bridge is not None else scheduler.spec_feature_bridge
        self.transactions = {}
        self.fixture_pins = {}
        self.current_admit = None
        self.retract_forward_seen = set()
        self.cow_receipts = {}
        self.emit('installed', engine=engine_snapshot(scheduler))

    def emit(self, event, **values):
        with self._event_lock:
            self.sequence += 1
            append_json(self.path, {'seq': self.sequence, 'rank': self.rank,
                                   'pid': os.getpid(), 'monotonic_ns': time.monotonic_ns(),
                                   'event': event, **values})

    def arm_for(self, req=None, request_id=None):
        arm = read_json(self.directory / 'arm.json', {})
        if not arm:
            return None
        if req is not None:
            if req.rid != arm['rid'] or (req.spec_capture or {}).get('sample_id') != arm['sample_id']:
                return None
        elif request_id != arm['rid']:
            return None
        return arm

    def fire(self, arm, point):
        if arm is None or arm['kind'] != point or self.rank not in arm['ranks']:
            return False
        key = (arm['case_id'], point)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.emit('fault_fired', case_id=arm['case_id'], kind=point)
        return True

    def command(self):
        command = read_json(self.directory / 'command.json', {})
        if not command or command['id'] == self.last_command:
            return
        self.last_command = command['id']
        try:
            self._apply_command(command)
        except Exception as exc:
            # A fixture error must not kill the scripted owner and strand peer
            # collectives/HTTP cleanup. Return an explicit failed rank receipt.
            import traceback
            atomic_json(self.directory / f'ack-{command["id"]}-rank{self.rank}.json',
                        {'rank':self.rank,'action':command['action'],
                         'error':repr(exc),'traceback':traceback.format_exc()})
            self.emit('command_failed',command_id=command['id'],action=command['action'],error=repr(exc))

    def _apply_command(self, command):
        action = command['action']
        probe = None
        if action == 'release_fixture_pins':
            for key in tuple(self.fixture_pins):
                self._release_fixture_pin(key)
        elif action == 'release_gate':
            gate = self.gates.get(command['case_id'])
            if gate is not None:
                gate.set()
        elif action in ('pin_kv', 'release_kv', 'pin_hs', 'release_hs', 'coverage'):
            tokens = tuple(command.get('tokens', ()))
            key = command.get('pin_id', 'fixture')
            if action == 'pin_kv':
                if key in self.fixture_pins:
                    raise AssertionError('duplicate live fixture pin identifier')
                result = self.probe_kv(tokens)
                probe = {'L':len(result.device_indices),'N':len(tokens),
                         'namespace':self.bridge.namespace,
                         **{name:getattr(result,name,None) for name in
                            ('full_kv_hit_length','mamba_branching_seqlen','last_device_node','best_match_node')}}
                self.emit('fixture_kv_probe',command_id=command['id'],pin_id=key,**probe)
                if not len(result.device_indices):
                    raise AssertionError('KV pin fixture has no actual reusable checkpoint')
                receipt = self.scheduler.tree_cache.inc_lock_ref(result.last_device_node)
                self.fixture_pins[key] = ('kv', result.last_device_node, receipt)
            elif action == 'release_kv':
                self._release_fixture_pin(key)
            elif self.bridge.writer:
                if action == 'pin_hs':
                    if key in self.fixture_pins:
                        raise AssertionError('duplicate live fixture pin identifier')
                    lease = self.bridge.cache.lookup_and_pin(self.bridge.namespace, tokens, len(tokens))
                    self.fixture_pins[key] = ('hs', lease)
                    if not lease.coverage_end:
                        raise AssertionError('HS pin fixture has no actual READY coverage')
                elif action == 'release_hs':
                    self._release_fixture_pin(key)
        elif action == 'retire_delay':
            delayed = self.delay_events.pop(command['case_id'], None)
            if delayed is not None:
                if not delayed['event'].query() or any(ref() is not None for ref in delayed['sources']):
                    raise AssertionError('actual delayed CUDA event/source ownership not drained')
                if delayed.get('gate') is not None:
                    delayed['gate'].close_after_drain()
                    self.cuda_gates.pop(command['case_id'])
        elif action != 'snapshot':
            raise ValueError('unsupported test command: ' + action)
        coverage = None
        if action == 'coverage' and self.bridge.writer:
            lease = self.bridge.cache.lookup_and_pin(self.bridge.namespace, tuple(command['tokens']), len(command['tokens']))
            coverage = lease.coverage_end
            self.bridge.cache.release_lease(lease)
        atomic_json(self.directory / f'ack-{command["id"]}-rank{self.rank}.json',
                    {'rank': self.rank, 'action': action, 'coverage': coverage, 'kv_probe':probe,
                     'engine': engine_snapshot(self.scheduler)})

    def _release_fixture_pin(self, key):
        owned = self.fixture_pins[key]
        if owned[0] == 'kv':
            _, node, receipt = owned
            params = receipt.to_dec_params() if hasattr(receipt,'to_dec_params') else receipt
            self.scheduler.tree_cache.dec_lock_ref(node, params)
        elif owned[0] == 'hs':
            self.bridge.cache.release_lease(owned[1])
        else:
            raise AssertionError('unknown fixture pin owner kind')
        # Do not discard the ownership record if its actual release raises.
        del self.fixture_pins[key]

    def probe_kv(self, tokens):
        from array import array
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        return self.scheduler.tree_cache.match_prefix(MatchPrefixParams(
            key=RadixKey(array('q',tokens), extra_key=self.bridge.namespace, limit=max(len(tokens)-1, 0)),
            cow_mamba=False, req=None))

    def wrap_admit(self, original, req, adder, *args, **kwargs):
        arm = self.arm_for(req)
        if arm is None:
            return original(req, adder, *args, **kwargs)
        self.request_arms[req.rid] = arm
        transaction = self.transactions.get(req.rid, 0) + 1
        self.transactions[req.rid] = transaction
        if arm['kind'] == 'coverage':
            result = self.probe_kv(tuple(req.origin_input_ids))
            self.emit('uncapped_engine_probe', case_id=arm['case_id'], rid=req.rid,
                      L=len(result.device_indices), cow_mamba=False)
        before = engine_snapshot(self.scheduler)
        slot_before = scalar(req.mamba_pool_idx)
        match = req.init_next_round_input
        add = adder.add_one_req
        allocator = getattr(self.scheduler.req_to_token_pool, 'mamba_allocator', None)
        freed = []
        fresh = []
        match_count = 0

        def matched(*match_args, **match_kwargs):
            nonlocal match_count
            match_count += 1
            value = match(*match_args, **match_kwargs)
            now = scalar(req.mamba_pool_idx)
            src = scalar(req.mamba_cow_src_index)
            proof = slot_before is None and now is not None and src is not None
            if proof:
                fresh.append(now)
                self.cow_receipts[req.rid] = {'transaction': transaction,
                    'slots': index_list(now), 'sources': index_list(src)}
            self.emit('after_actual_match', case_id=arm['case_id'], rid=req.rid,
                      transaction=transaction, call=match_count, cap=req.spec_feature_limit, L=len(req.prefix_indices),
                      fresh_cow=proof, slot=now, src=src)
            if self.fire(arm, 'match_error'):
                if arm.get('require_cow') and not proof:
                    raise AssertionError('fault fixture did not allocate an actual fresh Mamba COW slot')
                raise RuntimeError('test injected after actual engine match')
            return value

        def added(*add_args, **add_kwargs):
            from sglang.srt.managers.schedule_policy import AddReqResult
            for kind, result in (('admit_no_token', AddReqResult.NO_TOKEN),
                                 ('admit_no_membership', AddReqResult.CONTINUE)):
                if self.fire(arm, kind):
                    if arm.get('require_cow') and not fresh:
                        raise AssertionError('fault fixture never observed actual fresh COW')
                    return result
            result = add(*add_args, **add_kwargs)
            if self.fire(arm, 'admit_after_append'):
                if not any(item is req for item in adder.can_run_list):
                    raise AssertionError('actual PrefillAdder did not append target before injected exception')
                if arm.get('require_cow') and not fresh:
                    raise AssertionError('after-append fixture never observed actual fresh COW')
                raise RuntimeError('test injected after actual PrefillAdder append')
            return result

        def free(indices):
            freed.append(scalar(indices))
            return original_free(indices)

        original_free = allocator.free if allocator is not None else None
        prior_admit = self.current_admit
        self.current_admit = (arm['case_id'], transaction)
        try:
            with patched_attribute(req, 'init_next_round_input', matched), patched_attribute(adder, 'add_one_req', added):
                if allocator is None:
                    return original(req, adder, *args, **kwargs)
                with patched_attribute(allocator, 'free', free):
                    return original(req, adder, *args, **kwargs)
        finally:
            self.current_admit = prior_admit
            self.emit('admit_return', case_id=arm['case_id'], rid=req.rid,
                      transaction=transaction,
                      fresh_slots=fresh, freed_slots=freed,
                      member=any(item is req for item in adder.can_run_list),
                      slot_after=scalar(req.mamba_pool_idx), src_after=scalar(req.mamba_cow_src_index),
                      before=before, after=engine_snapshot(self.scheduler))

    def maybe_retract_prefill(self, plan, *, retract_all_fn=None, next_batch_plan=None):
        """Force a genuine allocated prefill retraction before its first forward.

        Every TP rank enters through the same ordinary get-next-batch call.
        No generated output is synthesized; the original HTTP request is
        requeued and must run the next genuine prefill under a new epoch.
        """
        batch = plan.batch_to_run
        if batch is None:
            return plan
        matches = [(req, self.arm_for(req)) for req in batch.reqs]
        matches = [(req, arm) for req, arm in matches if arm is not None and arm['kind'] == 'prefill_retract']
        if not matches:
            return plan
        req, arm = matches[0]
        if (arm['case_id'], 'prefill_retract') in self.seen:
            return plan
        if len(matches) != 1 or len(batch.reqs) != 1 or sorted(arm['ranks']) != list(range(self.scheduler.ps.tp_size)):
            raise AssertionError('prefill retract requires one exact request and every TP rank')
        if (not batch.forward_mode.is_extend() or not batch.is_prefill_only or req.sampling_params.max_new_tokens != 0
                or any(item is req for item in plan.running_batch.reqs)):
            raise AssertionError('prefill retract must preserve a max-new0 EXTEND batch')
        scalar_plan = dict(getattr(req, 'spec_feature_scalar_plan', {}) or {})
        old_epoch = req.spec_feature_execution_epoch
        if (old_epoch != 0 or req.retraction_count != 0 or req.finished()
                or req.req_pool_idx is None or req.kv is None or req.rid in self.retract_forward_seen
                or self.transactions.get(req.rid) != 1 or scalar_plan.get('execution_epoch') != 0):
            raise AssertionError('prefill retract lacks a first genuine allocated unexecuted admission')
        if (req.spec_feature_attempt.state != 'ADMITTED' or scalar_plan.get('L', 0) <= 0
                or scalar_plan.get('H', 0) < scalar_plan['L'] or len(req.prefix_indices) != scalar_plan['L']):
            raise AssertionError('prefill retract fixture lacks a real positive legal KV/HS checkpoint; warm an ordinary request before retrying the fixture')
        before = engine_snapshot(self.scheduler)
        controller_before = self.bridge.controller.stats() if self.bridge.controller is not None else None
        cache_before = self.bridge.cache.stats() if self.bridge.cache is not None else None
        if self.bridge.writer and (controller_before is None or controller_before['output_reservations'] != 1 or controller_before['output_bytes'] <= 0):
            raise AssertionError('prefill retract lacks a real owned output reservation')
        if self.bridge.writer and (cache_before is None or cache_before['active_leases'] <= 0):
            raise AssertionError('positive-L prefill retract lacks an active HS lease')
        if self.bridge.writer:
            frozen = req.spec_feature_plan
            self.bridge.cache._validate_plan_lease(frozen.lease, frozen.L, frozen.namespace, frozen.tokens)
            self.bridge.controller._reservation(frozen.reservation)
        if retract_all_fn is None or next_batch_plan is None:
            import torch
            from sglang.srt.managers.schedule_batch import retract_all, NextBatchPlan
            # Match/COW and prepare-for-extend can enqueue local GPU copies.
            # No forward was launched; drain those owned schedule-stream writes
            # once before the real pool release. This is a fault fixture only.
            schedule_stream = getattr(self.scheduler, 'schedule_stream', None)
            if schedule_stream is not None:
                schedule_stream.synchronize()
            torch.cuda.current_stream().synchronize()
            retract_all_fn, next_batch_plan = retract_all, NextBatchPlan
        if not self.fire(arm, 'prefill_retract'):
            raise AssertionError('prefill retract unexpectedly missed its all-rank one-shot')
        pool = self.scheduler.req_to_token_pool
        kv_allocator = self.scheduler.token_to_kv_pool_allocator
        mamba_allocator = getattr(pool, 'mamba_allocator', None)
        start, end = req.cache_protected_len, req.kv.kv_allocated_len
        if start != len(req.prefix_indices) or not start < end or kv_allocator.page_size != 1:
            raise AssertionError('retract requires an exact page1 protected-prefix/suffix allocation boundary')
        kv_indices = index_list(pool.req_to_token[req.req_pool_idx, start:end])
        cow = self.cow_receipts.get(req.rid, {})
        active = index_list(getattr(req, 'mamba_pool_idx', None))
        ping_pong = [index for index in index_list(getattr(req, 'mamba_ping_pong_track_buffer', None)) if index != -1]
        if mamba_allocator is not None and (cow.get('transaction') != 1 or active != cow.get('slots')):
            raise AssertionError('retract active Mamba slot lacks its actual fresh COW match receipt')
        owned = {'req_pool_idx': req.req_pool_idx, 'kv_start': start, 'kv_end': end,
                 'prefix_len': len(req.prefix_indices), 'kv_indices': kv_indices,
                 'hybrid': mamba_allocator is not None, 'fresh_cow_slots': active,
                 'ping_pong_slots': ping_pong, 'mamba_slots': active + ping_pong,
                 'cache_owned_cow_sources': cow.get('sources', [])}
        returned = {'owned': owned, 'kv_free_calls': [], 'mamba_free_calls': [],
                    'kv_free_before': owned_free_membership(kv_allocator, kv_indices),
                    'mamba_free_before': owned_free_membership(mamba_allocator, owned['mamba_slots'], mamba=True)}
        # Observe the original allocator operations; wrappers never manufacture a
        # free or change indices. Nested free_segment/free calls are harmless:
        # the oracle compares exact IDs and actual free-list/count state.
        with ExitStack() as stack:
            for name in ('free', 'free_segment', 'free_segments'):
                original = getattr(kv_allocator, name, None)
                if original is None:
                    continue
                def kv_free(value, *args, _name=name, _original=original, **kwargs):
                    indices = ([index for segment, _ in value for index in index_list(segment)]
                               if _name == 'free_segments' else index_list(value))
                    result = _original(value, *args, **kwargs)
                    returned['kv_free_calls'].append({'method': _name, 'indices': indices})
                    return result
                stack.enter_context(patched_attribute(kv_allocator, name, kv_free))
            if mamba_allocator is not None:
                original_mamba_free = mamba_allocator.free
                def mamba_free(value, *args, **kwargs):
                    indices = index_list(value)
                    result = original_mamba_free(value, *args, **kwargs)
                    returned['mamba_free_calls'].append({'method': 'free', 'indices': indices})
                    return result
                stack.enter_context(patched_attribute(mamba_allocator, 'free', mamba_free))
            retract_all_fn(reqs=[req], server_args=self.scheduler.server_args,
                           req_to_token_pool=pool, token_to_kv_pool_allocator=kv_allocator,
                           tree_cache=self.scheduler.tree_cache,
                           hisparse_coordinator=self.scheduler.hisparse_coordinator, offload_kv=False)
        after = engine_snapshot(self.scheduler)
        returned['kv_free_after'] = owned_free_membership(kv_allocator, kv_indices)
        returned['mamba_free_after'] = owned_free_membership(mamba_allocator, owned['mamba_slots'], mamba=True)
        self.emit('prefill_retract_allocation_return', case_id=arm['case_id'], rid=req.rid,
                  allocation_return=returned, before=before, after=after)
        assert_retract_allocation_return(returned, before, after)
        if (req.req_pool_idx is not None or req.kv is not None or req.spec_feature_plan is not None
                or req.spec_feature_limit != 0 or req.spec_feature_execution_epoch != old_epoch + 1
                or req.retraction_count != 1 or not req.spec_feature_force_full
                or req.spec_feature_attempt.state != 'QUEUED'
                or not all(drain.done() for drain in req.spec_feature_attempt.drains)
                or after['req_pool_free'] != before['req_pool_free'] + 1):
            raise AssertionError('real prefill retract did not free pools/old plan and advance the epoch')
        budgets = self.bridge.controller.stats() if self.bridge.controller is not None else None
        cache = self.bridge.cache.stats() if self.bridge.cache is not None else None
        if budgets is not None and any(budgets[name] for name in ('pinned_bytes', 'output_bytes', 'device_bytes', 'output_reservations', 'pending_copies')):
            raise AssertionError('old unexecuted prefill retained a controller reservation')
        if cache is not None and any(cache[name] for name in ('active_leases', 'fill_reservations', 'writing_blocks')):
            raise AssertionError('old unexecuted prefill retained an HS lease/fill')
        self.scheduler._add_request_to_queue(req)
        if sum(item is req for item in self.scheduler.waiting_queue) != 1:
            raise AssertionError('actual retracted prefill was not requeued exactly once')
        self.emit('prefill_retract_applied', case_id=arm['case_id'], rid=req.rid,
                  old_plan=scalar_plan, execution_epoch=req.spec_feature_execution_epoch,
                  retraction_count=req.retraction_count, force_full=req.spec_feature_force_full,
                  old_drains_done=True, pre_forward_stream_drained=True,
                  before=before, after=after, controller_before=controller_before, cache_before=cache_before,
                  controller=budgets, cache=cache, allocation_return=returned)
        return next_batch_plan(batch_to_run=None, running_batch=plan.running_batch)

    def observe_retracted_forward(self, batch):
        for req in batch.reqs:
            arm = self.arm_for(req)
            if arm is None or arm['kind'] != 'prefill_retract':
                continue
            scalar_plan = dict(getattr(req, 'spec_feature_scalar_plan', {}) or {})
            if ((arm['case_id'], 'prefill_retract') not in self.seen or req.rid in self.retract_forward_seen
                    or req.spec_feature_execution_epoch != 1 or req.retraction_count != 1
                    or not req.spec_feature_force_full or scalar_plan.get('execution_epoch') != 1
                    or scalar_plan.get('L') != 0 or len(req.prefix_indices) != 0
                    or self.transactions.get(req.rid) != 2):
                raise AssertionError('retracted HTTP request did not rematch and start exactly one full epoch1 forward')
            self.retract_forward_seen.add(req.rid)
            self.emit('prefill_retract_forward', case_id=arm['case_id'], rid=req.rid,
                      retraction_count=req.retraction_count, plan=scalar_plan,
                      transaction=self.transactions[req.rid])

    def hold_publication(self, arm):
        gate = threading.Event()
        self.gates[arm['case_id']] = gate
        self.emit('publish_gate_entered', case_id=arm['case_id'])
        atomic_json(self.directory / f'gate-{arm["case_id"]}.json', {'entered': True})
        if not gate.wait(float(arm.get('gate_timeout_s', 45))):
            raise TimeoutError('test publication barrier was not released')
        self.emit('publish_gate_released', case_id=arm['case_id'])


def install_bridge(state):
    bridge = state.bridge
    original_admit = bridge.admit
    bridge.admit = lambda req, adder, *args, **kwargs: state.wrap_admit(original_admit, req, adder, *args, **kwargs)
    original_phase = bridge.phase
    def phase():
        state.command()
        return original_phase()
    bridge.phase = phase
    originals = [(bridge, 'admit', original_admit), (bridge, 'phase', original_phase)]
    for name in ('broadcast', 'gather'):
        original = getattr(bridge, name)
        originals.append((bridge, name, original))
        def collective(value, _original=original, _name=name):
            if state.current_admit is not None:
                case_id, transaction = state.current_admit
                state.emit('admit_collective', method=_name, case_id=case_id, transaction=transaction)
            arm = read_json(state.directory / 'arm.json', {})
            if arm:
                state.emit('collective', method=_name, case_id=arm['case_id'])
            return _original(value)
        setattr(bridge, name, collective)
    original_control = bridge.control
    def control(body):
        release_gate = None
        if body.get('action') == 'resolve' and body.get('cancel'):
            arm = read_json(state.directory / 'arm.json', {})
            entry = state.delay_events.get(arm.get('case_id'))
            if entry is not None and body.get('sample_id') == arm.get('sample_id'):
                if not entry.get('cancel_observed'):
                    entry['cancel_observed'] = True
                    pending = not entry['event'].query()
                    state.emit('cancel_event_observation', case_id=arm['case_id'], event_pending=pending,
                               event_handle=int(entry['event'].cuda_event))
                    if entry.get('gate') is not None:
                        release_gate = entry['gate']
        try:
            result = original_control(body)
        except BaseException:
            if release_gate is not None:
                release_gate.release('control_error')
            raise
        else:
            # Invalidate the execution while the DMA is genuinely pending;
            # only then allow the real controller-owned stream to drain.
            if release_gate is not None:
                state.emit('cancel_control_applied',case_id=arm['case_id'],
                           state=result.get('state'),writer_drained=result.get('writer_drained'))
                release_gate.release('observed_cancel')
            return result
    bridge.control = control
    originals.append((bridge, 'control', original_control))
    # Collectives restored on worker exit via the global installation closure.
    return originals


def install_prefill_retract(state):
    scheduler = state.scheduler
    original_next = scheduler.get_next_batch_to_run
    original_run = scheduler.run_batch
    def next_batch(*args, **kwargs):
        return state.maybe_retract_prefill(original_next(*args, **kwargs))
    def run_batch(batch, *args, **kwargs):
        state.observe_retracted_forward(batch)
        return original_run(batch, *args, **kwargs)
    scheduler.get_next_batch_to_run, scheduler.run_batch = next_batch, run_batch
    return [(scheduler, 'get_next_batch_to_run', original_next), (scheduler, 'run_batch', original_run)]


def install_controller(state):
    controller = state.bridge.controller
    if controller is None:
        return []
    original_copy = controller._copy_suffix
    def copy(job):
        previous = getattr(_LOCAL, 'copy', None)
        arm = state.request_arms.get(job.plan.execution.request_id)
        _LOCAL.copy = {'state': state, 'arm': arm, 'job': job, 'issued': []}
        try:
            return original_copy(job)
        finally:
            _LOCAL.copy = previous
    controller._copy_suffix = copy
    return [(controller, '_copy_suffix', original_copy)]


def guarded_copy(original, target, source, *args, **kwargs):
    context = getattr(_LOCAL, 'copy', None)
    if (context is None or not threading.current_thread().name.startswith('spec-feature-capture')
            or not getattr(source, 'is_cuda', False) or not target.is_pinned()):
        return original(target, source, *args, **kwargs)
    state, arm, job = context['state'], context['arm'], context['job']
    names = [name for name, tensor in job.features.items()
             if tensor.data_ptr() == source.data_ptr() and tuple(tensor.shape) == tuple(source.shape)]
    if len(names) != 1:
        return original(target, source, *args, **kwargs)
    name = names[0]
    if name == 'aux' and arm is not None and arm['kind'] == 'd2h_partial' and state.rank in arm['ranks'] and not context['issued']:
        import torch
        torch.cuda._sleep(int(arm.get('delay_cycles', 2_000_000_000)))
    if name == 'aux' and arm is not None and arm['kind'] == 'd2h_cancel' and state.rank in arm['ranks'] and not context['issued']:
        import torch
        from actual_faults.cuda_gate import CudaCopyGate
        context['cancel_gate'] = CudaCopyGate(source.device, torch.cuda.current_stream(source.device),
            timeout_s=float(arm.get('gate_timeout_s',45)),
            emit=lambda event, **values: state.emit(event, case_id=arm['case_id'], **values))
        # Keep flag/stream ownership even if the following real copy raises.
        # Unknown drains remain retained until the owned test process exits.
        state.cuda_gates[arm['case_id']] = context['cancel_gate']
        context['cancel_gate'].arm()
    if name == 'last_hidden' and state.fire(arm, 'd2h_partial'):
        if not context['issued']:
            raise AssertionError('partial-copy fixture did not submit any actual CUDA D2H before failure')
        import torch
        stream = torch.cuda.current_stream(source.device)
        event = torch.cuda.Event()
        event.record(stream)
        pending = not event.query()
        state.delay_events[arm['case_id']] = {'event':event, 'stream':stream,
                                              'sources':[weakref.ref(t) for t in job.features.values()]}
        state.emit('partial_d2h_exception', case_id=arm['case_id'], issued=context['issued'], event_pending=pending)
        atomic_json(state.directory / f'partial-{arm["case_id"]}.json', {'pending':pending})
        raise RuntimeError('test injected second-feature D2H failure after real first DMA')
    value = original(target, source, *args, **kwargs)
    context['issued'].append(name)
    if name == 'aux' and state.fire(arm, 'd2h_cancel'):
        # Observe the actual first DMA on the controller-owned copy stream.
        # Delaying the producer snapshot event can expire before the scheduler
        # returns from forward completion and reaches HTTP control dispatch.
        import torch
        stream = torch.cuda.current_stream(source.device)
        event = torch.cuda.Event()
        event.record(stream)
        pending = not event.query()
        state.delay_events[arm['case_id']] = {'event': event, 'stream': stream,
            'sources': [weakref.ref(t) for t in context['job'].features.values()],
            'gate': context['cancel_gate']}
        state.emit('delayed_real_cuda_event', case_id=arm['case_id'], event_pending=pending,
                   stage='after_real_aux_d2h', issued=list(context['issued']),
                   event_handle=int(event.cuda_event), producer_event_handle=int(context['job'].ready_event.cuda_event),
                   source_pointers={key:value.data_ptr() for key,value in context['job'].features.items()})
        atomic_json(state.directory / f'event-{arm["case_id"]}.json',
                    {'pending': pending, 'stage': 'after_real_aux_d2h', 'issued': list(context['issued']),
                     'event_handle': int(event.cuda_event), 'monotonic_ns': time.monotonic_ns()})
    return value


def install(directory):
    """Called in every scheduler process before its original entry function."""
    import torch
    from sglang.srt.spec_feature_bridge import SpecFeatureBridge
    from sglang.srt.spec_capture_sink import SpecCaptureSink
    old_init = SpecFeatureBridge.__init__
    old_put = SpecCaptureSink.put_samples
    old_copy = torch.Tensor.copy_
    restorations = []
    def init(bridge, scheduler):
        old_init(bridge, scheduler)
        state = RankState(scheduler, directory, bridge=bridge)
        _STATES.append(state)
        restorations.extend(install_bridge(state))
        restorations.extend(install_prefill_retract(state))
        restorations.extend(install_controller(state))
        state.emit('source_manifest', sources={inspect.getsourcefile(type(bridge)): source_hash(inspect.getsourcefile(type(bridge)))})
    def put(sink, samples):
        for state in _STATES:
            for spec, _, _ in samples:
                arm = read_json(state.directory / 'arm.json', {})
                if spec.get('sample_id') != arm.get('sample_id') or state.rank not in arm.get('ranks', []):
                    continue
                if state.fire(arm, 'publish_hold'):
                    state.hold_publication(arm)
                if state.fire(arm, 'publish_fail'):
                    result = old_put(sink, samples)
                    state.emit('native_publish_succeeded_then_failed', case_id=arm['case_id'], objects=[list(row['features']) for row in result])
                    raise RuntimeError('test injected failure after actual native Mooncake publication')
        return old_put(sink, samples)
    SpecFeatureBridge.__init__, SpecCaptureSink.put_samples = init, put
    torch.Tensor.copy_ = lambda target, source, *args, **kwargs: guarded_copy(old_copy, target, source, *args, **kwargs)
    def restore():
        for owner, name, original in reversed(restorations):
            setattr(owner, name, original)
        SpecFeatureBridge.__init__, SpecCaptureSink.put_samples, torch.Tensor.copy_ = old_init, old_put, old_copy
    return restore
