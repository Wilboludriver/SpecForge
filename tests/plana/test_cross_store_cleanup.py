"""Actual FeatureStore + actual bridge cross-process cleanup permission tests.

Two store instances share a byte backend and real server control logic. The
in-memory backend only models raw transport; fencing is never reimplemented.
"""
from collections import deque
import ctypes
import importlib.util
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

try:
    from sglang.srt.spec_feature_bridge import AttemptRegistry,SpecFeatureBridge,digest_request
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
    from specforge.runtime.contracts import FeatureSpec,SampleRef
    AVAILABLE=hasattr(MooncakeFeatureStore,'track_capture_fence')
except ModuleNotFoundError as exc:
    if exc.name not in ('sglang','sglang.srt.spec_feature_bridge'):raise
    AVAILABLE=False


class ByteBackend:
    def __init__(self):self.data={};self.remove_calls=[];self.get_calls=[];self.fail_remove=False;self.fail_get=False
    def register_buffer(self,ptr,size):return 0
    def unregister_buffer(self,ptr):return 0
    def is_exist(self,key):return int(key in self.data)
    def put_from(self,key,ptr,size,config=None):self.data[key]=ctypes.string_at(ptr,size);return 0
    def get_into(self,key,ptr,size):
        self.get_calls.append(key)
        if self.fail_get:raise RuntimeError('injected uncertain native receive failure')
        if key not in self.data:return -1
        data=self.data[key];ctypes.memmove(ptr,data,min(size,len(data)));return min(size,len(data))
    def remove(self,key,**kw):
        self.remove_calls.append(key)
        if self.fail_remove:return -1
        self.data.pop(key,None);return 0


@unittest.skipUnless(AVAILABLE,'v1.2 bridge/store not integrated; pending cross-process cleanup acceptance')
class CrossStoreCleanup(unittest.TestCase):
    def setUp(self):
        self.backend=ByteBackend();self.clock_value=0.;self.stores=[];self.events=[];self.lost_cleanup_reply=False;self.lost_read_reply=False
        bridge=SpecFeatureBridge.__new__(SpecFeatureBridge)
        bridge.s=SimpleNamespace(ps=SimpleNamespace(tp_size=1,tp_rank=0),waiting_queue=[])
        bridge.writer=True;bridge.enabled=False;bridge.epoch='epoch';bridge.namespace='ns';bridge.signature='sig'
        bridge.registry=AttemptRegistry('epoch',capacity=16);bridge.jobs={};bridge.batches=[];bridge.history=deque(maxlen=8)
        bridge.cache=None;bridge.controller=None;self.bridge=bridge
    def tearDown(self):
        # No implicit destructive fake cleanup should hide a failing assertion.
        # Connections are injected and own no background service/registration.
        for store in self.stores:
            pool=getattr(store,'_receive_pool',None)
            if pool is not None:pool.close()
    def control(self,url,*,json_body,timeout):
        result=self.bridge.control(json_body);self.events.append((json_body['action'],result.get('state')))
        if json_body['action']=='read_acquire' and self.lost_read_reply:
            self.lost_read_reply=False
            raise TimeoutError('injected lost read permit response after server grant')
        if json_body['action']=='cleanup_done' and self.lost_cleanup_reply:
            self.lost_cleanup_reply=False
            raise TimeoutError('injected lost response after server retirement')
        return result
    def begin(self,attempt):
        spec={'features':{'aux':'hidden_states'},'passthrough':[]}
        body={'protocol_version':1,'action':'begin','server_epoch':'epoch','store_id':'shared-store','sample_id':'same-sample','gen':1,
              'attempt_id':attempt,'request_digest':digest_request((1,2,3,4),spec)}
        wire=self.bridge.control(body)
        return {**wire,'control_url':'http://fake/spec_capture/control','timeout_s':.1}
    def published(self,wire,value=1.):
        self.bridge.registry.lookup(wire).state='PUBLISHED'
        tensor=torch.full((1,4,2),value,dtype=torch.float32)
        key='shared-store/same-sample/g1/hidden_states';self.backend.data[key]=tensor.numpy().tobytes()
        return SampleRef(sample_id='same-sample',run_id='fixture',source_task_id=wire['attempt_id'],feature_store_uri='mooncake://shared-store/same-sample',
                         feature_keys={'hidden_states':'same-sample/hidden_states'},feature_specs={'hidden_states':FeatureSpec('hidden_states',(1,4,2),'float32')},
                         strategy='dflash',num_tokens=4,estimated_bytes=32,metadata={'generation':1,'capture_fence':wire})
    def store(self,wire,ref):
        store=MooncakeFeatureStore(store=self.backend,store_id='shared-store',max_hold_age_s=1.,clock=lambda:self.clock_value)
        store.track_capture_fence(wire,['hidden_states'],control_fn=self.control);store.adopt(ref);self.stores.append(store)
        return store
    def new_old_pair(self):
        wire=self.begin('attempt-a');ref=self.published(wire)
        producer=self.store(wire,ref);consumer=self.store(wire,ref)
        return wire,ref,producer,consumer
    def assert_replacement(self,value):
        key='shared-store/same-sample/g1/hidden_states'
        self.assertIn(key,self.backend.data)
        expected=torch.full((1,4,2),value,dtype=torch.float32).numpy().tobytes()
        self.assertEqual(self.backend.data[key],expected)

    def test_old_producer_gc_cannot_delete_replacement_after_consumer_retires(self):
        old,ref,producer,consumer=self.new_old_pair()
        _,handle=consumer.get(ref);consumer.release(handle)
        replacement=self.begin('attempt-b');self.published(replacement,7.)
        calls=len(self.backend.remove_calls);self.clock_value=10.
        producer.gc();producer.drain_pending_removals(max_attempts=2,retry_interval_s=0)
        self.assertEqual(len(self.backend.remove_calls),calls,'stale producer performed a physical delete')
        self.assert_replacement(7.)
        self.assertEqual(self.bridge.registry.lookup(replacement).attempt_id,'attempt-b')

    def test_failed_remove_retains_permit_and_second_owner_is_busy(self):
        old,ref,producer,consumer=self.new_old_pair();self.backend.fail_remove=True
        _,handle=consumer.get(ref);consumer.release(handle)
        fence=consumer._capture_fences[ref.sample_id];token=fence.cleanup_token
        self.assertTrue(token)
        reply=self.control('fake',json_body={**old,'action':'cleanup_acquire','cleanup_owner':fence.cleanup_owner},timeout=.1)
        self.assertEqual(reply['cleanup_token'],token,'same-owner acquire was not idempotent')
        calls=len(self.backend.remove_calls)
        with self.assertRaises(Exception):producer.abort(ref.sample_id,attempt_token=old['reservation_token'])
        self.assertEqual(len(self.backend.remove_calls),calls,'other owner removed keys without a permit')
        with self.assertRaises(ValueError):self.begin('attempt-b')
        self.backend.fail_remove=False;consumer.drain_pending_removals(max_attempts=3,retry_interval_s=0)
        replacement=self.begin('attempt-b');self.published(replacement,9.)
        self.clock_value=10.;producer.gc();self.assert_replacement(9.)

    def test_lost_cleanup_done_response_is_idempotent_after_replacement(self):
        old,ref,producer,consumer=self.new_old_pair();self.lost_cleanup_reply=True
        _,handle=consumer.get(ref)
        with self.assertRaises(TimeoutError):consumer.release(handle)
        replacement=self.begin('attempt-b');self.published(replacement,11.)
        calls=len(self.backend.remove_calls)
        consumer.drain_pending_removals(max_attempts=3,retry_interval_s=0)
        self.assertEqual(len(self.backend.remove_calls),calls)
        self.assert_replacement(11.)
        self.assertEqual(self.bridge.registry.lookup(replacement).attempt_id,'attempt-b')

    def test_lost_begin_reply_recovers_same_intent_before_cleanup_and_retry(self):
        from specforge.inference.capture_protocol import CaptureProtocol
        lost=[True];seen=[]
        def transport(url,*,json_body,timeout):
            reply=self.control(url,json_body=json_body,timeout=timeout)
            if json_body['action']=='begin':
                seen.append((json_body['attempt_id'],json_body['request_digest'],reply['reservation_token']))
                if lost[0]:
                    lost[0]=False
                    raise TimeoutError('injected lost begin response after server registration')
            return reply
        protocol=CaptureProtocol('http://fake',transport,.1)
        payload={'store_id':'shared-store','sample_id':'same-sample','gen':1,'features':{'aux':'hidden_states'},'passthrough':[]}
        store=MooncakeFeatureStore(store=self.backend,store_id='shared-store');self.stores.append(store)
        with self.assertRaises(TimeoutError):
            store.begin_capture_attempt(protocol,{'server_epoch':'epoch'},payload,(1,2,3,4),'begin-attempt',['hidden_states'])
        self.assertEqual(len(self.bridge.registry.active),1)
        self.assertIn('same-sample',store._capture_pending_begin)
        store.prepare_capture_retry('same-sample')
        self.assertEqual(len(seen),2);self.assertEqual(seen[0],seen[1],'lost begin recovery changed attempt or token')
        self.assertEqual(store._capture_pending_begin,{})
        self.assertEqual(len(self.bridge.registry.active),0)
        self.assertEqual(store.health()['provisional_external'],0)
        self.begin('replacement-after-recovered-begin')

    def test_fresh_consumer_stale_oldref_rejected_before_fetching_replacement(self):
        old,ref,producer,consumer=self.new_old_pair()
        _,handle=consumer.get(ref);consumer.release(handle)
        replacement=self.begin('attempt-b');self.published(replacement,17.)
        fresh=MooncakeFeatureStore(store=self.backend,store_id='shared-store');self.stores.append(fresh)
        # Supply only the real control transport seam; the fresh store has no
        # local released tombstone and must validate oldref at the server.
        fresh.track_capture_fence(old,['hidden_states'],control_fn=self.control)
        calls=len(self.backend.get_calls)
        with self.assertRaises(Exception):fresh.get(ref)
        self.assertEqual(len(self.backend.get_calls),calls,'stale oldref reached physical read of new g1 bytes')
        self.assert_replacement(17.)

    def test_active_remote_read_handle_blocks_cleanup_until_release(self):
        old,ref,producer,consumer=self.new_old_pair()
        tensors,handle=consumer.get(ref)
        self.assertTrue(torch.equal(tensors['hidden_states'],torch.ones(1,4,2)))
        calls=len(self.backend.remove_calls)
        try:producer.abort(ref.sample_id,attempt_token=old['reservation_token'])
        except Exception:pass
        self.assertEqual(len(self.backend.remove_calls),calls,'cleanup removed keys held by another process read handle')
        self.assert_replacement(1.)
        with self.assertRaises(ValueError):self.begin('attempt-b')
        consumer.release(handle)
        replacement=self.begin('attempt-b');self.published(replacement,19.)
        producer.drain_pending_removals(max_attempts=2,retry_interval_s=0)
        self.assert_replacement(19.)

    def test_lost_read_acquire_recovers_held_owner_after_producer_revocation(self):
        old,ref,producer,consumer=self.new_old_pair();self.lost_read_reply=True
        calls=len(self.backend.get_calls)
        with self.assertRaises(TimeoutError):consumer.get(ref)
        self.assertEqual(len(self.backend.get_calls),calls,'failed read-acquire reached native fetch')
        self.assertEqual(consumer.health()['capture_read_intents'],1)
        try:producer.abort(ref.sample_id,attempt_token=old['reservation_token'])
        except Exception:pass
        self.assert_replacement(1.)
        self.clock_value=10.
        # Recovery must return the already-held owner's token even when A was
        # revoked, so it can release without authorizing any new reader.
        consumer.gc();consumer.drain_pending_removals(max_attempts=3,retry_interval_s=0)
        self.assertEqual(consumer.health()['capture_read_intents'],0)
        replacement=self.begin('attempt-b');self.published(replacement,23.)
        producer.drain_pending_removals(max_attempts=2,retry_interval_s=0)
        self.assert_replacement(23.)

    def test_uncertain_native_receive_keeps_read_permit_and_blocks_cleanup(self):
        old,ref,producer,consumer=self.new_old_pair();self.backend.fail_get=True
        with self.assertRaises(Exception):consumer.get(ref)
        self.assertGreater(consumer.health()['capture_failed_reads'],0)
        calls=len(self.backend.remove_calls)
        try:producer.abort(ref.sample_id,attempt_token=old['reservation_token'])
        except Exception:pass
        self.assertEqual(len(self.backend.remove_calls),calls)
        with self.assertRaises(Exception):consumer.drain_pending_removals(max_attempts=1,retry_interval_s=0)
        self.assertEqual(len(self.backend.remove_calls),calls)
        with self.assertRaises(ValueError):self.begin('attempt-b')
        self.assert_replacement(1.)

    def test_old_handle_release_keeps_original_fence_after_reentrant_replacement(self):
        old,ref,producer,consumer=self.new_old_pair();_,handle=consumer.get(ref)
        original=consumer._release_capture_read;replacement=[];new_refs=[]
        def switch_after_remote_reader_release(intent):
            original(intent)
            consumer.abort(ref.sample_id,attempt_token=old['reservation_token'])
            wire=self.begin('replacement-between-reader-release-and-cleanup');new_ref=self.published(wire,29.)
            consumer.track_capture_fence(wire,['hidden_states'],control_fn=self.control);consumer.adopt(new_ref)
            replacement.append(wire);new_refs.append(new_ref)
        with mock.patch.object(consumer,'_release_capture_read',side_effect=switch_after_remote_reader_release):
            consumer.release(handle)
        self.assert_replacement(29.)
        self.assertEqual(self.bridge.registry.lookup(replacement[0]).attempt_id,'replacement-between-reader-release-and-cleanup')
        self.assertNotIn(('same-sample',1),consumer._freed,'old release tombstoned authorized replacement')
        self.assertNotIn('same-sample',consumer._release_pending)
        values,new_handle=consumer.get(new_refs[0]);self.assertTrue(torch.equal(values['hidden_states'],torch.full((1,4,2),29.)))
        consumer.release(new_handle)

    def test_retain_on_release_never_tombstones_or_acquires_cleanup(self):
        old,ref,producer,consumer=self.new_old_pair();consumer.retain_on_release=True
        _,handle=consumer.get(ref);consumer.release(handle)
        self.assertEqual(self.backend.remove_calls,[])
        self.assertNotIn(('same-sample',1),consumer._freed);self.assertNotIn('same-sample',consumer._release_pending)
        self.assertFalse(any(action=='cleanup_acquire' for action,state in self.events),'retained read release requested deletion permission')
        values,handle2=consumer.get(ref);self.assertTrue(torch.equal(values['hidden_states'],torch.ones(1,4,2)));consumer.release(handle2)
        self.assertEqual(consumer.health()['active_leases'],0);self.assertEqual(consumer.health()['capture_active_reads'],0)
        self.assertEqual(self.bridge.registry.lookup(old).state,'PUBLISHED');self.assert_replacement(1.)

    def test_unknown_old_server_state_fails_closed_before_delete(self):
        old,ref,producer,consumer=self.new_old_pair()
        self.bridge.registry.tokens.clear();self.bridge.registry.active.clear();self.bridge.registry.retired.clear()
        calls=len(self.backend.remove_calls);self.clock_value=10.
        try:producer.gc()
        except Exception:pass
        self.assertEqual(len(self.backend.remove_calls),calls,'unknown server reservation authorized deletion')
        self.assert_replacement(1.)


if __name__=='__main__':unittest.main()
