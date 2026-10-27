"""Actual FeatureStore + actual bridge cross-process cleanup permission tests.

Two store instances share a byte backend and real server control logic. The
in-memory backend only models raw transport; fencing is never reimplemented.
"""
from collections import deque
import ctypes
import importlib.util
from types import SimpleNamespace
import unittest

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
    def __init__(self):self.data={};self.remove_calls=[];self.fail_remove=False
    def register_buffer(self,ptr,size):return 0
    def unregister_buffer(self,ptr):return 0
    def is_exist(self,key):return int(key in self.data)
    def put_from(self,key,ptr,size,config=None):self.data[key]=ctypes.string_at(ptr,size);return 0
    def get_into(self,key,ptr,size):
        if key not in self.data:return -1
        data=self.data[key];ctypes.memmove(ptr,data,min(size,len(data)));return min(size,len(data))
    def remove(self,key,**kw):
        self.remove_calls.append(key)
        if self.fail_remove:return -1
        self.data.pop(key,None);return 0


@unittest.skipUnless(AVAILABLE,'v1.2 bridge/store not integrated; pending cross-process cleanup acceptance')
class CrossStoreCleanup(unittest.TestCase):
    def setUp(self):
        self.backend=ByteBackend();self.clock_value=0.;self.stores=[];self.events=[];self.lost_cleanup_reply=False
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

    def test_unknown_old_server_state_fails_closed_before_delete(self):
        old,ref,producer,consumer=self.new_old_pair()
        self.bridge.registry.tokens.clear();self.bridge.registry.active.clear();self.bridge.registry.retired.clear()
        calls=len(self.backend.remove_calls);self.clock_value=10.
        try:producer.gc()
        except Exception:pass
        self.assertEqual(len(self.backend.remove_calls),calls,'unknown server reservation authorized deletion')
        self.assert_replacement(1.)


if __name__=='__main__':unittest.main()
