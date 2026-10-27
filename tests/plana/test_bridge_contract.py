"""Independent tests invoke the actual B registry/journal/bridge methods.

Fake engine pool and sink are deterministic resource/future boundaries; they do
not replace any registry, transaction, rollback, or publication-fence logic.
"""
from collections import deque
from concurrent.futures import Future
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

try:
    import sglang.srt
    if os.environ.get('PLANA_RUNTIME_PYTHON'):
        sglang.srt.__path__.append(str(Path(os.environ['PLANA_RUNTIME_PYTHON'])/'sglang/srt'))
    from sglang.srt.spec_feature_bridge import AdmissionJournal,AttemptRegistry,SpecFeatureBridge,digest_request
    BRIDGE=True
except ModuleNotFoundError as exc:
    if exc.name not in ('sglang','sglang.srt.spec_feature_bridge'):raise
    BRIDGE=False


class Pool:
    def __init__(self):self.freed=[]
    def free(self,value):self.freed.append(value.detach().clone())


class LockReceipt:
    def __init__(self,value):self.value=value
    def to_dec_params(self):return self.value


class Tree:
    def __init__(self):
        self.released=[];self.pool=Pool()
        self.req_to_token_pool=SimpleNamespace(mamba_allocator=self.pool)
    def dec_lock_ref(self,node,params=None):self.released.append((node,params))


@unittest.skipUnless(BRIDGE,'candidate bridge not integrated; pending B acceptance')
class BridgeContracts(unittest.TestCase):
    def bridge(self):
        value=SpecFeatureBridge.__new__(SpecFeatureBridge)
        value.s=SimpleNamespace(ps=SimpleNamespace(tp_size=1,tp_rank=0),tree_cache=Tree(),waiting_queue=[])
        value.writer=True;value.enabled=True;value.epoch='server-epoch';value.namespace='capture-namespace';value.signature='capture-signature'
        value.registry=AttemptRegistry(value.epoch,capacity=2);value.jobs={};value.batches=[];value.history=deque(maxlen=8)
        value.cache=None;value.controller=None
        return value
    def body(self,attempt='a',sample='same-g1'):
        tokens=(1,2,3,4)
        spec={'features':{'aux':'hidden_states','last_hidden':'target_last_hidden_states'},'passthrough':[]}
        return {'protocol_version':1,'server_epoch':'server-epoch','action':'begin','store_id':'store','sample_id':sample,'gen':1,
                'attempt_id':attempt,'request_digest':digest_request(tokens,spec),**spec}

    def test_delayed_writer_and_producer_cleanup_both_block_replacement(self):
        bridge=self.bridge();body=self.body();wire=bridge.control(body);entry=bridge.registry.lookup(wire)
        req=SimpleNamespace(rid='request-a',spec_feature_attempt=entry,spec_feature_execution_epoch=0,spec_feature_plan=None,spec_capture=body)
        writer=Future();sink=SimpleNamespace(submit_samples=lambda samples:writer)
        output=bridge.queue_captures([(req,body,None,None,False,0)],sink)
        self.assertEqual(entry.deliveries,1)
        resolve=bridge.control({**wire,'action':'resolve','cancel':True})
        self.assertFalse(resolve['writer_drained'])
        with self.assertRaises(ValueError):bridge.control(self.body(attempt='b'))
        with self.assertRaises(ValueError):bridge.control({**wire,'action':'cleanup_done'})
        writer.set_result([{'sample_id':'same-g1','features':{}}]);bridge.poll()
        self.assertTrue(output.done());self.assertEqual(entry.deliveries,0)
        self.assertEqual(entry.state,'WRITER_DRAINED')
        # Writer completion alone does not authorize replacement: producer's
        # remove/drain and explicit cleanup_done have not occurred yet.
        with self.assertRaises(ValueError):bridge.control(self.body(attempt='b'))
        retired=bridge.control({**wire,'action':'cleanup_done'});self.assertEqual(retired['state'],'RETIRED')
        replacement=bridge.control(self.body(attempt='b'))
        self.assertNotEqual(replacement['reservation_token'],wire['reservation_token'])
        # Replayed old cleanup must not retire the new same-g1 reservation.
        bridge.control({**wire,'action':'cleanup_done'})
        self.assertEqual(bridge.registry.lookup(replacement).attempt_id,'b')

    def test_retired_token_delayed_generate_is_rejected_and_registry_bounded(self):
        bridge=self.bridge();body=self.body();old=bridge.control(body);entry=bridge.registry.lookup(old)
        with self.assertRaises(ValueError):bridge.registry.claim({**body,**old,'reservation_token':'unknown'},(1,2,3,4))
        entry.state='PUBLISHED';bridge.control({**old,'action':'cleanup_done'})
        fresh=bridge.control(self.body(attempt='b'))
        with self.assertRaises(ValueError):bridge.registry.claim({**body,**old},(1,2,3,4))
        self.assertNotEqual(old['reservation_token'],fresh['reservation_token'])
        bridge.control(self.body(sample='second'))
        with self.assertRaises(ValueError):bridge.control(self.body(sample='third'))
        self.assertEqual(len(bridge.registry.active),2)

    def test_journal_restores_after_append_and_releases_exact_lock_once(self):
        tree=Tree();existing=object();req=SimpleNamespace(extend_range='before',skip_lock_node_ids={'old'},swa_uuid_for_lock='swa-before')
        adder=SimpleNamespace(tree_cache=tree,can_run_list=[existing],preempt_list=[],rem_total_tokens=100.,rem_input_tokens=80,hit_tokens=3)
        original_list=adder.can_run_list;journal=AdmissionJournal(adder,req)
        journal.record_lock('node-1',LockReceipt({'receipt':1}));journal.record_lock('node-2',LockReceipt({'receipt':2}))
        adder.can_run_list.append(req);adder.preempt_list.append(existing);adder.rem_total_tokens=0.;adder.rem_input_tokens=0;adder.hit_tokens=99
        req.extend_range='after';req.skip_lock_node_ids={'new'};req.swa_uuid_for_lock='swa-after'
        journal.rollback();journal.rollback()
        self.assertIs(adder.can_run_list,original_list);self.assertEqual(adder.can_run_list,[existing]);self.assertEqual(adder.preempt_list,[])
        self.assertEqual((adder.rem_total_tokens,adder.rem_input_tokens,adder.hit_tokens),(100.,80,3))
        self.assertEqual((req.extend_range,req.skip_lock_node_ids,req.swa_uuid_for_lock),('before',{'old'},'swa-before'))
        self.assertEqual(tree.released,[('node-2',{'receipt':2}),('node-1',{'receipt':1})])

    def test_fresh_cow_slot_rollback_preserves_preexisting_state(self):
        bridge=self.bridge();req=SimpleNamespace(mamba_pool_idx=torch.tensor(7),mamba_cow_src_index=3,mamba_needs_clear=True)
        before={'mamba_pool_idx':None,'mamba_cow_src_index':None,'mamba_needs_clear':False}
        bridge._rollback_match(req,before)
        self.assertEqual([v.tolist() for v in bridge.s.tree_cache.pool.freed],[[7]])
        self.assertIsNone(req.mamba_pool_idx);self.assertIsNone(req.mamba_cow_src_index);self.assertFalse(req.mamba_needs_clear)
        existing=torch.tensor(9);req.mamba_pool_idx=existing
        bridge._rollback_match(req,{'mamba_pool_idx':existing,'mamba_cow_src_index':1,'mamba_needs_clear':False})
        self.assertIs(req.mamba_pool_idx,existing);self.assertEqual(len(bridge.s.tree_cache.pool.freed),1)

    def test_actual_admit_refusal_and_after_append_exception_reclaim_resources(self):
        try:
            from sglang.srt.spec_feature_cache import CaptureController,ExecutionIdentity,FeatureSchema,FeatureSpec,PrefixFeatureCache
        except ModuleNotFoundError:
            self.skipTest('actual A runtime not supplied for B integration fixture')
        from sglang.srt.managers.schedule_policy import AddReqResult
        for mode in ('CONTINUE-without-membership','NO_TOKEN-before-append','exception-after-append'):
            bridge=self.bridge();schema=FeatureSchema((FeatureSpec('aux',2,'float32',(1,)),FeatureSpec('last_hidden',1,'float32',())))
            bridge.cache=PrefixFeatureCache(schema,chunk_tokens=4,capacity_bytes=4096)
            bridge.controller=CaptureController(bridge.cache,schema,4096,4096,4096)
            self.addCleanup(bridge.controller.shutdown,wait=True)
            entry=bridge.registry.begin(self.body(attempt=mode));req=SimpleNamespace(rid=mode,origin_input_ids=list(range(8)),spec_capture={'passthrough':[]},
                spec_feature_attempt=entry,spec_feature_execution_epoch=0,prefix_indices=torch.empty(0,dtype=torch.int64),mamba_pool_idx=None,mamba_cow_src_index=None,mamba_needs_clear=False,
                extend_range='before',skip_lock_node_ids=set(),swa_uuid_for_lock=None)
            def match(tree):
                req.prefix_indices=torch.empty(0,dtype=torch.int64);req.mamba_pool_idx=torch.tensor(7);req.mamba_cow_src_index=3
            req.init_next_round_input=match
            adder=SimpleNamespace(tree_cache=bridge.s.tree_cache,can_run_list=[],preempt_list=[],rem_total_tokens=100,rem_input_tokens=100)
            def add(*a,**kw):
                adder.rem_total_tokens-=8;req.extend_range='after'
                if mode=='exception-after-append':
                    adder.can_run_list.append(req);adder._spec_feature_journal.record_lock('node',LockReceipt({'exact':1}));raise RuntimeError('injected after append')
                return AddReqResult.NO_TOKEN if mode.startswith('NO_TOKEN') else AddReqResult.CONTINUE
            adder.add_one_req=add
            if mode.startswith('exception'):
                with self.assertRaises(RuntimeError):bridge.admit(req,adder,False,1)
            else:self.assertEqual(bridge.admit(req,adder,False,1),AddReqResult.OTHER)
            self.assertEqual(adder.can_run_list,[]);self.assertEqual(adder.rem_total_tokens,100);self.assertEqual(req.extend_range,'before')
            self.assertIsNone(req.mamba_pool_idx);self.assertEqual(len(bridge.s.tree_cache.pool.freed),1)
            stats=bridge.controller.stats()
            for name in ('active_leases','output_bytes','pinned_bytes','device_bytes'):self.assertEqual(stats[name],0,(mode,name,stats))


if __name__=='__main__':unittest.main()
