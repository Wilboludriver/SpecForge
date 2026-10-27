"""Independent public-API cache/controller acceptance fixtures.

Run with the committed candidate SGLang on PYTHONPATH. A missing runtime is a
visible skip for the official-only baseline, never evidence of runtime passing.
No mocks replace cache/controller logic; manual events delay actual workers.
"""
import dataclasses
import importlib.util
import threading
import unittest

import torch

try:
    from sglang.srt.spec_feature_cache import (
        CaptureController,CapturedSpan,ExecutionIdentity,FeatureSchema,
        FeatureSpec,PrefixFeatureCache,
    )
    AVAILABLE=True
except ModuleNotFoundError as exc:
    if exc.name not in ('sglang','sglang.srt.spec_feature_cache'):
        raise
    AVAILABLE=False


class CompletionGate:
    def __init__(self):
        self.entered=threading.Event();self.complete=threading.Event()
    def synchronize(self):
        self.entered.set()
        if not self.complete.wait(10):
            raise TimeoutError('test event was never explicitly completed')


@unittest.skipUnless(AVAILABLE,'candidate spec_feature_cache not yet integrated; pending runtime acceptance')
class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.schema=FeatureSchema((FeatureSpec('aux',2,'float32',(1,3)),FeatureSpec('last_hidden',1,'float32',())))
        self.tokens=tuple(range(11));self.ns='independent-causal-fixture';self.block_bytes=4*self.schema.row_bytes
        self.cache=PrefixFeatureCache(self.schema,chunk_tokens=4,capacity_bytes=8*self.block_bytes)
        self.execution=ExecutionIdentity('server-a','request-a','attempt-a',0)
    def values(self,n=11):
        # Distinct channels and absolute rows: exact binary32 integers.
        row=torch.arange(n,dtype=torch.float32).unsqueeze(1)
        return {'aux':torch.cat((row*16+1,row*16+7),dim=1),'last_hidden':row*16+13}
    def fill(self,key,values=None,cache=None):
        cache=cache or self.cache;reservation=cache.reserve_fill(key,self.execution)
        self.assertIsNotNone(reservation)
        full=values or self.values()
        self.assertTrue(cache.publish_ready(reservation,{name:value[key.start:key.end].clone() for name,value in full.items()}))
        return reservation
    def warm(self,count=2):
        keys=self.cache.block_keys(self.ns,self.tokens)
        for key in keys[:count]:self.fill(key)
        return keys
    def read(self,lease,end):
        out={spec.name:torch.empty((end,spec.width),dtype=torch.float32) for spec in self.schema.features}
        self.cache.copy_prefix_into(lease,end,out)
        return out
    def controller(self):
        controller=CaptureController(self.cache,self.schema,pinned_budget_bytes=4096,output_budget_bytes=4096,device_budget_bytes=4096)
        self.addCleanup(controller.shutdown,wait=True)
        return controller
    def plan(self,controller,L=0,execution=None):
        execution=execution or self.execution
        lease=self.cache.lookup_and_pin(self.ns,self.tokens,len(self.tokens))
        lease=self.cache.shrink_lease(lease,L)
        reservation=controller.try_reserve_output(execution,11,L,0)
        self.assertIsNotNone(reservation)
        return controller.freeze_plan(execution,self.ns,self.tokens,11,L,lease,reservation)
    def zero_private(self,controller):
        # Owner polling is part of the real lifecycle: consume both terminal
        # queues before demanding that execution metadata has been reaped.
        controller.poll_completed();controller.poll_drained()
        stats=controller.stats()
        for name in ('active_leases','fill_reservations','writing_blocks','pinned_bytes','output_bytes','device_bytes','pending_executions','pending_copies'):
            self.assertEqual(stats[name],0,(name,stats))

    def test_hole_writing_and_missing_feature_never_cover(self):
        tokens=tuple(range(12));keys=self.cache.block_keys(self.ns,tokens)
        self.fill(keys[0],self.values(12));self.fill(keys[2],self.values(12))
        middle=self.cache.reserve_fill(keys[1],self.execution)
        lease=self.cache.lookup_and_pin(self.ns,tokens,12)
        self.assertEqual(lease.coverage_end,4);self.cache.release_lease(lease)
        with self.assertRaises(ValueError):self.cache.publish_ready(middle,{'aux':torch.zeros(4,2)})
        lease=self.cache.lookup_and_pin(self.ns,tokens,12)
        self.assertEqual(lease.coverage_end,4);self.cache.release_lease(lease)
        self.assertEqual(self.cache.stats()['writing_blocks'],0)

    def test_incomplete_query_chunk_namespace_and_history(self):
        keys=self.warm()
        for tokens,limit,want in ((self.tokens[:3],3,0),(self.tokens,7,4),(self.tokens,11,8)):
            lease=self.cache.lookup_and_pin(self.ns,tokens,limit);self.assertEqual(lease.coverage_end,want);self.cache.release_lease(lease)
        other=(99,98,97,96)+self.tokens[4:]
        self.assertNotEqual(keys[1].digest,self.cache.block_keys(self.ns,other)[1].digest)
        lease=self.cache.lookup_and_pin(self.ns+'-other',self.tokens,11)
        self.assertEqual(lease.coverage_end,0);self.cache.release_lease(lease)

    def test_shrink_stale_release_does_not_unpin_new_lease(self):
        self.warm();old=self.cache.lookup_and_pin(self.ns,self.tokens,11)
        lease=self.cache.shrink_lease(old,6);self.cache.release_lease(old);self.cache.release_lease(old)
        self.assertEqual(self.cache.stats()['active_leases'],1)
        for name,value in self.read(lease,6).items():torch.testing.assert_close(value,self.values()[name][:6],rtol=0,atol=0)
        self.cache.release_lease(lease);self.cache.release_lease(lease)
        self.assertEqual(self.cache.stats()['active_leases'],0)

    def test_single_owner_capacity_and_no_alias_after_publish(self):
        cache=PrefixFeatureCache(self.schema,chunk_tokens=4,capacity_bytes=self.block_bytes)
        key=cache.block_keys(self.ns,self.tokens)[0]
        fill=cache.reserve_fill(key,self.execution);self.assertIsNone(cache.reserve_fill(key,self.execution))
        forged=dataclasses.replace(fill,reservation_id='forged-owner')
        self.assertFalse(cache.publish_ready(forged,{k:v[:4].clone() for k,v in self.values().items()}));cache.abort_fill(forged)
        source={k:v[:4].clone() for k,v in self.values().items()}
        self.assertTrue(cache.publish_ready(fill,source))
        for value in source.values():value.fill_(-999)
        lease=cache.lookup_and_pin(self.ns,self.tokens,11)
        destination={spec.name:torch.empty(4,spec.width) for spec in self.schema.features};cache.copy_prefix_into(lease,4,destination)
        for name,value in destination.items():torch.testing.assert_close(value,self.values()[name][:4],rtol=0,atol=0)
        other_key=cache.block_keys(self.ns+'-pressure',self.tokens)[0]
        self.assertIsNone(cache.reserve_fill(other_key,self.execution));self.assertLessEqual(cache.stats()['cache_bytes'],self.block_bytes)
        cache.release_lease(lease);pressure=cache.reserve_fill(other_key,self.execution);self.assertIsNotNone(pressure)
        self.assertLessEqual(cache.stats()['cache_peak_bytes'],self.block_bytes)
        cache.abort_fill(pressure);self.assertEqual(cache.stats()['fill_reservations'],0)

    def test_nonchunk_aligned_seam_and_sink_owned_output(self):
        self.warm();controller=self.controller();plan=self.plan(controller,L=6)
        span=CapturedSpan(self.execution,6,11,{k:v[6:].clone() for k,v in self.values().items()},None)
        sample=controller.submit(plan,span).result(timeout=10)
        for name,value in sample.features.items():torch.testing.assert_close(value,self.values()[name],rtol=0,atol=0)
        self.assertGreater(controller.stats()['output_bytes'],0)
        drain=controller.invalidate(self.execution,'sink reader is still active')
        self.assertFalse(drain.done());self.assertFalse(drain.wait(.01))
        sample.release();sample.release();self.assertTrue(drain.wait(10));self.zero_private(controller)

    def test_cancel_does_not_treat_timeout_as_copy_drain(self):
        controller=self.controller();plan=self.plan(controller);gate=CompletionGate()
        future=controller.submit(plan,CapturedSpan(self.execution,0,11,self.values(),gate))
        self.assertTrue(gate.entered.wait(10),'worker did not reach real completion event')
        drain=controller.invalidate(self.execution,'injected cancellation before event')
        self.assertFalse(drain.done());self.assertFalse(drain.wait(.01));self.assertGreater(controller.stats()['output_bytes'],0)
        gate.complete.set()
        self.assertTrue(drain.wait(10))
        with self.assertRaises(Exception):future.result(timeout=10)
        self.zero_private(controller)

    def test_exact_independent_budgets_include_passthrough_and_no_oversize_fallback(self):
        controller=CaptureController(self.cache,self.schema,256,139,256);self.addCleanup(controller.shutdown,wait=True)
        reservation=controller.try_reserve_output(self.execution,11,0,7)
        self.assertEqual(controller.stats()['output_bytes'],139)
        other=dataclasses.replace(self.execution,request_id='other')
        self.assertIsNone(controller.try_reserve_output(other,1,0,0))
        controller.release_output(reservation);controller.release_output(reservation);self.zero_private(controller)
        with self.assertRaises(ValueError):controller.try_reserve_output(other,11,0,8)
        self.zero_private(controller)
        small=CaptureController(self.cache,self.schema,64,132,64);self.addCleanup(small.shutdown,wait=True)
        valid=small.try_reserve_output(self.execution,11,6,0);self.assertIsNotNone(valid);small.release_output(valid);self.zero_private(small)
        with self.assertRaises(ValueError):small.try_reserve_output(other,11,0,0)
        self.zero_private(small)

    def test_cache_too_small_skips_fill_without_losing_complete_delivery(self):
        cache=PrefixFeatureCache(self.schema,chunk_tokens=4,capacity_bytes=self.block_bytes-1)
        controller=CaptureController(cache,self.schema,4096,4096,4096);self.addCleanup(controller.shutdown,wait=True)
        lease=cache.lookup_and_pin(self.ns,self.tokens,11)
        reservation=controller.try_reserve_output(self.execution,11,0,0)
        plan=controller.freeze_plan(self.execution,self.ns,self.tokens,11,0,lease,reservation)
        sample=controller.submit(plan,CapturedSpan(self.execution,0,11,self.values(),None)).result(timeout=10)
        for name,value in sample.features.items():torch.testing.assert_close(value,self.values()[name],rtol=0,atol=0)
        sample.release();controller.poll_completed();controller.poll_drained()
        self.assertEqual(cache.stats()['ready_blocks'],0);self.assertEqual(cache.stats()['cache_bytes'],0)
        self.assertGreater(cache.stats()['fill_skips'],0);self.zero_private(controller)

    def test_same_key_concurrent_fill_has_one_committer(self):
        key=self.cache.block_keys(self.ns,self.tokens)[0];start=threading.Barrier(3);finish=threading.Barrier(3);fills=[];errors=[]
        def reserve(index):
            try:
                start.wait(timeout=5)
                fills.append(self.cache.reserve_fill(key,dataclasses.replace(self.execution,request_id=f'writer-{index}')))
                finish.wait(timeout=5)
            except BaseException as exc:errors.append(exc)
        workers=[threading.Thread(target=reserve,args=(i,)) for i in range(2)]
        for worker in workers:worker.start()
        start.wait(timeout=5);finish.wait(timeout=5)
        for worker in workers:worker.join(timeout=5);self.assertFalse(worker.is_alive())
        self.assertEqual(errors,[]);owners=[fill for fill in fills if fill is not None];self.assertEqual(len(owners),1)
        self.assertTrue(self.cache.publish_ready(owners[0],{k:v[:4].clone() for k,v in self.values().items()}))
        self.cache.abort_fill(owners[0]);self.assertEqual(self.cache.stats()['ready_blocks'],1)
        self.assertEqual(self.cache.stats()['fill_reservations'],0)

    def test_stale_epoch_or_short_span_rejected_before_materialization(self):
        for mode in ('epoch','start','end','feature'):
            controller=self.controller();execution=ExecutionIdentity('server-a',mode,mode,0);plan=self.plan(controller,execution=execution)
            features=self.values();span_execution=execution;start,end=0,11
            if mode=='epoch':span_execution=dataclasses.replace(execution,execution_epoch=1)
            if mode=='start':start=1
            if mode=='end':end=10
            if mode=='feature':features.pop('last_hidden')
            with self.subTest(mode=mode),self.assertRaises(ValueError):controller.submit(plan,CapturedSpan(span_execution,start,end,features,None))
            drain=controller.invalidate(execution,'rejected span cleanup');self.assertTrue(drain.wait(10));self.zero_private(controller)


if __name__=='__main__':unittest.main()
