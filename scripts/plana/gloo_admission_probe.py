#!/usr/bin/env python3
"""Actual bridge.admit on two CPU Gloo ranks with real A cache/controller.

Engine tensors/adder/pools are small deterministic fixtures. This tests real
rank votes, refusal/exception/rematch and ownership rollback; hybrid GPU state
restoration and NCCL remain separate acceptance gates.
"""
import argparse
from collections import deque
from datetime import timedelta
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
import traceback


class Pool:
    def __init__(self):self.freed=[]
    def free(self,value):self.freed.append(value.detach().tolist())


class Receipt:
    def to_dec_params(self):return {'exact_lock_receipt':1}


class Tree:
    def __init__(self):
        self.pool=Pool();self.released=[]
        self.req_to_token_pool=SimpleNamespace(mamba_allocator=self.pool)
    def dec_lock_ref(self,node,params=None):self.released.append((node,params))


def setup_runtime(runtime_python):
    import sglang.srt
    if runtime_python:sglang.srt.__path__.append(str(Path(runtime_python)/'sglang/srt'))


def worker(rank,directory,runtime_python):
    import torch
    import torch.distributed as dist
    setup_runtime(runtime_python)
    from sglang.srt.spec_feature_bridge import AttemptRegistry,SpecFeatureBridge
    from sglang.srt.spec_feature_cache import CaptureController,ExecutionIdentity,FeatureSchema,FeatureSpec,PrefixFeatureCache
    from sglang.srt.managers.schedule_policy import AddReqResult
    root=Path(directory);records=[]
    try:
        dist.init_process_group('gloo',init_method=f'file://{root}/rendezvous',rank=rank,world_size=2,timeout=timedelta(seconds=20))
        for mode in ('revoked-rank1','no-membership-rank1','after-append-error-rank1','match-error-rank1','L-disagreement','output-pressure','unanimous-success'):
            bridge=SpecFeatureBridge.__new__(SpecFeatureBridge);tree=Tree()
            bridge.s=SimpleNamespace(ps=SimpleNamespace(tp_size=2,tp_rank=rank),tp_group=SimpleNamespace(ranks=[0,1]),tp_cpu_group=dist.group.WORLD,waiting_queue=[],tree_cache=tree)
            bridge.writer=rank==0;bridge.enabled=True;bridge.epoch='gloo-epoch';bridge.namespace='gloo-namespace';bridge.signature='gloo-signature'
            bridge.registry=AttemptRegistry(bridge.epoch);bridge.jobs={};bridge.batches=[];bridge.history=deque(maxlen=16);bridge.cache=bridge.controller=None
            schema=FeatureSchema((FeatureSpec('aux',2,'float32',(1,)),FeatureSpec('last_hidden',1,'float32',())))
            if rank==0:
                bridge.cache=PrefixFeatureCache(schema,chunk_tokens=4,capacity_bytes=4096)
                bridge.controller=CaptureController(bridge.cache,schema,4096,96 if mode=='output-pressure' else 4096,4096)
                key=bridge.cache.block_keys(bridge.namespace,tuple(range(8)))[0]
                fill=bridge.cache.reserve_fill(key,ExecutionIdentity(bridge.epoch,'warm','warm',0))
                bridge.cache.publish_ready(fill,{'aux':torch.zeros(4,2),'last_hidden':torch.zeros(4,1)})
            blocker=None
            if rank==0 and mode=='output-pressure':
                blocker=bridge.controller.try_reserve_output(ExecutionIdentity(bridge.epoch,'blocker','blocker',0),8,0,0)
                if blocker is None:raise AssertionError('pressure fixture did not reserve initial output')
            entry=bridge.registry.begin({'store_id':'store','sample_id':mode,'gen':1,'attempt_id':mode,'request_digest':'fixture'})
            if mode=='revoked-rank1' and rank==1:entry.revoked=True
            req=SimpleNamespace(rid=mode,origin_input_ids=list(range(8)),spec_capture={'passthrough':[]},spec_feature_attempt=entry,spec_feature_execution_epoch=0,
                                prefix_indices=torch.empty(0,dtype=torch.int64),mamba_pool_idx=None,mamba_cow_src_index=None,mamba_needs_clear=False,
                                extend_range='before',skip_lock_node_ids=set(),swa_uuid_for_lock=None)
            before={'mamba_pool_idx':None,'mamba_cow_src_index':None,'mamba_needs_clear':False,'prefix_indices':req.prefix_indices}
            matches=[]
            def match(_tree):
                cap=req.spec_feature_limit;matches.append(cap);req.mamba_pool_idx=torch.tensor(rank+7);req.mamba_cow_src_index=5
                if mode=='match-error-rank1' and rank==1 and len(matches)==1:raise RuntimeError('injected rank-local match failure')
                L=0 if (mode=='L-disagreement' and rank==1 and len(matches)==1) else min(cap,4)
                req.prefix_indices=torch.arange(L,dtype=torch.int64)
            req.init_next_round_input=match
            adder=SimpleNamespace(tree_cache=tree,can_run_list=[],preempt_list=[],rem_total_tokens=100,rem_input_tokens=100)
            def add(*a,**kw):
                adder.rem_total_tokens-=8;adder.rem_input_tokens-=8;req.extend_range='after'
                if mode=='no-membership-rank1' and rank==1:return AddReqResult.CONTINUE
                adder.can_run_list.append(req);adder._spec_feature_journal.record_lock('node',Receipt())
                if mode=='after-append-error-rank1' and rank==1:raise RuntimeError('injected rank-local after-append failure')
                return AddReqResult.CONTINUE
            adder.add_one_req=add
            try:
                result=bridge.admit(req,adder,False,1);outcome=result.name
            except RuntimeError as exc:
                if mode!='after-append-error-rank1':raise
                outcome='ERROR'
            expected={'revoked-rank1':'OTHER','no-membership-rank1':'OTHER','after-append-error-rank1':'ERROR','output-pressure':'OTHER'}.get(mode,'CONTINUE')
            if outcome!=expected:raise AssertionError((mode,rank,outcome,expected))
            outcomes=bridge.gather(outcome)
            if outcomes!=[expected,expected]:raise AssertionError(('rank decisions differ',outcomes))
            committed=expected=='CONTINUE'
            if committed:
                if len(adder.can_run_list)!=1 or adder.rem_total_tokens!=92:raise AssertionError('unanimous admission did not commit')
                if mode in ('L-disagreement','match-error-rank1') and matches!=[4,0]:raise AssertionError(('no safe common rematch',matches))
                if rank==0:
                    plan=req.spec_feature_plan
                    if plan.L!=(0 if mode in ('L-disagreement','match-error-rank1') else 4):raise AssertionError(('wrong final engine L',plan.L))
                    if not bridge.controller.invalidate(plan.execution,'test sink never submitted').wait(5):raise AssertionError('committed output failed to drain')
                bridge._rollback_match(req,before)
            else:
                if adder.can_run_list or adder.rem_total_tokens!=100 or adder.rem_input_tokens!=100 or req.extend_range!='before':raise AssertionError('adder transaction did not roll back')
                if req.mamba_pool_idx is not None:raise AssertionError('fresh COW slot still owned after refusal')
            expected_frees=0 if mode=='revoked-rank1' else (2 if mode in ('L-disagreement','match-error-rank1') else 1)
            if len(tree.pool.freed)!=expected_frees:raise AssertionError(('COW free count',mode,rank,tree.pool.freed))
            if not committed and mode=='after-append-error-rank1' and tree.released!=[('node',{'exact_lock_receipt':1})]:raise AssertionError('exact lock receipt not released once')
            if rank==0:
                if blocker is not None:bridge.controller.release_output(blocker)
                bridge.controller.poll_completed();bridge.controller.poll_drained();stats=bridge.controller.stats()
                for name in ('active_leases','output_bytes','pinned_bytes','device_bytes','pending_executions','pending_copies'):
                    if stats[name]!=0:raise AssertionError((mode,name,stats))
                bridge.controller.shutdown(wait=True)
            record={'mode':mode,'rank':rank,'outcome':outcome,'matches':matches,'cow_freed':tree.pool.freed,'exact_lock_releases':tree.released,'passed':True}
            records.append(record)
            (root/f'rank{rank}.json').write_text(json.dumps({'passed':True,'records':records},indent=2))
            bridge.gather(f'{mode}-all-clean')
    except BaseException:
        (root/f'rank{rank}.json').write_text(json.dumps({'passed':False,'records':records,'error':traceback.format_exc()},indent=2));raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True);parser.add_argument('--runtime-python');args=parser.parse_args()
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=False);setup_runtime(args.runtime_python)
    import sglang.srt.spec_feature_bridge as bridge
    import sglang.srt.spec_feature_cache as runtime
    sources=[Path(bridge.__file__),*(Path(runtime.__file__).parent/name for name in ('__init__.py','protocol.py','cache.py','controller.py'))]
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (root/'manifest.json').write_text(json.dumps({'source_sha256':hashes,'scope':'actual bridge.admit + A runtime on two CPU Gloo ranks; fake engine resources'},indent=2))
    context=multiprocessing.get_context('spawn');processes=[context.Process(target=worker,args=(rank,str(root),args.runtime_python)) for rank in range(2)]
    for process in processes:process.start()
    for process in processes:process.join(timeout=45)
    for process in processes:
        if process.is_alive():process.terminate();process.join(timeout=5)
        if process.is_alive():process.kill();process.join(timeout=5)
    unchanged=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==value for path,value in hashes.items())
    result={'passed':all(p.exitcode==0 for p in processes) and unchanged,'exitcodes':[p.exitcode for p in processes],'source_unchanged':unchanged}
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
