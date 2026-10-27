#!/usr/bin/env python3
"""Real CUDA Qwen3 feature D2H/assembly with strict physical pinned accounting.

No target model, trainer, server, or port is created. Run only on root-assigned
GPU0 in a fresh process with the exact committed candidate PYTHONPATH.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import weakref

from validation_helpers import append_json,tensor_sha256,write_json


def pattern(rows,width,offset,device):
    import torch
    row=torch.arange(rows,dtype=torch.int32,device=device).view(-1,1)
    channel=torch.arange(width,dtype=torch.int32,device=device).view(1,-1)
    return (((row*3+channel*5+offset)%64).float()-32).div_(8).to(torch.bfloat16).contiguous()


def make_sources(schema,N,stream):
    import torch
    with torch.cuda.stream(stream):
        features={spec.name:pattern(N,spec.width,index*11,'cuda') for index,spec in enumerate(schema.features)}
        # Real producer work delays readiness. No event wrapper or fake CUDA API
        # replaces actual event/stream ordering in this test.
        torch.cuda._sleep(500_000_000)
        ready=torch.cuda.Event();ready.record(stream)
    return features,ready


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--sglang-source',required=True);parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--repeats',type=int,default=2);args=parser.parse_args()
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    result={'passed':False,'scope':'real CUDA storage/event/physical pinned bound only; no model/forward/training acceptance'}
    controller=None
    try:
        import torch
        import sglang.srt.spec_feature_cache as runtime
        from sglang.srt.spec_feature_cache import (CaptureController,CapturedSpan,ExecutionIdentity,FeatureSchema,FeatureSpec,PrefixFeatureCache)
        source=Path(args.sglang_source).resolve()
        commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        status=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
        if commit!=args.expected_commit or status:raise AssertionError(f'candidate is not frozen expected commit: {commit}, dirty={status}')
        if not Path(runtime.__file__).resolve().is_relative_to(source/'python'):raise AssertionError(f'wrong runtime import {runtime.__file__}')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.device_count()!=1:raise AssertionError('this allocation is exclusively physical GPU0')
        torch.set_num_threads(4);torch.cuda.init();torch.cuda.synchronize()
        schema=FeatureSchema((FeatureSpec('aux',12800,'bfloat16',(1,9,17,25,33)),FeatureSpec('last_hidden',2560,'bfloat16',())))
        pinned_budget=64<<20;output_budget=128<<20;device_budget=64<<20
        cache=PrefixFeatureCache(schema,chunk_tokens=256,capacity_bytes=0)
        controller=CaptureController(cache,schema,pinned_budget,output_budget,device_budget)
        baseline=torch.cuda.memory.host_memory_stats()
        if 'allocated_bytes.current' not in baseline:raise AssertionError('actual PyTorch host allocator counters unavailable')
        manifest={'python':sys.executable,'runtime_import':str(runtime.__file__),'source_commit':commit,'source_status':status,
                  'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                  'schema':[{'name':s.name,'width':s.width,'dtype':s.dtype,'layer_ids':s.layer_ids} for s in schema.features],
                  'budgets':{'pinned':pinned_budget,'output':output_budget,'device':device_budget,'cache':0},'host_before':baseline}
        write_json(root/'manifest.json',manifest)
        stream=torch.cuda.Stream();pointers=[];maximum_delta=0;records=[]
        sizes=[2048,1024,512,256,2048]*args.repeats
        for index,N in enumerate(sizes):
            identity=ExecutionIdentity('cuda-storage-epoch',f'request-{index}',f'attempt-{index}',0)
            tokens=tuple(range(N));lease=cache.lookup_and_pin('cuda-storage-schema',tokens,N)
            reservation=controller.try_reserve_output(identity,N,0,N*16)
            if reservation is None:raise AssertionError('serial fitting request unexpectedly blocked')
            plan=controller.freeze_plan(identity,'cuda-storage-schema',tokens,N,0,lease,reservation)
            features,ready=make_sources(schema,N,stream)
            refs={name:weakref.ref(value) for name,value in features.items()};event_ref=weakref.ref(ready)
            span=CapturedSpan(identity,0,N,features,ready);future=controller.submit(plan,span)
            producer_pending=not ready.query()
            del features,span
            if producer_pending and any(ref() is None for ref in refs.values()):raise AssertionError('source owner dropped before producer event completed')
            del ready
            sample=future.result(timeout=60)
            hashes={}
            for feature_index,spec in enumerate(schema.features):
                actual=sample.features[spec.name];expected=pattern(N,spec.width,feature_index*11,'cpu')
                if actual.is_pinned():raise AssertionError('durable sample output unexpectedly pins the cache/sample')
                if not torch.equal(actual,expected):raise AssertionError(f'full D2H/assembly values differ for N={N} {spec.name}')
                hashes[spec.name]=tensor_sha256(actual)
                del actual,expected
            active=controller.stats()
            if active['output_bytes']<=0:raise AssertionError('sink-owned output budget returned before sample.release')
            arena=controller._pinned_arena
            if arena is None or not arena.is_pinned():raise AssertionError('actual pinned arena not allocated')
            pointers.append(arena.untyped_storage().data_ptr())
            sample.release();del sample,arena
            controller.poll_completed();controller.poll_drained();gc.collect();torch.cuda.synchronize()
            stats=controller.stats();host=torch.cuda.memory.host_memory_stats();delta=host['allocated_bytes.current']-baseline['allocated_bytes.current']
            maximum_delta=max(maximum_delta,delta)
            if delta>pinned_budget:raise AssertionError(f'physical host pinned allocator exceeds64MiB: delta={delta}, counters={host}')
            for name in ('pinned_bytes','output_bytes','device_bytes','active_leases','fill_reservations','writing_blocks','pending_executions','pending_copies','output_reservations','quarantined_executions'):
                if stats[name]!=0:raise AssertionError(f'reservation/resource leak after explicit drain: {name}={stats[name]}')
            if stats['pinned_arena_bytes']!=pinned_budget or stats['pinned_capacity_bytes']!=pinned_budget:raise AssertionError(f'retained arena not fully accounted: {stats}')
            if any(ref() is not None for ref in refs.values()) or event_ref() is not None:raise AssertionError('CUDA source/event Python owners retained after completed copy drain')
            record={'index':index,'N':N,'producer_pending_at_submit':producer_pending,'hashes':hashes,'controller':stats,'host':host,'host_allocated_delta':delta,'passed':True}
            append_json(root/'sizes.jsonl',record);records.append(record)
            print(f'CUDA_STORAGE N={N} physical_pinned_delta={delta} retained_arena={stats["pinned_arena_bytes"]} private_reserved=0',flush=True)
        if len(set(pointers))!=1:raise AssertionError('different input sizes allocated different pinned arenas')
        # Actual cancellation waits for the real delayed CUDA producer event;
        # no copy/source can be reclaimed by treating a timeout as a drain.
        N=2048;identity=ExecutionIdentity('cuda-storage-epoch','cancelled','cancelled',1);tokens=tuple(range(N))
        lease=cache.lookup_and_pin('cuda-storage-schema',tokens,N);reservation=controller.try_reserve_output(identity,N,0,N*16)
        plan=controller.freeze_plan(identity,'cuda-storage-schema',tokens,N,0,lease,reservation)
        features,ready=make_sources(schema,N,stream);refs=[weakref.ref(value) for value in features.values()]
        future=controller.submit(plan,CapturedSpan(identity,0,N,features,ready));pending=not ready.query()
        drain=controller.invalidate(identity,'actual CUDA cancellation before producer ready')
        del features,ready
        if pending and drain.wait(.001):raise AssertionError('cancellation treated pending CUDA event as drained')
        if not drain.wait(60):raise AssertionError('actual CUDA cancellation did not drain')
        try:future.result(timeout=60)
        except Exception:pass
        else:raise AssertionError('cancelled execution produced a deliverable sample')
        controller.poll_completed();controller.poll_drained();gc.collect();torch.cuda.synchronize()
        final=controller.stats();host=torch.cuda.memory.host_memory_stats();delta=host['allocated_bytes.current']-baseline['allocated_bytes.current']
        if delta>pinned_budget or any(ref() is not None for ref in refs):raise AssertionError('cancelled source/physical pinned allocation leaked')
        for name in ('pinned_bytes','output_bytes','device_bytes','active_leases','fill_reservations','pending_executions','pending_copies'):
            if final[name]!=0:raise AssertionError(f'cancel leak {name}={final[name]}')
        result.update(passed=True,sizes=len(records),sequence=sizes,arena_reused=True,host_max_allocated_delta=maximum_delta,
                      retained_arena_bytes=final['pinned_arena_bytes'],cancel_producer_pending=pending,final=final,host_after=host)
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}';(root/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        if controller is not None:
            try:controller.shutdown(wait=True)
            except BaseException as exc:result['passed']=False;result['shutdown_error']=f'{type(exc).__name__}: {exc}'
        result['ended_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(root/'result.json',result);print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
