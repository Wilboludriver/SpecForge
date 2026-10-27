#!/usr/bin/env python3
"""Two real CPU Gloo ranks invoke actual bridge control collectives/phase.

This checks collective ordering and None payloads, not GPU NCCL or a model.
The watchdog terminates only the two process objects owned by this probe.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import traceback


def worker(rank,directory):
    import torch.distributed as dist
    from sglang.srt.spec_feature_bridge import SpecFeatureBridge
    root=Path(directory);trace=[]
    try:
        dist.init_process_group('gloo',init_method=f'file://{root}/rendezvous',rank=rank,world_size=2,timeout=timedelta(seconds=20))
        bridge=SpecFeatureBridge.__new__(SpecFeatureBridge);bridge.writer=rank==0
        bridge.s=SimpleNamespace(ps=SimpleNamespace(tp_size=2,tp_rank=rank),tp_group=SimpleNamespace(ranks=[0,1]),tp_cpu_group=dist.group.WORLD,waiting_queue=[])
        for seq,payload in enumerate((None,{'candidate':'r1','H':768},'admission-refusal',{'error':'rank1-match-error'},'after-rollback')):
            got=bridge.broadcast(payload if rank==0 else None)
            if got!=payload:raise AssertionError((rank,seq,got,payload))
            votes=bridge.gather({'rank':rank,'seq':seq,'got':got})
            if [v['rank'] for v in votes]!=[0,1] or any(v['seq']!=seq for v in votes):raise AssertionError(votes)
            trace.append({'seq':seq,'broadcast':got,'votes':votes})
        bridge.phase();trace.append({'phase':'empty','passed':True})
        bridge.s.waiting_queue=[SimpleNamespace(rid='r1')];bridge.phase();trace.append({'phase':'matching-queue','passed':True})
        bridge.s.waiting_queue=[SimpleNamespace(rid=f'rank-{rank}-different')]
        try:bridge.phase()
        except RuntimeError:trace.append({'phase':'queue-disagreement','rejected':True})
        else:raise AssertionError('queue disagreement not rejected')
        if bridge.gather('after-common-failure')!=['after-common-failure']*2:raise AssertionError('post-error collective mismatch')
        (root/f'rank{rank}.json').write_text(json.dumps({'passed':True,'trace':trace},indent=2))
    except BaseException:
        (root/f'rank{rank}.json').write_text(json.dumps({'passed':False,'trace':trace,'error':traceback.format_exc()},indent=2));raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    import sglang.srt.spec_feature_bridge as module
    source=Path(module.__file__)
    manifest={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'actual bridge broadcast/gather/phase; CPU Gloo only'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    context=multiprocessing.get_context('spawn');processes=[context.Process(target=worker,args=(rank,str(root))) for rank in range(2)]
    for process in processes:process.start()
    for process in processes:process.join(timeout=35)
    for process in processes:
        if process.is_alive():process.terminate();process.join(timeout=5)
        if process.is_alive():process.kill();process.join(timeout=5)
    result={'exitcodes':[p.exitcode for p in processes],'passed':all(p.exitcode==0 for p in processes),'source_unchanged':manifest['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()}
    result['passed']=result['passed'] and result['source_unchanged']
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
