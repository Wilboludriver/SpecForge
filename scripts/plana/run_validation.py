#!/usr/bin/env python3
"""Actual selected-model capture/get/release and pretrained DFlash replay.

Run only under root-assigned resources. Starts/stops only service processes
created from the supported managed-local plan. Production source is read-only.
The default performs two full sequential 16-case cache-off capture passes.
--baseline-dir compares a new candidate pass against a completed baseline.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from unittest import mock

from extended_checks import (checkpoint_subset_audit, compare_metric_trees, extract_capture_result, feature_regions, validate_diagnostics)
from validation_helpers import (append_json, canonical_model_info, file_sha256, load_cases,
                                tensor_errors, tensor_sha256, validate_features, write_json)


def git_identity(path):
    def command(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    return {"root": command("rev-parse", "--show-toplevel"), "commit": command("rev-parse", "HEAD"),
            "branch": command("branch", "--show-current"), "status": command("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(path), "diff", "HEAD"])).hexdigest()}


def runtime_manifest(args):
    imports = {}
    for name in ("specforge", "sglang", "sglang.srt.spec_capture_sink", "torch", "transformers", "mooncake.store"):
        module = importlib.import_module(name)
        imports[name] = str(getattr(module, "__file__", None))
    for name, root in (("specforge", args.specforge_source), ("sglang.srt.spec_capture_sink", Path(args.sglang_source)/"python")):
        if not Path(imports[name]).resolve().is_relative_to(Path(root).resolve()):
            raise AssertionError(f"wrong runtime import {name}: {imports[name]}")
    return {"imports": imports, "specforge": git_identity(args.specforge_source),
            "sglang": git_identity(args.sglang_source), "harness": git_identity(Path(__file__).resolve().parents[2]),
            "harness_file_sha256": file_sha256(__file__), "python": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": sys.argv,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "transport": "mooncake-tcp-host"}


def isolated_config(args, run_dir):
    import yaml
    original = yaml.safe_load(Path(args.config).read_text())
    original['run_id'] = run_dir.name
    original['output_dir'] = str(run_dir/'output')
    original['data']['cache_dir'] = str(run_dir/'data-cache')
    disagg = original['deployment']['disaggregated']
    disagg['control_dir'], disagg['consumer_state_dir'] = str(run_dir/'control'), str(run_dir/'consumer-state')
    # Resource identities come from the assigned input config; do not invent
    # unknown cache args or silently change the target, TP or backend.
    path = run_dir/'config.yaml'
    path.write_text(yaml.safe_dump(original, sort_keys=False))
    return path


def start_services(plan, run_dir):
    from specforge.launch_plan import _managed_preflight, _spawn_service, _wait_for_service, _stop_services
    _managed_preflight(plan)
    Path(plan.managed_root, 'logs').mkdir(parents=True, exist_ok=False)
    started = []
    try:
        for service in sorted(plan.services, key=lambda s: s.phase):
            process = _spawn_service(service, popen=subprocess.Popen)
            started.append((service, process))
            append_json(run_dir/'service-events.jsonl', {"action": "spawn", "label": service.command.label, "pid": process.pid, "argv": list(service.command.argv)})
            _wait_for_service(service, process, tuple(started))
            append_json(run_dir/'service-events.jsonl', {"action": "ready", "label": service.command.label, "pid": process.pid})
        return started
    except BaseException:
        _stop_services(started, grace_s=30)
        raise


def cleanup_store(store, expected_keys, run_dir, require_fences=False):
    store.discard_external_attempts(reason='plana-validation-finally')
    store.drain_pending_removals(max_attempts=30, retry_interval_s=.1)
    health = store.health()
    remaining = [key for key in expected_keys if store._store_exists(key)]
    write_json(run_dir/'store-final.json', {"health": health, "expected_keys": len(expected_keys), "remaining_keys": remaining})
    fields = ('resident_samples', 'provisional_external', 'active_leases', 'resident_bytes', 'release_pending', 'quarantined_buffers', 'quarantined_bytes')
    fence_fields = ('capture_pending_begin','capture_fences','capture_ack_pending','capture_handle_attempts','capture_read_intents','capture_active_reads','capture_failed_reads')
    if require_fences and any(name not in health for name in fence_fields):
        raise AssertionError('negotiated store omitted attempt/read resource counters')
    fields += tuple(name for name in fence_fields if name in health)
    if remaining or any(health[name] != 0 for name in fields):
        raise AssertionError(f"store not drained: remaining={remaining}, health={health}")
    store.close()



def server_drained(adapter,run_dir,require_cache):
    protocol=getattr(adapter,'_capture_protocol',None)
    if protocol is None:
        if require_cache:raise AssertionError('cache candidate did not negotiate control protocol')
        return
    capability=protocol.capability()
    if capability is None:
        if require_cache:raise AssertionError('cache candidate fell back to unknown capability')
        return
    if require_cache and not capability['cache_enabled']:
        raise AssertionError('cache-on candidate capability reports cache disabled')
    stats=protocol.call({'action':'stats','server_epoch':capability['server_epoch']})
    write_json(run_dir/'server-final.json',stats)
    for name in ('active_attempts','active_readers','pending_deliveries'):
        if stats[name]!=0:raise AssertionError(f'server retained {name}: {stats}')
    if capability['cache_enabled']:
        limits=capability['limits'];cache=stats['cache'];controller=stats['controller']
        for name in ('active_leases','fill_reservations','writing_blocks'):
            if cache[name]!=0:raise AssertionError(f'cache retained {name}: {cache}')
        for name in ('pinned_bytes','output_bytes','device_bytes','output_reservations','pending_executions','pending_copies','quarantined_executions'):
            if controller[name]!=0:raise AssertionError(f'controller retained {name}: {controller}')
        for value,limit in ((cache['cache_peak_bytes'],limits['cache_bytes']),
                            (controller['pinned_peak_bytes'],limits['pinned_capacity_bytes']),
                            (controller['pinned_arena_bytes'],limits['pinned_capacity_bytes']),
                            (controller['output_peak_bytes'],limits['output_bytes']),
                            (controller['device_peak_bytes'],limits['device_bytes'])):
            if value>limit:raise AssertionError(f'configured hard byte limit exceeded: {value}>{limit}')


def capture_pass(args, cfg, resolved, plan, run_dir, pass_name, rows):
    import torch
    from specforge.algorithms.common.providers import resolve_server_capture_layout
    from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter, ServerCaptureSchema
    from specforge.inference.capture import CaptureConfig
    from specforge.runtime.contracts import PromptTask, SampleRef
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
    from specforge.training.model_loading import resolve_draft_config
    from specforge.modeling.target.target_utils import load_target_config, target_hidden_size
    draft_config = resolve_draft_config(cfg, provider=resolved.algorithm.providers.model.draft_config)
    target_config = load_target_config(cfg.model.target_model_path, trust_remote_code=cfg.model.trust_remote_code)
    h = target_hidden_size(target_config)
    layers = tuple(resolved.algorithm.providers.model.resolve_capture_layers(cfg, draft_config, target_config))
    layout = resolve_server_capture_layout(resolved.algorithm, cfg, modality='text')
    if layout.last_hidden_feature is None:
        raise AssertionError('teacher metrics must require last_hidden')
    schema = ServerCaptureSchema(aux_feature=layout.aux_feature, last_hidden_feature=layout.last_hidden_feature,
                                 passthrough=layout.passthrough, attention_mask_feature=layout.attention_mask_feature)
    server = next(s for s in plan.services if s.command.label.startswith('capture-server-'))
    env = server.command.env
    setup = {"local_hostname": env['MOONCAKE_LOCAL_HOSTNAME'], "metadata_server": env['MOONCAKE_METADATA_SERVER'],
             "global_segment_size": int(env['MOONCAKE_GLOBAL_SEGMENT_SIZE']), "local_buffer_size": int(env['MOONCAKE_LOCAL_BUFFER_SIZE']),
             "protocol": env['MOONCAKE_PROTOCOL'], "rdma_devices": "", "master_server_addr": env['MOONCAKE_MASTER_SERVER_ADDR']}
    if setup['protocol'] != 'tcp' or env['SGLANG_SPEC_CAPTURE_GPU_PUT'] != '0':
        raise AssertionError('this baseline harness is explicitly TCP host publication')
    namespace = f'{run_dir.name}-{pass_name}'
    store = MooncakeFeatureStore(store_id=namespace, setup_kwargs=setup, receive_buffers='pinned', receive_pool_bytes=512<<20, max_resident_bytes=1<<30)
    captures = []
    from specforge.inference.adapters.server_capture import _SessionPost
    post = _SessionPost()
    def record_post(url, *, json_body, timeout):
        result = post(url, json_body=json_body, timeout=timeout)
        append_json(run_dir/'http-responses.jsonl', {"pass": pass_name, "url": url, "body": json_body, "response": result})
        if url.endswith('/generate'):
            captures.append(result)
        return result
    adapter_options = {}
    if 'control_fn' in inspect.signature(SGLangServerCaptureAdapter).parameters:
        adapter_options['control_fn'] = record_post
    elif args.require_cache:
        raise AssertionError('cache candidate lacks negotiated control_fn interface')
    adapter = SGLangServerCaptureAdapter(env['DISAGG_SERVER_URLS'], store, run_id=namespace, algorithm='dflash', schema=schema, server_capture='patch', post_fn=record_post, **adapter_options)
    contract = CaptureConfig.from_strategy(required_features={'input_ids','loss_mask','hidden_states','target_last_hidden_states'},
                                           aux_hidden_state_layer_ids=layers, target_repr='hidden_state', target_hidden_size=h)
    directory = run_dir/pass_name
    directory.mkdir()
    expected_keys = []
    diagnostics = []
    try:
        for index, row in enumerate(rows):
            task = PromptTask(task_id=f'{index:02d}-{row["id"]}', run_id=namespace, source_id='plana-native16',
                              payload={key: row[key] for key in ('input_ids','loss_mask')}, max_length=len(row['input_ids']))
            started = time.perf_counter()
            ref, = adapter.produce_refs([task], capture=contract)
            if not isinstance(ref, SampleRef):
                raise AssertionError(f'{row["id"]}: capture failed: {ref}')
            raw_result = extract_capture_result(captures[-1], ref.sample_id)
            diagnostic = validate_diagnostics(raw_result, len(row['input_ids']), require_cache=args.require_cache)
            diagnostics.append(diagnostic)
            keys = [store._tkey(ref.sample_id, int(ref.metadata['generation']), name) for name in ref.feature_specs]
            expected_keys.extend(keys)
            handle = None
            try:
                features, handle = store.get(ref)
                validate_features(features, row, hidden_size=h, layers=layers)
                # Persist actual fetched tensors before release, not specs alone.
                torch.save({k:v.detach().cpu().clone() for k,v in features.items()}, directory/f'{index:02d}-{row["id"]}.pt')
                hashes = {k:tensor_sha256(v) for k,v in features.items()}
                if ref.num_tokens != len(row['input_ids']):
                    raise AssertionError('SampleRef token count mismatch')
            finally:
                if handle is not None:
                    store.release(handle)
                else:
                    store.abort(ref.sample_id, reason='plana-fetch-failure')
            store.drain_sample_removals([ref.sample_id], max_attempts=30, retry_interval_s=.1)
            remaining = [k for k in keys if store._store_exists(k)]
            if remaining:
                raise AssertionError(f'{row["id"]}: keys remain after explicit release drain: {remaining}')
            result = {"pass": pass_name, "index": index, "case": row['id'], "N": len(row['input_ids']),
                      "seconds": time.perf_counter()-started, "hashes": hashes, "keys_removed": len(keys), "ref": dataclasses.asdict(ref), "health": store.health(), "diagnostics": diagnostic}
            append_json(run_dir/'case-results.jsonl', result)
            print(f'CAPTURE {pass_name} {index+1:02d}/16 {row["id"]} N={len(row["input_ids"])} keys_removed={len(keys)}', flush=True)
    finally:
        cleanup_store(store, expected_keys, directory, require_fences=args.require_cache)
        server_drained(adapter,directory,args.require_cache)
    sequences=[d['issue_sequence'] for d in diagnostics if 'issue_sequence' in d]
    if sequences and (len(sequences)!=len(rows) or any(b<=a for a,b in zip(sequences,sequences[1:]))):
        raise AssertionError('server-issued attempt sequence is missing or nonmonotonic')
    identities={(d['server_epoch'],d['capture_signature'],d['capture_namespace']) for d in diagnostics if 'server_epoch' in d}
    if len(identities)>1:raise AssertionError('server capture identity changed during one controlled pass')
    if args.require_cache and not any(d['L'] > 0 for d in diagnostics):
        raise AssertionError('cache-on candidate performed no actual prefix reuse')
    write_json(directory/'diagnostics.json', diagnostics)
    info=canonical_model_info({"hidden_size": h, "layers": list(layers), "draft_config": draft_config.to_dict()})
    write_json(directory/'model-info.json',info)
    return [directory/f'{i:02d}-{row["id"]}.pt' for i,row in enumerate(rows)], info, diagnostics


def compare_features(args, rows, reference_paths, actual_paths, model_info, run_dir, diagnostics):
    import torch
    results = []
    for index,(row,reference_path,actual_path) in enumerate(zip(rows,reference_paths,actual_paths,strict=True)):
        reference, actual = [torch.load(p, map_location='cpu', weights_only=True) for p in (reference_path,actual_path)]
        validate_features(actual,row,hidden_size=model_info['hidden_size'],layers=model_info['layers'])
        for name in ('input_ids','loss_mask'):
            if not torch.equal(actual[name],reference[name]):
                raise AssertionError(f'{row["id"]}: integer feature changed: {name}')
        region_checks = feature_regions(actual,reference,L=diagnostics[index]['L'],layers=model_info['layers'],hidden_size=model_info['hidden_size'],rtol=args.feature_rtol,atol=args.feature_atol)
        for name in ('hidden_states','target_last_hidden_states'):
            region_checks.append({'feature':name,'region':'all',**tensor_errors(actual[name],reference[name],rtol=args.feature_rtol,atol=args.feature_atol)})
        for check in region_checks:
            record={"case":row['id'],"L":diagnostics[index]['L'],**check}
            append_json(run_dir/'feature-errors.jsonl',record);results.append(record)
    if not all(record['passed'] for record in results):
        raise AssertionError('feature numerical comparison failed; see feature-errors.jsonl')
    return {"comparisons":len(results),"max_abs":max(v['max_abs'] for v in results),"max_relative_l2":max(v['relative_l2'] for v in results)}


def capture_rng():
    import numpy as np
    import torch
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def serial_metrics(value):
    import torch
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value,dict):
        return {str(k):serial_metrics(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [serial_metrics(v) for v in value]
    return value


def replay_gradients(args,cfg,resolved,rows,reference_paths,actual_paths,run_dir):
    import numpy as np
    import torch
    from specforge.training.assembly import build_model_bundle
    from specforge.training.strategies.base import DFlashTrainStrategy,StepContext
    from specforge.runtime.contracts import TrainBatch
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    bundle=build_model_bundle(cfg,algorithm=resolved.algorithm)
    model=bundle.model.train()
    if model.num_anchors != 8 or not model.teacher_metrics:
        raise AssertionError('actual trainer requires eight anchors and teacher metrics')
    checkpoint_checks=checkpoint_subset_audit(bundle,cfg.model.draft_checkpoint_path)
    write_json(run_dir/'checkpoint-byte-audit.json',checkpoint_checks)
    before=model_digest(model)
    write_json(run_dir/'draft-initial.json',{"state_sha256":before,"checkpoint":cfg.model.draft_checkpoint_path,"draft_config":bundle.draft_config.to_dict(),
                                          "trainable_parameters":{name:list(p.shape) for name,p in model.named_parameters() if p.requires_grad}})
    strategy=DFlashTrainStrategy(model)
    sampler=model._sample_anchor_positions
    all_passed=True
    for index,(row,ref_path,actual_path) in enumerate(zip(rows,reference_paths,actual_paths,strict=True)):
        rng=capture_rng();outputs=[];anchors=[]
        npstate=rng['numpy'];safe_rng={**rng,'numpy':(npstate[0],npstate[1].tolist(),npstate[2],npstate[3],npstate[4])}
        torch.save(safe_rng,run_dir/f'rng-{index:02d}.pt')
        for variant,path in enumerate((ref_path,actual_path)):
            model.zero_grad(set_to_none=True);restore_rng(rng)
            tensors=torch.load(path,map_location='cpu',weights_only=True)
            calls=[]
            def sample(*a,**kw):
                sampled=sampler(*a,**kw)
                recorded=tuple(t.detach().cpu().clone() for t in sampled)
                if variant:
                    if len(calls)>=len(anchors) or any(not torch.equal(x,y) for x,y in zip(recorded,anchors[len(calls)],strict=True)):
                        raise AssertionError(f'{row["id"]}: saved RNG did not reproduce anchors')
                    # Replay exactly the off pass anchors after checking the real sampler.
                    sampled=tuple(t.to(sampled[0].device) for t in anchors[len(calls)])
                calls.append(recorded)
                return sampled
            batch=TrainBatch(sample_ids=[row['id']],strategy='dflash',tensors=tensors)
            with mock.patch.object(model,'_sample_anchor_positions',side_effect=sample):
                output=strategy.forward_loss(batch,ctx=StepContext(global_step=index,total_steps=len(rows),collect_detailed_metrics=True))
                output.loss.backward()
            torch.cuda.synchronize()
            if not calls:
                raise AssertionError('actual DFlash sampler was never invoked')
            if not variant:
                anchors=calls;after_rng=capture_rng()
                torch.save({"anchors":anchors},run_dir/f'anchors-{index:02d}.pt')
            elif len(calls)!=len(anchors):
                raise AssertionError('anchor call count changed')
            grads={name:None if p.grad is None else p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
            for name,p in model.named_parameters():
                if not p.requires_grad and p.grad is not None:
                    raise AssertionError(f'frozen target received gradient: {name}')
            if not bool(torch.isfinite(output.loss)):
                raise AssertionError('nonfinite actual DFlash loss')
            outputs.append((output.loss.detach().cpu(),grads,serial_metrics({"metrics":output.metrics,"ratio_metrics":output.ratio_metrics,"loss_terms":output.loss_terms,"sum_metrics":output.sum_metrics})))
        restore_rng(after_rng)
        (ref_loss,ref_grads,ref_metrics),(loss,grads,metrics)=outputs
        loss_error=tensor_errors(loss,ref_loss,rtol=args.loss_rtol,atol=args.loss_atol)
        metric_checks=compare_metric_trees(metrics,ref_metrics,rtol=args.loss_rtol,atol=args.loss_atol)
        if not any('/loss_terms/' in check['path'] for check in metric_checks) or not any('teacher/' in check['path'] for check in metric_checks):
            raise AssertionError('loss numerator/denominator or teacher metrics were not actually evaluated')
        numerator=denominator=0.;passed=loss_error['passed'] and all(check['passed'] for check in metric_checks);nonzero=0;parameter_records=[]
        for name,expected in ref_grads.items():
            actual=grads[name]
            if (actual is None)!=(expected is None):
                raise AssertionError(f'None gradient pattern mismatch: {name}')
            if expected is None:
                record={"name":name,"none":True,"passed":True}
            else:
                error=tensor_errors(actual,expected,rtol=args.gradient_rtol,atol=args.gradient_atol)
                delta=(actual.float()-expected.float()).square().sum(dtype=torch.float64).item()
                norm=expected.float().square().sum(dtype=torch.float64).item()
                numerator+=delta;denominator+=norm;nonzero+=int(norm>0)
                record={"name":name,**error,"reference_sha256":tensor_sha256(expected),"actual_sha256":tensor_sha256(actual)}
                passed=passed and error['passed']
            parameter_records.append(record)
        if nonzero==0:
            raise AssertionError('no nonzero trainable draft gradient')
        relative_l2=(numerator/max(denominator,1e-60))**.5
        passed=passed and relative_l2<=args.gradient_relative_l2
        result={"index":index,"case":row['id'],"loss":float(loss),"reference_loss":float(ref_loss),"loss_error":loss_error,
                "gradient_relative_l2":relative_l2,"nonzero_gradient_parameters":nonzero,"passed":passed,"parameters":parameter_records,
                "reference_metrics":ref_metrics,"actual_metrics":metrics,"metric_comparisons":metric_checks,"anchors":[[t.tolist() for t in call] for call in anchors]}
        append_json(run_dir/'loss-gradient-results.jsonl',result)
        if not passed:
            torch.save({"reference":ref_grads,"actual":grads},run_dir/f'failure-gradients-{index:02d}.pt')
        all_passed=all_passed and passed
        print(f'GRADIENT {index+1:02d}/16 {row["id"]} loss={float(loss):.9g} relative_l2={relative_l2:.6g} passed={passed}',flush=True)
        del outputs,ref_grads,grads,output,batch,tensors
    after=model_digest(model)
    if after!=before:
        raise AssertionError('model state changed during no-update paired replay')
    model.zero_grad(set_to_none=True)
    write_json(run_dir/'draft-final.json',{"state_sha256":after,"matches_initial":True,"cuda_peak_allocated":torch.cuda.max_memory_allocated(),"cuda_peak_reserved":torch.cuda.max_memory_reserved()})
    if not all_passed:
        raise AssertionError('pretrained draft loss/gradient comparison failed')
    return {"cases":len(rows),"state_sha256":before,"passed":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--specforge-source',required=True);parser.add_argument('--sglang-source',required=True)
    parser.add_argument('--cases',required=True);parser.add_argument('--baseline-dir')
    parser.add_argument('--capture-only',action='store_true')
    parser.add_argument('--require-cache',action='store_true',help='Require complete negotiated diagnostics and actual nonzero prefix reuse')
    parser.add_argument('--replay-captures-from',help='Reuse a completed off/off run tensor artifact pair, starting no services')
    parser.add_argument('--feature-rtol',type=float,default=.02);parser.add_argument('--feature-atol',type=float,default=.02)
    parser.add_argument('--loss-rtol',type=float,default=.02);parser.add_argument('--loss-atol',type=float,default=.02)
    parser.add_argument('--gradient-rtol',type=float,default=.02);parser.add_argument('--gradient-atol',type=float,default=1e-5)
    parser.add_argument('--gradient-relative-l2',type=float,default=.02)
    args=parser.parse_args()
    def terminate(signum,frame):
        raise KeyboardInterrupt(f'owned harness received signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGHUP,terminate)
    run_dir=Path(args.run_dir).resolve();run_dir.mkdir(parents=True,exist_ok=False)
    result={"passed":False,"mode":'candidate-vs-baseline' if args.baseline_dir else 'off-off-repeatability'}
    started=[];manifest={}
    try:
        from specforge.config import load_config
        from specforge.application import resolve_run
        from specforge.launch_plan import build_launch_plan,_stop_services
        manifest=runtime_manifest(args);write_json(run_dir/'manifest.json',manifest)
        write_json(run_dir/'tolerances-predeclared.json',{k:v for k,v in vars(args).items() if 'rtol' in k or 'atol' in k or k=='gradient_relative_l2'})
        rows=load_cases(args.cases);write_json(run_dir/'cases.json',rows)
        path=isolated_config(args,run_dir);cfg=load_config(str(path));resolved=resolve_run(cfg)
        plan=build_launch_plan(resolved.config,algorithm=resolved.algorithm,config_path=str(path),env=os.environ)
        (run_dir/'launch-plan.json').write_text(plan.render())
        # Full CLI trainer is separately tested by root. Here start exactly the
        # supported one-server/Mooncake services and drive every case directly.
        if sum(s.command.label.startswith('capture-server-') for s in plan.services)!=1:
            raise AssertionError('exactly one capture server required')
        if args.replay_captures_from:
            if args.baseline_dir or args.require_cache:
                raise AssertionError('replay-only mode is for completed off/off evidence')
            previous=Path(args.replay_captures_from)
            if not json.loads((previous/'result.json').read_text()).get('passed'):
                raise AssertionError('cannot replay an incomplete baseline')
            if json.loads((previous/'cases.json').read_text()) != rows:
                raise AssertionError('replay input cases differ from completed baseline')
            reference=[previous/'off-0'/f'{i:02d}-{r["id"]}.pt' for i,r in enumerate(rows)]
            actual=[previous/'off-1'/f'{i:02d}-{r["id"]}.pt' for i,r in enumerate(rows)]
            info=json.loads((previous/'model-info.json').read_text())
            diagnostics=[{'H':0,'L':0,'N':len(row['input_ids']),'span':[0,len(row['input_ids'])],'source':'saved official-only evidence'} for row in rows]
            write_json(run_dir/'reused-evidence.json',{'source_run':str(previous),'reference':[str(p) for p in reference],'actual':[str(p) for p in actual]})
        else:
            started=start_services(plan,run_dir)
            if args.baseline_dir:
                baseline=Path(args.baseline_dir)
                prior=json.loads((baseline/'result.json').read_text())
                if not prior.get('passed'):
                    raise AssertionError('candidate requires a completed passing off baseline')
                if json.loads((baseline/'cases.json').read_text()) != rows:
                    raise AssertionError('candidate input cases differ from completed baseline')
                reference=[baseline/'off-0'/f'{i:02d}-{r["id"]}.pt' for i,r in enumerate(rows)]
                actual,info,diagnostics=capture_pass(args,cfg,resolved,plan,run_dir,'candidate',rows)
                write_json(run_dir/'model-info.json',info)
                if canonical_model_info(info)!=canonical_model_info(json.loads((baseline/'model-info.json').read_text())):
                    raise AssertionError('candidate capture/draft schema differs from completed baseline')
            else:
                reference,info,_=capture_pass(args,cfg,resolved,plan,run_dir,'off-0',rows)
                actual,info2,diagnostics=capture_pass(args,cfg,resolved,plan,run_dir,'off-1',rows)
                if info!=info2:
                    raise AssertionError('capture schema changed between repeats')
            _stop_services(started,grace_s=30);started=[]
        write_json(run_dir/'model-info.json',info)
        result['feature_comparison']=compare_features(args,rows,reference,actual,info,run_dir,diagnostics)
        if not args.capture_only:
            result['loss_gradient_comparison']=replay_gradients(args,cfg,resolved,rows,reference,actual,run_dir)
        result['capture_cases_per_pass']=len(rows);result['passed']=True
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}';(run_dir/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        if started:
            from specforge.launch_plan import _stop_services
            _stop_services(started,grace_s=30)
        result['ended_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(run_dir/'result.json',result)
        print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
