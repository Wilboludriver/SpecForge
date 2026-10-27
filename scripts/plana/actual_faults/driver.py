"""Actual loaded-engine fault probes. Default invocation only validates a plan.

Execution requires an explicit allocation/config from root; this module never
selects a GPU, port, model alternative, or weaker numerical tolerance.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from actual_faults import atomic_json, read_json, source_hash, wait_for

PROJECT = next(parent for parent in Path(__file__).resolve().parents
               if (parent/'NEXT_SESSION_PROMPT.txt').is_file())
FIXED = {
    'qwen3-4b': ('Qwen3-4B', 2560, [1,9,17,25,33], False),
    'qwen35-4b': ('qwen35-4B', 2560, [1,5,9,13,17,21,25,29], True),
    'qwen36-a3b': ('qwen36_a3b_0723_map_v2', 2048, [1,6,11,16,22,27,32,37], True),
}
KINDS = ('match_error', 'admit_no_token', 'admit_no_membership', 'admit_after_append',
         'd2h_partial', 'publish_fail', 'publish_hold_cancel', 'publish_hold_timeout', 'd2h_cancel',
         'prefill_retract', 'hs_only', 'kv_only', 'hs_shorter_than_kv', 'server_restart', 'same_content_fill_overlap')

DEFAULT_KINDS = tuple(kind for kind in KINDS if kind not in ('kv_only','hs_shorter_than_kv','server_restart','same_content_fill_overlap'))


def validate_config(config):
    name = config['model']['name']
    folder, width, layers, hybrid = FIXED[name]
    if config['model'] != {'name':name, 'hidden_size':width, 'layers':layers, 'hybrid':hybrid}:
        raise ValueError('model schema must match its fixed actual target configuration')
    args = config['engine_kwargs']
    expected_path = '/workdir/spectrain/spectrain-16g/models/' + folder
    if args['model_path'] != expected_path:
        raise ValueError('actual selected target model path mismatch')
    required = {'enable_spec_capture':True, 'spec_feature_cache':True, 'dtype':'bfloat16',
                'attention_backend':'fa3', 'page_size':1, 'context_length':2056,
                'max_running_requests':1, 'chunked_prefill_size':-1, 'disable_cuda_graph':True}
    for key, value in required.items():
        if args.get(key) != value:
            raise ValueError(f'engine {key} must explicitly be {value!r}')
    if args.get('spec_capture_aux_layer_ids') != layers or args.get('spec_capture_method') != 'dflash':
        raise ValueError('actual ordered DFlash capture layers/method mismatch')
    if hybrid and args.get('mamba_radix_cache_strategy') != 'extra_buffer':
        raise ValueError('actual hybrid probes require extra_buffer')
    if args.get('enable_deterministic_inference') is not True:
        raise ValueError('feature reuse requires the same deterministic-inference mode as the accepted numeric reference')
    allocation = config['allocation']
    ports = [allocation[key] for key in ('http_port', 'dispatch_port', 'nccl_port')]
    if len(set(ports)) != 3 or any(type(port) is not int or not 1024 <= port <= 65535 for port in ports):
        raise ValueError('three distinct root-assigned explicit ports are required')
    if args['port'] != allocation['http_port'] or args['nccl_port'] != allocation['nccl_port']:
        raise ValueError('server ports differ from the root allocation')
    tp = args['tp_size']
    if tp not in (1,2) or len(allocation['cuda_visible_devices'].split(',')) != tp:
        raise ValueError('root-assigned visible device count must exactly match TP1/TP2')
    if (name=='qwen3-4b' and tp!=1) or (name=='qwen36-a3b' and tp!=2):
        raise ValueError('fixed model acceptance requires Qwen3 TP1 and A3B TP2')
    if config['store_setup']['protocol'] != 'tcp' or config['mooncake_env'].get('MOONCAKE_PROTOCOL') != 'tcp':
        raise ValueError('these actual fault probes are TCP host publication only')
    if str(config['mooncake_env'].get('SGLANG_SPEC_CAPTURE_GPU_PUT')) != '0':
        raise ValueError('actual CPU prefix feature cache requires host publication')
    if config['feature_tolerances'] != {'rtol':.02, 'atol':.02}:
        raise ValueError('predeclared feature tolerances remain rtol=.02, atol=.02')
    if not 1 <= int(config.get('delay_cycles',2_000_000_000)) <= 2_000_000_000:
        raise ValueError('test CUDA delay must be explicitly bounded within two billion cycles')
    if not 1 <= float(config.get('http_timeout_s',10)) <= 30:
        raise ValueError('held-publication HTTP timeout must be between one and thirty seconds')
    unknown = set(config.get('kinds', KINDS)) - set(KINDS)
    if unknown:
        raise ValueError('unsupported fault kinds: ' + str(unknown))
    if 'server_restart' in config.get('kinds', DEFAULT_KINDS) and config.get('kinds') != ['server_restart']:
        raise ValueError('server_restart requires its own dedicated bounded run')
    if 'same_content_fill_overlap' in config.get('kinds', DEFAULT_KINDS):
        from actual_faults.concurrent_fill import validate_config as validate_fill
        validate_fill(config)
    return config


def source_manifest(config):
    result = {}
    for name in ('sglang', 'SpecForge'):
        root = PROJECT / 'src' / name
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
        head, status = git('rev-parse','HEAD'), git('status','--porcelain')
        if head != config['expected_commits'][name] or status:
            raise AssertionError(f'{name} is not the clean frozen expected commit: {head}, {status}')
        result[name] = {'root':str(root), 'commit':head, 'status':status}
    result['test_tools'] = {str(path):source_hash(path) for path in Path(__file__).parent.glob('*.py')}
    import sglang
    import specforge
    for name,module in (('sglang',sglang),('SpecForge',specforge)):
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(PROJECT/'src'/name):
            raise AssertionError('runtime import does not belong to frozen source: '+str(path))
        result[name]['import']=str(path)
    return result


def require_fault_outcome(client, future, row, reference, *, may_retry):
    from specforge.inference.adapters.server_capture import ServerCaptureFailure
    from specforge.runtime.contracts import SampleRef
    try:
        value = future.result(timeout=client.timeout)
    except Exception as exc:
        return {'outcome':'request_error', 'error':repr(exc)}
    if isinstance(value, ServerCaptureFailure):
        return {'outcome':'explicit_capture_failure', 'reason':value.reason}
    if not isinstance(value, SampleRef):
        raise AssertionError('fault returned an unknown output type: ' + repr(value))
    if not may_retry:
        raise AssertionError('injected transfer/publication fault unexpectedly produced a SampleRef')
    data, handle = client.store.get(value)
    try:
        from actual_faults.scenarios import check_features
        check_features(data, row, client.model, reference=reference, **client.config['feature_tolerances'])
    finally:
        client.store.release(handle)
    return {'outcome':'safe_retry_complete_sample', 'diagnostics':value.metadata.get('capture_diagnostics')}


def apply_hs_pressure(client, case_id, query, base, *, capacity, row_bytes, coverage):
    """Evict through real fills without keeping the victim hot by observing it."""
    import math
    needed = math.ceil(capacity / (len(base['input_ids']) * row_bytes)) + 4
    first_tokens = [token for token in range(18, 145)
                    if token != query['input_ids'][0]]
    if needed > len(first_tokens):
        raise ValueError('pressure working set exceeds the bounded distinct-token fixture')
    before = coverage()
    if before[0]['coverage'] <= 0:
        raise AssertionError('pressure victim lacks initial real HS coverage')
    # lookup_and_pin is an LRU access. Query only outside the pressure interval.
    for index, token in enumerate(first_tokens[:needed]):
        pressure = {name:list(base[name]) for name in ('input_ids','loss_mask')}
        pressure['input_ids'][0] = token
        client.fetch(client.submit(case_id+f'-pressure-{index}',pressure,force_full=True)[0],pressure)
    after = coverage()
    if after[0]['coverage'] != 0:
        raise AssertionError('real HS pressure did not evict the target prefix')
    return {'requests':needed, 'distinct_first_tokens':first_tokens[:needed],
            'before':before, 'after':after}


def run(config, directory):
    from actual_faults.launcher import start, drive, execute
    from actual_faults.scenarios import (Client, arm, command, rank_events, assert_fault_receipts,
                                        assert_prefill_retract_receipts, assert_cuda_cancel_receipts)
    from specforge.inference.capture_protocol import CaptureProtocol
    from specforge.inference.adapters.server_capture import _SessionPost
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest(config)
    atomic_json(directory/'manifest.json', {'source':manifest, 'config':config, 'scope':'actual loaded engine fault probes; no training/performance claim'})
    rows = {row['id']:row for row in map(json.loads, Path(config['cases_file']).read_text().splitlines())}
    base, query, replacement = rows['cold-1024'], rows['extension-1280'], rows['same-tail-different-history']
    server = client = script_pool = script_future = None
    result = {'passed':False, 'cases':[], 'scope':'actual engine faults, complete feature delivery and fenced cleanup'}
    try:
        server = start(config, directory)
        script_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='actual-script-driver')
        script_future = script_pool.submit(execute, server, drive, args=(str(directory),), timeout_s=float(config.get('suite_timeout_s',1800)))
        wait_for(lambda:(directory/'driver-ready.json').exists(), 90, 'official driver script ready')
        client = Client(config, directory, server._base_url)
        atomic_json(directory/'arm.json', {})
        if config.get('kinds') == ['same_content_fill_overlap']:
            from actual_faults.concurrent_fill import run_case
            result['cases'].append(run_case(client,directory,query))
            result['source_unchanged']=manifest==source_manifest(config)
            if not result['source_unchanged']:raise AssertionError('production/test source changed during concurrent-fill run')
            # Finally must finish HTTP cleanup/close before stopping the driver.
            result['passed']=True
            return result
        golden, _ = client.fetch(client.submit('golden-query',query,force_full=True)[0], query)
        replacement_golden, _ = client.fetch(client.submit('golden-replacement',replacement,force_full=True)[0], replacement)
        client.fetch(client.submit('warm-base',base,force_full=True)[0], base)
        client.cleanup()
        tp, timeout = client.tp_size, client.timeout
        initial = command(directory,'snapshot',tp_size=tp,timeout=timeout)
        if config['model']['hybrid']:
            for snapshot in initial:
                names=' '.join(snapshot['engine']['cache_components']).upper()
                if 'FULL' not in names or 'MAMBA' not in names:
                    raise AssertionError('actual hybrid fixture is not the observed FULL+MAMBA component backend')
        atomic_json(directory/'engine-baseline.json', initial)
        for kind in config.get('kinds', DEFAULT_KINDS):
            targets_list = ([list(range(tp))] if kind == 'prefill_retract' else
                            [[rank] for rank in range(tp)] if kind.startswith('admit_') or kind=='match_error' else [[0]])
            for targets in targets_list:
                case_id = f'{kind}-rank{targets[0]}'
                task_id = case_id
                rid = 'actual-' + uuid.uuid4().hex
                sample_id = client.run_id + ':' + task_id
                require_cow = config['model']['hybrid'] and (kind.startswith('admit_') or kind=='match_error')
                case = {'case_id':case_id,'kind':kind,'targets':targets,'require_cow':require_cow}
                if kind == 'prefill_retract':
                    # Warm this exact causal prefix under the live model, then retract its
                    # next genuinely allocated admission before any forward is launched.
                    client.fetch(client.submit(case_id+'-warm',query,force_full=True)[0],query,reference=golden)
                    client.cleanup()
                    arm(directory,case_id,kind,rid,sample_id,list(range(tp)))
                    _,ref=client.fetch(client.submit(task_id,query,rid=rid)[0],query,reference=golden)
                    case['diagnostics']=ref.metadata['capture_diagnostics']
                    assert_prefill_retract_receipts(directory,case_id,tp,case['diagnostics'])
                elif kind == 'hs_only':
                    before_flush = client.cleanup()
                    response = client.transport._session().post(
                        client.base_url+'/flush_cache', json={}, timeout=timeout)
                    response.raise_for_status()
                    after_flush = client.cleanup()
                    keys = ('ready_blocks','ready_bytes','cache_bytes')
                    before_hs = {key:before_flush['cache'][key] for key in keys}
                    after_hs = {key:after_flush['cache'][key] for key in keys}
                    if response.status_code != 200 or before_hs != after_hs:
                        raise AssertionError('actual engine flush did not preserve READY HS')
                    case['engine_flush'] = {'status':response.status_code,'body':response.text,
                                            'before_hs':before_hs,'after_hs':after_hs}
                    arm(directory,case_id,'coverage',rid,sample_id,[],require_cow=False)
                    _, ref = client.fetch(client.submit(task_id,query,rid=rid)[0],query,reference=golden)
                    diag = ref.metadata['capture_diagnostics']
                    if not (diag['H']>0 and diag['L']==0):
                        raise AssertionError('HS-only fixture did not observe actual H>0/L=0: '+repr(diag))
                    case['diagnostics']=diag
                elif kind in ('kv_only','hs_shorter_than_kv'):
                    C=config['engine_kwargs'].get('spec_feature_chunk_tokens',256)
                    row_bytes=config['model']['hidden_size']*(len(config['model']['layers'])+1)*2
                    capacity=config['engine_kwargs'].get('spec_feature_cache_bytes',1<<30)
                    pin_id='coverage-'+case_id
                    if kind=='hs_shorter_than_kv':
                        if capacity!=C*row_bytes:
                            raise ValueError('HS<KV requires a dedicated run with CPU cache exactly one complete feature chunk')
                        short={name:list(rows['chunk-exact'][name]) for name in ('input_ids','loss_mask')}
                        if C != len(short['input_ids']) or short['input_ids'] != query['input_ids'][:C]:
                            raise ValueError('HS<KV fixture requires the complete native chunk-exact row as its causal prefix')
                        client.fetch(client.submit(case_id+'-short-warm',short,force_full=True)[0],short)
                        command(directory,'pin_hs',tp_size=tp,timeout=timeout,tokens=short['input_ids'],pin_id=pin_id)
                        try:
                            client.fetch(client.submit(case_id+'-long-warm',query,force_full=True)[0],query,reference=golden)
                        finally:
                            command(directory,'release_hs',tp_size=tp,timeout=timeout,pin_id=pin_id)
                        case['ordinary_kv_probes']=client.warm_ordinary_checkpoint(query,case_id)
                    else:
                        client.fetch(client.submit(case_id+'-kv-warm-base',base,force_full=True)[0],base)
                        case['ordinary_kv_probes']=client.warm_ordinary_checkpoint(query,case_id)
                        case['kv_pin_receipts']=command(directory,'pin_kv',tp_size=tp,timeout=timeout,tokens=query['input_ids'],pin_id=pin_id)
                        try:
                            case['pressure_receipts']=apply_hs_pressure(
                                client,case_id,query,base,capacity=capacity,row_bytes=row_bytes,
                                coverage=lambda:command(directory,'coverage',tp_size=tp,
                                    timeout=timeout,tokens=query['input_ids']))
                        finally:
                            command(directory,'release_kv',tp_size=tp,timeout=timeout,pin_id=pin_id)
                    arm(directory,case_id,'coverage',rid,sample_id,[])
                    _,ref=client.fetch(client.submit(task_id,query,rid=rid)[0],query,reference=golden)
                    diag=ref.metadata['capture_diagnostics']
                    probes=[row for row in rank_events(directory,case_id) if row['event']=='uncapped_engine_probe']
                    if {row['rank'] for row in probes}!=set(range(tp)):
                        raise AssertionError('coverage fixture lacks actual uncapped checkpoint proof on every rank')
                    if kind=='kv_only':
                        if diag['H']!=0 or diag['L']!=0 or any(row['L']<=0 for row in probes):
                            raise AssertionError('KV-only fixture did not observe actual KV>0 / HS=0 / L=0')
                    elif not 0<diag['H']<min(row['L'] for row in probes):
                        raise AssertionError('HS<KV fixture did not observe actual positive H shorter than engine checkpoint coverage')
                    case['diagnostics']=diag;case['actual_uncapped_probes']=probes
                else:
                    mapped = 'publish_hold' if kind.startswith('publish_hold_') else kind
                    arm(directory,case_id,mapped,rid,sample_id,targets,require_cow=require_cow,
                        gate_timeout_s=min(timeout,45),delay_cycles=int(config.get('delay_cycles',2_000_000_000)))
                    request_timeout = float(config.get('http_timeout_s',10)) if kind=='publish_hold_timeout' else None
                    future,_ = client.submit(task_id,query,rid=rid,http_timeout=request_timeout)
                    if kind.startswith('publish_hold_') or kind=='d2h_cancel':
                        marker = directory / (f'event-{case_id}.json' if kind=='d2h_cancel' else f'gate-{case_id}.json')
                        wait_for(marker.exists, timeout, 'actual held transfer/publication boundary')
                        if kind=='d2h_cancel' and not read_json(marker)['pending']:
                            raise AssertionError('real DMA gate was not pending before cancellation')
                        with client.store._lock:
                            fence = client.store._capture_fences[sample_id]
                        if kind=='publish_hold_timeout':
                            if future.done():
                                raise AssertionError('HTTP timeout happened before the held publication boundary was observed')
                            try:
                                future.result(timeout=timeout)
                            except Exception:
                                pass
                            else:
                                raise AssertionError('held HTTP request did not time out')
                        status=fence.control('resolve',cancel=True)
                        if status['writer_drained']:
                            raise AssertionError('pending actual event/writer advertised prematurely drained')
                        protocol=CaptureProtocol(client.base_url,client.post,timeout)
                        capability=protocol.capability()
                        payload={'store_id':client.run_id,'sample_id':sample_id,'gen':1,
                                 'features':{'aux':'hidden_states','last_hidden':'target_last_hidden_states'},'passthrough':[]}
                        try:
                            protocol.begin(capability,payload,replacement['input_ids'],uuid.uuid4().hex)
                        except Exception as exc:
                            case['replacement_blocked']=repr(exc)
                        else:
                            raise AssertionError('replacement began before old writer and cleanup drain')
                        if kind.startswith('publish_hold_'):
                            command(directory,'release_gate',tp_size=tp,timeout=timeout,case_id=case_id)
                        case.update(require_fault_outcome(client,future,query,golden,may_retry=False))
                        if kind=='d2h_cancel':
                            case['pending_dma_proof']=assert_cuda_cancel_receipts(
                                rank_events(directory,case_id),case_id,read_json(marker))
                    else:
                        if kind=='d2h_partial':
                            marker=directory/f'partial-{case_id}.json'
                            wait_for(marker.exists,timeout,'actual first DMA issued before injected second-feature failure')
                            if not read_json(marker)['pending']:
                                raise AssertionError('partial-D2H fixture had no actual pending copy-stream event')
                            with client.store._lock:fence=client.store._capture_fences[sample_id]
                            if fence.control('resolve',cancel=False)['writer_drained']:
                                raise AssertionError('partial-D2H error advertised drain while issued DMA was still pending')
                        case.update(require_fault_outcome(client,future,query,golden,may_retry=kind.startswith('admit_') or kind=='match_error'))
                    assert_fault_receipts(directory,case_id,tp,targets,require_cow=require_cow)
                client.cleanup()
                if kind in ('d2h_cancel','d2h_partial'):
                    command(directory,'retire_delay',tp_size=tp,timeout=timeout,case_id=case_id)
                atomic_json(directory/'arm.json', {})
                # Same sample/g1, independent new attempt and different content.
                client.fetch(client.submit(task_id,replacement)[0],replacement,reference=replacement_golden)
                client.cleanup()
                snapshots=command(directory,'snapshot',tp_size=tp,timeout=timeout)
                for before,after in zip(initial,snapshots):
                    if after['engine']['locked_nodes']!=before['engine']['locked_nodes']:
                        raise AssertionError('actual engine retained request component locks after drain')
                    if after['engine']['req_pool_free']!=before['engine']['req_pool_free']:
                        raise AssertionError('actual engine retained request pool rows after drain')
                case['passed']=True;result['cases'].append(case)
                atomic_json(directory/'case-results.json', result['cases'])
        client.cleanup()
        result['source_unchanged']=manifest==source_manifest(config)
        if not result['source_unchanged']:
            raise AssertionError('production/test source changed during actual-engine run')
        # The final cleanup/close still needs the active scripted engine.
        result['passed']=True
    except BaseException as exc:
        import traceback
        result['error']=repr(exc)
        (directory/'failure.txt').write_text(traceback.format_exc())
    finally:
        # Never claim zero ownership if the real writer/receive drain is unknown.
        if server is not None and client is not None and script_future is not None and not script_future.done():
            current=read_json(directory/'arm.json',{})
            try:
                if current:command(directory,'release_gate',tp_size=client.tp_size,timeout=5,case_id=current['case_id'])
            except BaseException as exc:result['barrier_release_error']=repr(exc)
        if server is not None and client is not None and script_future is not None and not script_future.done():
            try:command(directory,'release_fixture_pins',tp_size=client.tp_size,timeout=5)
            except BaseException as exc:
                result['fixture_release_error']=repr(exc);result['passed']=False
        if client is not None:
            try:
                client.cleanup()
            except BaseException as exc:
                result['cleanup_error']=repr(exc);result['passed']=False
                with client.store._lock:
                    pending={sid:{'wire':fence.wire,'names':fence.names} for sid,fence in client.store._capture_fences.items()}
                    begins={sid:{'capability':intent[1],'payload':intent[2],'tokens':intent[3],
                                 'attempt_id':intent[4],'names':intent[5]}
                            for sid,intent in client.store._capture_pending_begin.items()}
                atomic_json(directory/'unresolved-ownership.json', {'health':client.store.health(),'fences':pending,'begin_intents':begins})
            try:client.close()
            except BaseException as exc:result['close_error']=repr(exc);result['passed']=False
        atomic_json(directory/'driver-stop.json', {'stop':True})
        if script_future is not None:
            try:script_future.result(timeout=10)
            except TimeoutError:
                # Touch only the Process object created by this launcher. The
                # bounded script poll notices its death before any socket reuse.
                server._terminate_process(server._server_process)
                try:script_future.result(timeout=5)
                except BaseException as exc:result['script_stop_error']=repr(exc);result['passed']=False
            except BaseException as exc:
                import traceback
                result['script_stop_error']=repr(exc);result['passed']=False
                (directory/'script-failure.txt').write_text(traceback.format_exc())
        if server is not None:
            try:server.shutdown()
            except BaseException as exc:result['shutdown_error']=repr(exc);result['passed']=False
        if script_pool is not None:script_pool.shutdown(wait=True,cancel_futures=False)
        result['ended_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        atomic_json(directory/'result.json',result)
    return result


def owned_tree_identity(root_pid):
    """Observe only descendants of the Process that this launcher created."""
    pending=[root_pid];result={}
    while pending:
        pid=pending.pop()
        if pid in result:continue
        try:
            stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            result[pid]=int(stat[19])
            pending.extend(int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
        except (FileNotFoundError,ProcessLookupError):pass
    return result


def owned_tree_still_live(identities):
    alive=[]
    for pid,born in identities.items():
        try:
            stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            if int(stat[19])==born and stat[0]!='Z':alive.append(pid)
        except (FileNotFoundError,ProcessLookupError):pass
    return alive


def run_restart(config, directory):
    """Actual owned-server restart: reject stale epoch without claiming key cleanup."""
    from actual_faults.launcher import start, drive, execute
    from actual_faults.scenarios import (Client, command, assert_restart_fail_closed, unresolved_ownership)
    from specforge.inference.capture_protocol import CaptureProtocol
    from specforge.runtime.contracts import SampleRef
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    manifest=source_manifest(config)
    atomic_json(directory/'manifest.json',{'source':manifest,'config':config,
        'scope':'actual server restart fail-closed; uncertain old ownership retained; no zero-leak or throughput claim'})
    rows={row['id']:row for row in map(json.loads,Path(config['cases_file']).read_text().splitlines())}
    query=rows['extension-1280'];slots=[];old_client=old_ref=None
    result={'passed':False,'scope':'actual owned S1/S2 restart fail-closed','cleanup_proven':False}
    def launch(name):
        path=directory/(directory.name+'-'+name);path.mkdir()
        server=start(config,path)
        slot={'name':name,'directory':path,'server':server,'pool':None,'future':None,'client':None,'closed':False}
        slots.append(slot)
        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='restart-script-'+name);slot['pool']=pool
        slot['future']=pool.submit(execute,server,drive,args=(str(path),),timeout_s=float(config.get('suite_timeout_s',1800)))
        wait_for(lambda:(path/'driver-ready.json').exists(),90,'restart '+name+' official driver ready')
        slot['client']=Client(config,path,server._base_url)
        return slot
    def stop_slot(slot):
        if slot['closed']:return
        server=slot['server'];path=slot['directory'];future=slot['future']
        identities=owned_tree_identity(server._server_process.pid)
        if server._server_process.is_alive() and server._server_process.pid not in identities:
            raise AssertionError('cannot establish the exact owned server process identity')
        atomic_json(path/'owned-tree-before-stop.json',identities)
        atomic_json(path/'driver-stop.json',{'stop':True})
        error=None
        try:
            if future is not None:future.result(timeout=10)
        except BaseException as exc:
            error=exc;server._terminate_process(server._server_process)
        finally:
            try:server.shutdown()
            finally:
                if slot['pool'] is not None:slot['pool'].shutdown(wait=True,cancel_futures=False)
        if server._server_process.is_alive():raise AssertionError('owned restart server leader survived shutdown')
        wait_for(lambda:not owned_tree_still_live(identities),10,'owned restart server descendants exit')
        slot['closed']=True
        atomic_json(path/'owned-tree-after-stop.json',{'live':owned_tree_still_live(identities),
                    'leader_exit':server._server_process.exitcode})
        if error is not None:raise error
    try:
        first=launch('server-epoch1');old_client=first['client']
        golden,_=old_client.fetch(old_client.submit('restart-forced-full-golden',query,force_full=True)[0],query)
        old_client.cleanup()
        # A completed HTTP publication proves this writer is done; no old consumer read starts.
        future,_=old_client.submit('retained-old-sample',query,force_full=True)
        old_ref=future.result(timeout=old_client.timeout)
        if not isinstance(old_ref,SampleRef):raise AssertionError('restart fixture did not publish an actual complete SampleRef')
        with old_client.store._lock:old_fence=old_client.store._capture_fences[old_ref.sample_id]
        published=old_fence.control('resolve',cancel=False)
        if published['state']!='PUBLISHED' or not published['writer_drained'] or published['active_readers']:
            raise AssertionError('restart fixture writer/reader state was not proven quiescent')
        protocol=CaptureProtocol(old_client.base_url,old_client.post,old_client.timeout)
        cap1=protocol.capability();stats=protocol.call({'action':'stats','server_epoch':cap1['server_epoch']})
        if stats['pending_deliveries'] or stats['active_readers'] or stats['controller']['pending_copies']:
            raise AssertionError('restart attempted with a live writer or receiver')
        old_keys={old_client.store._tkey(old_ref.sample_id,1,name) for name in old_ref.feature_specs}
        atomic_json(directory/'old-published-ownership.json',{'ref':__import__('dataclasses').asdict(old_ref),
                    'state':published,'stats':stats,'ownership':unresolved_ownership(old_client.store),
                    'physical_keys':sorted(old_keys)})
        stop_slot(first)
        second=launch('server-epoch2');new_client=second['client']
        protocol2=CaptureProtocol(new_client.base_url,new_client.post,new_client.timeout)
        cap2=protocol2.capability()
        if cap1['server_epoch']==cap2['server_epoch']:raise AssertionError('owned server restart reused its old epoch')
        outcome=assert_restart_fail_closed(old_client,old_ref,cap2['server_epoch'],
            lambda:old_client.submit('retained-old-sample',query)[0].result(timeout=old_client.timeout))
        atomic_json(directory/'unresolved-ownership.json',outcome['unresolved_ownership'])
        presence={}
        for key in sorted(old_keys):
            try:presence[key]='present' if old_client.store._store_exists(key) else 'absent'
            except Exception as exc:presence[key]={'state':'unknown','error':repr(exc)}
        outcome['old_key_presence_after_restart']=presence
        outcome['key_presence_is_retirement_proof']=False
        # A fresh unique run/store/sample namespace proves S2 still works normally.
        _,new_ref=new_client.fetch(new_client.submit('new-epoch-independent-sample',query,force_full=True)[0],query,reference=golden)
        if set(old_client.expected_keys)&set(new_client.expected_keys):raise AssertionError('restart reused old physical sample names')
        if new_ref.metadata['capture_fence']['server_epoch']!=cap2['server_epoch']:
            raise AssertionError('new namespace capture used the wrong restarted epoch')
        new_client.cleanup()
        command(second['directory'],'snapshot',tp_size=new_client.tp_size,timeout=new_client.timeout)
        atomic_json(directory/'restart-result.json',outcome)
        result.update(passed=True,fail_closed=outcome,new_namespace_complete=True,
                      old_ownership_retained=True,source_unchanged=manifest==source_manifest(config))
        if not result['source_unchanged']:raise AssertionError('restart production/test sources changed during run')
    except BaseException as exc:
        import traceback
        result.update(passed=False,error=repr(exc));(directory/'failure.txt').write_text(traceback.format_exc())
    finally:
        # Only the new namespace is eligible for ordinary cleanup. Old ownership
        # remains recorded even if native publisher teardown made its keys absent.
        if old_client is not None:
            try:atomic_json(directory/'unresolved-ownership.json',unresolved_ownership(old_client.store))
            except BaseException as exc:result.update(passed=False,ownership_snapshot_error=repr(exc))
        for slot in reversed(slots):
            client=slot['client']
            if client is not None:
                if client is not old_client:
                    try:client.cleanup()
                    except BaseException as exc:result.update(passed=False,new_cleanup_error=repr(exc))
                try:client.close()
                except BaseException as exc:result.update(passed=False,close_error=repr(exc))
            try:stop_slot(slot)
            except BaseException as exc:
                result.update(passed=False,shutdown_error=repr(exc))
                # Evidence/driver errors cannot skip owned Process teardown.
                try:slot['server']._teardown()
                except BaseException as cleanup:result['shutdown_fallback_error']=repr(cleanup)
                if slot['pool'] is not None:slot['pool'].shutdown(wait=True,cancel_futures=False)
        result['ended_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z');atomic_json(directory/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--run-dir');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--cpu-self-test',action='store_true')
    args=parser.parse_args()
    if args.cpu_self_test:
        return cpu_self_test()
    config=validate_config(read_json(args.config))
    if not args.execute:
        print(json.dumps({'validated':True,'model':config['model'],'allocation':config['allocation'],
                          'kinds':config.get('kinds',DEFAULT_KINDS),'scope':'plan only; no server/GPU/ports started'},indent=2))
        return 0
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=config['allocation']['cuda_visible_devices']:
        raise ValueError('CUDA_VISIBLE_DEVICES differs from explicit root assignment')
    execute=run_restart if config.get('kinds')==['server_restart'] else run
    result=execute(config,args.run_dir);print(json.dumps(result,indent=2));return 0 if result['passed'] else 1


def cpu_self_test():
    """Added below; never imports model/server launch paths."""
    import threading
    import unittest
    from types import SimpleNamespace
    import tempfile
    import torch
    from actual_faults.rank_hooks import RankState, patched_attribute, guarded_copy
    from actual_faults.scenarios import check_features
    class Mechanics(unittest.TestCase):
        def test_bridge_instrumentation_works_before_scheduler_assignment(self):
            with tempfile.TemporaryDirectory(dir=PROJECT/'implementation/validation/reviewer') as temp:
                pool=SimpleNamespace(available_size=lambda:4)
                scheduler=SimpleNamespace(ps=SimpleNamespace(tp_rank=0),waiting_queue=[],
                    req_to_token_pool=pool,token_to_kv_pool_allocator=pool,
                    tree_cache=SimpleNamespace(root_node=SimpleNamespace(children={})))
                bridge=SimpleNamespace()
                state=RankState(scheduler,temp,bridge=bridge)
                self.assertIs(state.bridge,bridge)
                self.assertFalse(hasattr(scheduler,'spec_feature_bridge'))
        def test_entry_wrapper_restores_even_if_original_worker_fails(self):
            from unittest import mock
            from actual_faults.launcher import scheduler_entry
            events=[]
            with mock.patch.dict(os.environ,{'PLANA_ACTUAL_FAULT_DIR':'/unused-test-directory'}), \
                 mock.patch('actual_faults.rank_hooks.install',side_effect=lambda path:(lambda:events.append('restore'))), \
                 mock.patch('sglang.srt.managers.scheduler.run_scheduler_process',side_effect=RuntimeError('worker failure')) as worker:
                with self.assertRaisesRegex(RuntimeError,'worker failure'):scheduler_entry('rank0-arguments',tp_rank=0)
                worker.assert_called_once_with('rank0-arguments',tp_rank=0)
            self.assertEqual(events,['restore'])
        def test_actual_global_shims_restore_and_cpu_copy_is_unchanged(self):
            from actual_faults.rank_hooks import install
            from sglang.srt.spec_feature_bridge import SpecFeatureBridge
            from sglang.srt.spec_capture_sink import SpecCaptureSink
            before=(SpecFeatureBridge.__init__,SpecCaptureSink.put_samples,torch.Tensor.copy_)
            restore=install('/unused-test-directory')
            try:
                value=torch.empty(2);value.copy_(torch.tensor([4.,5.]));self.assertEqual(value.tolist(),[4.,5.])
                self.assertIsNot(torch.Tensor.copy_,before[2])
            finally:restore()
            self.assertEqual((SpecFeatureBridge.__init__,SpecCaptureSink.put_samples,torch.Tensor.copy_),before)
        def test_exact_arm_and_one_shot(self):
            with tempfile.TemporaryDirectory(dir=PROJECT/'implementation/validation/reviewer') as temp:
                state=object.__new__(RankState);state.directory=Path(temp);state.rank=1;state.seen=set();state.emit=lambda *a,**k:None
                atomic_json(Path(temp)/'arm.json',{'case_id':'c','rid':'r','sample_id':'s','kind':'match_error','ranks':[1]})
                self.assertIsNone(state.arm_for(SimpleNamespace(rid='foreign-warmup',spec_capture={'sample_id':'s'})))
                self.assertIsNone(state.arm_for(SimpleNamespace(rid='r',spec_capture={'sample_id':'foreign'})))
                arm=state.arm_for(SimpleNamespace(rid='r',spec_capture={'sample_id':'s'}))
                self.assertTrue(state.fire(arm,'match_error'));self.assertFalse(state.fire(arm,'match_error'))
                state.rank=0;state.seen.clear();self.assertFalse(state.fire(arm,'match_error'))
        def test_instance_patch_restores_descriptor_and_errors(self):
            class Owner:
                def value(self):return 'original'
            owner=Owner()
            with self.assertRaises(RuntimeError):
                with patched_attribute(owner,'value',lambda:'patched'):
                    self.assertEqual(owner.value(),'patched');raise RuntimeError('boundary failure')
            self.assertEqual(owner.value(),'original');self.assertNotIn('value',vars(owner))
        def test_tensor_copy_guard_does_not_touch_foreign_cpu_work(self):
            src=torch.tensor([3.]);dst=torch.empty_like(src)
            calls=[]
            def original(target,source,**kw):calls.append(True);return target.copy_(source)
            self.assertIs(guarded_copy(original,dst,src),dst);self.assertEqual(dst.tolist(),[3.]);self.assertEqual(calls,[True])
        def test_full_feature_oracle_rejects_missing_and_trailing_rows(self):
            row={'input_ids':[1,2],'loss_mask':[0,1]};model={'hidden_size':2,'layers':[1,2]}
            features={'input_ids':torch.tensor([[1,2]]),'loss_mask':torch.tensor([[0,1]]),
                      'hidden_states':torch.ones(1,2,4,dtype=torch.bfloat16),'target_last_hidden_states':torch.ones(1,2,2,dtype=torch.bfloat16)}
            check_features(features,row,model)
            with self.assertRaises(AssertionError):check_features({k:v for k,v in features.items() if k!='target_last_hidden_states'},row,model)
            with self.assertRaises(AssertionError):check_features({**features,'hidden_states':features['hidden_states'][:,:1]},row,model)
        def test_publication_barrier_requires_explicit_release_receipt(self):
            with tempfile.TemporaryDirectory(dir=PROJECT/'implementation/validation/reviewer') as temp:
                state=object.__new__(RankState);state.directory=Path(temp);state.gates={};state.emit=lambda *a,**k:None
                completed=threading.Event();errors=[]
                def worker():
                    try:state.hold_publication({'case_id':'held','gate_timeout_s':3})
                    except Exception as exc:errors.append(exc)
                    finally:completed.set()
                thread=threading.Thread(target=worker);thread.start()
                wait_for(lambda:(Path(temp)/'gate-held.json').exists(),2,'explicit publication gate receipt')
                self.assertFalse(completed.is_set());state.gates['held'].set();thread.join(2)
                self.assertFalse(thread.is_alive());self.assertFalse(errors)
        def retract_fixture(self):
            # Real CPU cache/controller, Req.reset_for_retract and bridge invalidation;
            # actual CPU KV/Mamba slot allocators; the release coordinator is a
            # fixture. No model/forward is fabricated.
            from sglang.srt.spec_feature_cache import (FeatureSpec, FeatureSchema,
                PrefixFeatureCache, CaptureController, ExecutionIdentity)
            from sglang.srt.spec_feature_bridge import SpecFeatureBridge
            from sglang.srt.managers.schedule_batch import Req, NextBatchPlan
            from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
            from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
            schema=FeatureSchema((FeatureSpec('aux',2,'bfloat16',(1,)),FeatureSpec('last_hidden',2,'bfloat16',())))
            cache=PrefixFeatureCache(schema,2,4096)
            execution=ExecutionIdentity('cpu-server','r','attempt',0)
            tokens=(1,2,3,4)
            fill=cache.reserve_fill(cache.block_keys('cpu-namespace',tokens)[0],execution)
            cache.publish_ready(fill,{'aux':torch.ones(2,2,dtype=torch.bfloat16),
                                      'last_hidden':torch.ones(2,2,dtype=torch.bfloat16)})
            controller=CaptureController(cache,schema,4096,4096,4096)
            self.addCleanup(controller.shutdown)
            lease=cache.lookup_and_pin('cpu-namespace',tokens,4)
            reservation=controller.try_reserve_output(execution,4,2,0)
            frozen=controller.freeze_plan(execution,'cpu-namespace',tokens,4,2,lease,reservation)
            bridge=object.__new__(SpecFeatureBridge);bridge.writer=True;bridge.controller=controller;bridge.cache=cache
            entry=SimpleNamespace(state='ADMITTED',revoked=False,deliveries=0,drains=[],execution_epoch=0)
            req=SimpleNamespace(rid='r',spec_capture={'sample_id':'s'},spec_feature_attempt=entry,
                spec_feature_bridge=bridge,spec_feature_plan=frozen,spec_feature_limit=2,
                spec_feature_execution_epoch=0,spec_feature_force_full=False,retraction_count=0,
                req_pool_idx=3,kv=SimpleNamespace(kv_allocated_len=4),cache_protected_len=2,
                input_embeds=None,prefix_indices=torch.tensor([0,1]),
                sampling_params=SimpleNamespace(max_new_tokens=0),finished=lambda:False,
                spec_feature_scalar_plan={'execution_epoch':0,'H':2,'L':2,'N':4,'span':[2,4],'plan_digest':frozen.plan_digest})
            released=[]
            kv_allocator=TokenToKVPoolAllocator(20,torch.bfloat16,'cpu',None,False)
            kv=kv_allocator.alloc(4)
            mamba_allocator=MambaSlotAllocator(12,'cpu')
            cache_source=mamba_allocator.alloc(1)
            cow=mamba_allocator.alloc(1);ping_pong=mamba_allocator.alloc(2)
            req.mamba_pool_idx=cow[0];req.mamba_ping_pong_track_buffer=ping_pong
            mapping=torch.empty(4,4,dtype=torch.int64);mapping[3]=kv
            pool=SimpleNamespace(available_size=lambda:3+len(released),mamba_allocator=mamba_allocator,req_to_token=mapping)
            scheduler=SimpleNamespace(ps=SimpleNamespace(tp_rank=0,tp_size=2),server_args=object(),
                waiting_queue=[],req_to_token_pool=pool,token_to_kv_pool_allocator=kv_allocator,
                hisparse_coordinator=None,tree_cache=SimpleNamespace(root_node=SimpleNamespace(children={})))
            scheduler._add_request_to_queue=lambda value:scheduler.waiting_queue.append(value)
            state=object.__new__(RankState);state.scheduler=scheduler;state.bridge=bridge;state.rank=0
            state.seen=set();state.retract_forward_seen=set();state.transactions={'r':1};records=[]
            state.cow_receipts={'r':{'transaction':1,'slots':cow.tolist(),'sources':cache_source.tolist()}}
            state.emit=lambda event,**values:records.append(dict(event=event,rank=0,**values))
            armed={'case_id':'c','kind':'prefill_retract','rid':'r','sample_id':'s','ranks':[0,1]}
            state.arm_for=lambda value:armed if value.rid=='r' and value.spec_capture['sample_id']=='s' else None
            batch=SimpleNamespace(reqs=[req],forward_mode=SimpleNamespace(is_extend=lambda:True),is_prefill_only=True)
            plan=NextBatchPlan(batch_to_run=batch,running_batch=SimpleNamespace(reqs=[]))
            def release(**kwargs):
                self.assertEqual(kwargs['reqs'],[req]);self.assertIs(kwargs['tree_cache'],scheduler.tree_cache)
                self.assertFalse(kwargs['offload_kv']);released.append(req.req_pool_idx)
                kv_allocator.free_segment(mapping[req.req_pool_idx,req.cache_protected_len:req.kv.kv_allocated_len],start_pos=req.cache_protected_len)
                mamba_allocator.free(req.mamba_pool_idx.unsqueeze(0))
                mamba_allocator.free(req.mamba_ping_pong_track_buffer)
                req.kv=None;req.req_pool_idx=None;Req.reset_for_retract(req)
            return state,req,plan,release,NextBatchPlan,records,armed,released
        def test_real_cpu_plan_invalidates_on_prefill_retract_and_requires_new_forward(self):
            state,req,plan,release,make_plan,records,arm,released=self.retract_fixture()
            with self.assertRaisesRegex(AssertionError,'rematch'):
                state.observe_retracted_forward(plan.batch_to_run)
            result=state.maybe_retract_prefill(plan,retract_all_fn=release,next_batch_plan=make_plan)
            self.assertIsNone(result.batch_to_run);self.assertEqual(released,[3])
            self.assertEqual(state.scheduler.waiting_queue,[req]);self.assertEqual(req.spec_feature_execution_epoch,1)
            self.assertTrue(req.spec_feature_attempt.drains[0].done())
            self.assertEqual(state.bridge.cache.stats()['active_leases'],0)
            self.assertEqual(state.bridge.controller.stats()['output_reservations'],0)
            # One-shot never retracts the already requeued request a second time.
            self.assertIs(state.maybe_retract_prefill(plan,retract_all_fn=release,next_batch_plan=make_plan),plan)
            self.assertEqual(released,[3])
            req.spec_feature_scalar_plan={**req.spec_feature_scalar_plan,'execution_epoch':1,'L':0,'span':[0,4]}
            state.transactions['r']=2
            state.observe_retracted_forward(plan.batch_to_run)
            self.assertEqual([row['event'] for row in records],['fault_fired','prefill_retract_allocation_return','prefill_retract_applied','prefill_retract_forward'])
            with self.assertRaisesRegex(AssertionError,'exactly one'):
                state.observe_retracted_forward(plan.batch_to_run)
        def test_prefill_retract_guard_rejects_foreign_unallocated_decode_or_partial_rank(self):
            for mutation in ('foreign','unallocated','decode','partial_rank','no_warm_hs','no_legal_kv','no_reset'):
                with self.subTest(mutation=mutation):
                    state,req,plan,release,make_plan,records,arm,released=self.retract_fixture()
                    if mutation=='foreign':req.spec_capture={'sample_id':'foreign'}
                    elif mutation=='unallocated':req.kv=None
                    elif mutation=='decode':req.sampling_params.max_new_tokens=1
                    elif mutation=='partial_rank':arm['ranks']=[0]
                    elif mutation=='no_warm_hs':req.spec_feature_scalar_plan['H']=0
                    elif mutation=='no_legal_kv':req.spec_feature_scalar_plan['L']=0
                    elif mutation=='no_reset':release=lambda **_:None
                    if mutation=='foreign':
                        self.assertIs(state.maybe_retract_prefill(plan,retract_all_fn=release,next_batch_plan=make_plan),plan)
                        self.assertFalse(records)
                    else:
                        with self.assertRaises(AssertionError):
                            state.maybe_retract_prefill(plan,retract_all_fn=release,next_batch_plan=make_plan)
        def test_prefill_retract_rejects_missing_kv_or_mamba_returns_and_restores_allocators(self):
            from unittest import mock
            for kind in ('kv','mamba','duplicate_mamba'):
                with self.subTest(kind=kind):
                    state,req,plan,release,make_plan,records,arm,released=self.retract_fixture()
                    allocator=(state.scheduler.token_to_kv_pool_allocator if kind=='kv'
                               else state.scheduler.req_to_token_pool.mamba_allocator)
                    original=allocator.free
                    # The real release coordinator calls free, but the injected
                    # allocator loses the physical return. Req reset alone must
                    # never make this a successful retract receipt.
                    def defect(indices):
                        if kind=='duplicate_mamba':
                            original(indices);original(indices)
                    with mock.patch.object(allocator,'free',defect):
                        expected=('free list' if kind=='duplicate_mamba' else 'allocator did not return')
                        with self.assertRaisesRegex(AssertionError,expected):
                            state.maybe_retract_prefill(plan,retract_all_fn=release,next_batch_plan=make_plan)
                    self.assertEqual(allocator.free,original)
                    self.assertNotIn('free_segment',vars(state.scheduler.token_to_kv_pool_allocator))
        def test_prefill_retract_allows_eviction_without_counting_cache_owned_slots(self):
            state,req,plan,release,make_plan,records,arm,released=self.retract_fixture()
            pool=state.scheduler.req_to_token_pool
            cached_kv=pool.req_to_token[req.req_pool_idx,:req.cache_protected_len].clone()
            cached_mamba=torch.tensor(state.cow_receipts['r']['sources'])
            def release_and_evict(**kwargs):
                release(**kwargs)
                state.scheduler.token_to_kv_pool_allocator.free(cached_kv)
                pool.mamba_allocator.free(cached_mamba)
            state.maybe_retract_prefill(plan,retract_all_fn=release_and_evict,next_batch_plan=make_plan)
            event=next(row for row in records if row['event']=='prefill_retract_applied')
            owned=event['allocation_return']['owned']
            self.assertEqual(len(owned['kv_indices']),2)
            self.assertEqual(len(owned['mamba_slots']),3)
            self.assertFalse(set(owned['mamba_slots']) & set(owned['cache_owned_cow_sources']))
            self.assertEqual(event['after']['kv_pool_free']-event['before']['kv_pool_free'],4)
            self.assertEqual(event['after']['mamba_pool_free']-event['before']['mamba_pool_free'],4)
        def test_prefill_retract_oracle_rejects_missing_duplicate_or_cross_rank_epoch(self):
            import copy
            from actual_faults import append_json
            from actual_faults.scenarios import assert_prefill_retract_receipts
            state,req,plan,release,make_plan,events,arm,released=self.retract_fixture()
            state.maybe_retract_prefill(plan,retract_all_fn=release,next_batch_plan=make_plan)
            req.spec_feature_scalar_plan={**req.spec_feature_scalar_plan,'execution_epoch':1,'L':0,'span':[0,4]}
            state.transactions['r']=2;state.observe_retracted_forward(plan.batch_to_run)
            events.extend({'event':'admit_collective','rank':0,'case_id':'c','transaction':t,'method':'gather'} for t in (1,2))
            events.extend({'event':'admit_return','rank':0,'case_id':'c','transaction':t,'member':True} for t in (1,2))
            for defect in ('none','missing','duplicate','wrong_epoch','missing_kv_free','missing_mamba_free','cache_slot_as_owned','pool_return_only','free_call_only','old_rank_L'):
                with self.subTest(defect=defect),tempfile.TemporaryDirectory(dir=PROJECT/'implementation/validation/reviewer') as temp:
                    for rank in (0,1):
                        rows=copy.deepcopy(events)
                        for row in rows:row['rank']=rank
                        if rank==1 and defect=='missing':rows=[row for row in rows if row['event']!='prefill_retract_forward']
                        if rank==1 and defect=='duplicate':rows.append(copy.deepcopy(rows[0]))
                        if rank==1 and defect=='wrong_epoch':next(row for row in rows if row['event']=='prefill_retract_forward')['plan']['execution_epoch']=2
                        applied=next(row for row in rows if row['event']=='prefill_retract_applied')
                        receipt=applied['allocation_return']
                        if rank==1 and defect=='missing_kv_free':receipt['kv_free_calls']=[]
                        if rank==1 and defect=='missing_mamba_free':receipt['mamba_free_calls']=[]
                        if rank==1 and defect=='cache_slot_as_owned':receipt['owned']['cache_owned_cow_sources']=receipt['owned']['fresh_cow_slots']
                        if rank==1 and defect=='pool_return_only':receipt['kv_free_calls']=[{'method':'free','indices':[999]}]
                        if rank==1 and defect=='free_call_only':receipt['mamba_free_after']=[]
                        if rank==1 and defect=='old_rank_L':applied['old_plan']['L']=1
                        for row in rows:append_json(Path(temp)/f'rank-{rank}.jsonl',row)
                    if defect=='none':assert_prefill_retract_receipts(temp,'c',2,req.spec_feature_scalar_plan)
                    else:
                        with self.assertRaises(AssertionError):assert_prefill_retract_receipts(temp,'c',2,req.spec_feature_scalar_plan)
        def test_restart_stale_fence_uses_actual_control_rejection_and_blocks_physical_io(self):
            import requests
            from actual_faults.scenarios import assert_restart_fail_closed
            from specforge.inference.capture_protocol import WriterFence, CaptureControlRejected
            for defect in ('none','get_succeeds','fetch_crossed','generate_crossed','retired_on_reject'):
                with self.subTest(defect=defect):
                    wire={'server_epoch':'old-epoch','store_id':'old-store','sample_id':'s','gen':1,
                          'attempt_id':'a','reservation_token':'reservation','control_url':'http://unused/spec_capture/control','timeout_s':1}
                    store=SimpleNamespace(_lock=threading.RLock(),_capture_fences={},_capture_pending_begin={},_capture_read_intents={},
                        _get_tensors=lambda *_a:None,_store_remove=lambda *_a:None,_store_put_tensor=lambda *_a:None,
                        health=lambda:{'capture_fences':1})
                    def rejected(url,*,json_body,timeout):
                        response=SimpleNamespace(status_code=409)
                        raise requests.HTTPError('actual response409 epoch rejection',response=response)
                    client=SimpleNamespace(store=store,transport=rejected)
                    def post(url,*,json_body,timeout):return client.transport(url,json_body=json_body,timeout=timeout)
                    fence=WriterFence(wire,('aux','last'),post=post,adopted=True)
                    store._capture_fences['s']=fence
                    store.get=lambda ref:fence.acquire_read('owner')
                    store.abort=lambda *_a,**_kw:fence.acquire_cleanup()
                    retry=lambda:fence.control('resolve',cancel=False)
                    if defect=='get_succeeds':store.get=lambda ref:({},'fake-handle')
                    elif defect=='fetch_crossed':store.get=lambda ref:store._get_tensors(ref)
                    elif defect=='generate_crossed':retry=lambda:client.transport('http://unused/generate',json_body={},timeout=1)
                    elif defect=='retired_on_reject':
                        def abort(*_a,**_kw):
                            try:fence.acquire_cleanup()
                            except CaptureControlRejected:
                                fence.acknowledged=True;raise
                        store.abort=abort
                    ref=SimpleNamespace(sample_id='s',metadata={'capture_fence':dict(wire)})
                    if defect=='none':
                        report=assert_restart_fail_closed(client,ref,'new-epoch',retry)
                        self.assertEqual(len(report['explicit_rejections']),4)
                        self.assertFalse(any(report['physical_operations'].values()))
                        self.assertFalse(report['cleanup_proven']);self.assertIn('s',report['unresolved_ownership']['fences'])
                        with self.assertRaisesRegex(AssertionError,'did not change'):
                            assert_restart_fail_closed(client,ref,'old-epoch',retry)
                    else:
                        with self.assertRaises(AssertionError):assert_restart_fail_closed(client,ref,'new-epoch',retry)
        def test_owned_restart_tree_identity_ignores_reused_pids_and_zombies(self):
            from unittest import mock
            def stat(state,birth):return '1 (owned process) '+' '.join([state]+['0']*18+[str(birth)])
            values={'/proc/10/stat':stat('S',101),'/proc/10/task/10/children':'11',
                    '/proc/11/stat':stat('S',202),'/proc/11/task/11/children':''}
            def read(path,*_a,**_kw):
                try:return values[str(path)]
                except KeyError:raise FileNotFoundError(str(path))
            with mock.patch.object(Path,'read_text',read):
                identities=owned_tree_identity(10)
                self.assertEqual(identities,{10:101,11:202})
                self.assertEqual(owned_tree_still_live(identities),[10,11])
                values['/proc/10/stat']=stat('S',999);values['/proc/11/stat']=stat('Z',202)
                self.assertEqual(owned_tree_still_live(identities),[])
        def test_admission_trace_oracle_rejects_rank_sequence_mismatch(self):
            from actual_faults.scenarios import assert_fault_receipts
            from actual_faults import append_json
            with tempfile.TemporaryDirectory(dir=PROJECT/'implementation/validation/reviewer') as temp:
                for rank in (0,1):
                    path=Path(temp)/f'rank-{rank}.jsonl'
                    if rank==0:append_json(path,{'rank':rank,'event':'fault_fired','case_id':'c'})
                    append_json(path,{'rank':rank,'event':'admit_collective','case_id':'c','transaction':1,'method':'gather' if rank==0 else 'broadcast'})
                with self.assertRaisesRegex(AssertionError,'collective sequence differs'):
                    assert_fault_receipts(temp,'c',2,[0],require_cow=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Mechanics)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'scope':'CPU test-tool mechanics only; no actual loaded-engine/GPU acceptance',
                      'tests':result.testsRun,'passed':result.wasSuccessful()}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
