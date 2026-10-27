#!/usr/bin/env python3
"""Read matrix artifacts and emit honest JSON/CSV observations, never a speed claim."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

STAGES=('forward_ms','d2h_wait_ms','assembly_ms','cache_fill_ms','publication_queue_ms')
GROUPS=('device-off','host-off','host-on')
NONMEASURED=('failed','unavailable','planned','running','incomplete','not_started','conditional_pending',
             'conditional_not_selected','not_run_native_gdr_unavailable','invalid_data')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)

def number(value,name,*,positive=False):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0 and (not positive or value>0),'invalid '+name)
    return value


def expected_cases(matrix,contract):
    selected=None if contract['models']=='all' else set(contract['models'].split(','))
    workloads=None if contract['workloads']=='all' else set(contract['workloads'].split(','))
    result={}
    for model in matrix['models']:
        if selected is not None and model['id'] not in selected:continue
        for workload in model['workloads']:
            if workloads is not None and workload not in workloads:continue
            for repetition in range(1,(contract['hot_repetitions'] if workload=='hot' else 1)+1):
                for group in GROUPS:
                    for transport in (('rdma',) if group=='device-off' else ('rdma','tcp')):
                        identifier=f"{model['id']}/{workload}/r{repetition}/{transport}-{group}"
                        result[identifier]={'id':identifier,'model':model['id'],'workload':workload,'group':group,
                            'transport':transport,'repetition':repetition,
                            'measured_steps':matrix['real_measured_steps'] if workload=='real64' else matrix['synthetic_measured_steps']}
    require(bool(result),'empty checkpoint selection')
    return result


def validate_sequence(result,case):
    audit=result['consumed_order'];training=result['training'];fit=result['actual_fit']
    require(audit['full_order_matches'] is True and audit['measured_counter_matches'] is True,'consumed audit did not pass')
    fingerprints=audit['ordered_fingerprints'];steps=audit['steps'];measured=audit['measured_fingerprints']
    require(fingerprints and all(isinstance(value,str) and value for value in fingerprints) and len(steps)==len(fingerprints),'missing real consumed steps')
    require([row['step'] for row in steps]==list(range(1,len(steps)+1)),'missing/duplicate actual optimizer steps')
    require([row['content_fingerprint'] for row in steps]==fingerprints,'step/fetch fingerprints differ')
    require(all(isinstance(row['sample_id'],str) and row['sample_id'] for row in steps),'missing consumed sample identity')
    require(audit['ordered_sha256']==digest(fingerprints),'actual sequence hash differs')
    warm=training['warmup_optimizer_steps'];count=training['measured_optimizer_steps']
    require(isinstance(warm,int) and warm>=0 and count==case['measured_steps'],'wrong measured horizon')
    require(len(measured)==count and fingerprints[warm:]==measured and len(steps)==warm+count,'consumed warmup/measured count differs')
    require(dict(Counter(measured))==audit['measured_counter'],'measured Counter differs')
    require(fit['completed_steps']==len(steps) and fit['warmup_steps']==warm and fit['measured_steps']==count,'actual fit/optimizer count differs')
    wall=number(training['optimizer_step_wall_s'],'primary optimizer wall',positive=True)
    require(math.isclose(wall,number(fit['measured_fit_boundary_wall_s'],'actual fit boundary',positive=True)/count,rel_tol=1e-9,abs_tol=1e-12),'fit wall and step wall differ')
    return {'consumed_steps_total':len(steps),'measured_consumed_steps':count,'warmup_consumed_steps':warm,
        'sequence_sha256':audit['ordered_sha256'],'measured_counter_sha256':digest(audit['measured_counter']),
        'training_step_wall_s':wall,'controller_window_step_wall_s':number(training['controller_window_optimizer_step_wall_s'],'controller window',positive=True),
        'actual_fit_wall_s':number(fit['fit_wall_s'],'fit wall',positive=True),
        'training_primary_boundary':training['primary_boundary']}


def validate_capture(capture,count):
    require(capture['requests']==count,'capture count does not match consumed measured steps')
    N=capture['tokens'];L=capture['reused_tokens'];computed=capture['computed_tokens']
    require(all(isinstance(value,int) and not isinstance(value,bool) for value in (N,L,computed)) and N>0 and 0<=L<N and computed==N-L and computed>=count,'invalid token totals N/L/recompute')
    require(math.isclose(capture['token_weighted_L_over_N'],L/N,rel_tol=1e-10,abs_tol=1e-12),'weighted token reuse differs')
    stages={}
    for name in STAGES:
        row=capture['stages'][name];requests=row['requests'];coverage=row['coverage'];total=row['total_ms']
        require(isinstance(requests,int) and 0<=requests<=count,'bad stage request count')
        require(math.isclose(number(coverage,'stage coverage'),requests/count,rel_tol=1e-9,abs_tol=1e-12),'stage coverage differs')
        if requests==0:require(total is None,'missing stage was relabelled zero')
        else:number(total,'stage total')
        stages[name]={'total_ms':total,'requests':requests,'coverage':coverage}
    return {'N_tokens_total':N,'L_tokens_total':L,'recomputed_tokens':computed,'token_weighted_L_over_N':L/N,
            'capture_requests':count,'stages':stages,'token_boundary':'totals over actual measured optimizer-consumed samples'}


def source_check(manifest,after,state,case):
    side='candidate' if case['group']=='host-on' else 'official'
    expected=state['input_verification']['identities'][side]
    for name in ('specforge','sglang'):
        require(manifest[name]['commit']==after[name]['commit']==expected[name]['commit'],'source commit changed/mismatched')
    for key in ('harness_sha256','helpers_sha256'):
        require(manifest[key]==state['input_verification'][key],'harness/helper bytes differ')


def cleanup_check(run_dir,record):
    cleanup=load(run_dir/'cleanup-final.json');exits=load(run_dir/'owned-processes-final.json')
    require(cleanup['remaining_keys']==[] and number(cleanup['inspected_keys'],'inspected keys',positive=True),'published keys not proven removed')
    require(exits and all(row['exit'] is not None for row in exits),'owned harness process not drained')
    require(record['stage']=='drained' and record['exit']==0 and record['resource_release']['released'] is True,'matrix resource release not proven')
    tree_path=Path(record['owned_tree_report'])
    if not tree_path.exists():tree_path=run_dir.parents[4]/'logs'/Path(*run_dir.parts[-4:])/'owned-tree-final.json'
    tree=load(tree_path)
    require(tree['remaining']==[] and tree['coverage_errors']==[] and not tree.get('remaining_direct_children') and not tree.get('emergency_cleanup'),'matrix owned tree not safely drained')
    if cleanup['control_status']!=404:
        stats=cleanup['stats']
        require(all(stats[name]==0 for name in ('active_attempts','active_readers','pending_deliveries')),'server attempt/reader/delivery leak')
        if stats['cache_enabled']:
            require(all(stats['cache'][name]==0 for name in ('active_leases','fill_reservations','writing_blocks')),'cache ownership leak')
            require(all(stats['controller'][name]==0 for name in ('pinned_bytes','output_bytes','device_bytes','output_reservations','pending_executions','pending_copies','quarantined_executions')),'controller private buffer leak')
    return cleanup


def native_proof(proof,boot_id):
    require(proof['protocol']=='rdma' and proof['gpu_put'] is True and proof['cuda_source'] is True,'TCP/host is not native device publication')
    require(proof['cuda_register_status']==proof['put_status']==0 and proof['fetch_equal'] is True and proof['keys_removed'] is True,'native CUDA transfer proof failed')
    require(proof['rdma_devices'] and not proof['forbidden_overrides'] and proof['local_memcpy_setting']=='0','native transport overrides/identity invalid')
    require(proof['boot_id']==boot_id,'device proof and execution boot differ')


def unavailable_proof(record,case,state,checkpoint_root):
    evidence=record.get('gdr') or record.get('inherited_typed_proof') or record.get('fallback_native_unavailable')
    require(evidence and evidence['status']=='unavailable','typed native unavailable evidence missing')
    source=Path(evidence['run_dir']);result=load(source/'result.json');probe=load(source/'device-proof.json');cleanup=load(source/'probe-cleanup.json')
    require(result['status']=='unavailable' and result['passed'] is False and result['source_unchanged'] is True
        and result['error'].startswith('DevicePublicationUnavailable:') and result['group']=='device-off' and result['transport']=='rdma','unrelated failure was relabelled unavailable')
    require(probe['status']=='unavailable' and cleanup['key_removal_authorized'] is True and cleanup['errors']==[],'unavailable native probe cleanup uncertain')
    require(evidence['result_sha256']==file_sha(source/'result.json') and evidence['boot_id']==state['contract']['boot_id'],'unavailable proof changed/boot differs')
    source_id=str(source.relative_to(checkpoint_root/'runs'))
    require(source_id.split('/')[0]==case['model'],'native unavailable evidence belongs to a different model allocation')
    source_record=state['cases'][source_id]
    require(source_record['resource_release']['released'] is True,'unavailable native probe resources not released')
    source_check(load(source/'manifest.json'),load(source/'source-after.json'),state,{'group':'device-off'})
    require(all(row['exit'] is not None for row in load(source/'owned-processes-final.json')),'unavailable probe retained owned processes')
    tree=load(source_record['owned_tree_report'])
    require(tree['remaining']==[] and tree['coverage_errors']==[] and not tree.get('remaining_direct_children') and not tree.get('emergency_cleanup'),'unavailable probe tree not drained')
    return {'reason':result['error'],'source_run_dir':str(source),'probe':probe,'boot_id':evidence['boot_id'],'hca':evidence['hca']}


def server_stages(values,count):
    require(isinstance(values,dict),'server stage mapping missing')
    result={}
    for name,row in values.items():
        events=row['events'];total=row['milliseconds']
        require(isinstance(events,int) and not isinstance(events,bool) and events>=0,'invalid server stage event count')
        if events==0:require(total is None,'missing server stage relabelled zero')
        else:number(total,'server stage milliseconds')
        result[name]={**row,'events_per_measured_request':events/count}
    return result


def memory_fields(result,cleanup):
    fit=result['actual_fit'];observed=result['observed_resources'];measured=result['measured_window_observed_resources']
    for section in (observed,measured):
        require('host_metric_coverage' in section and 'sample_count' in section and 'errors' in section,'resource coverage missing')
        for name in ('host_pss_peak_bytes','host_rss_sum_peak_bytes','host_locked_sum_peak_bytes','gpu_physical_used_sum_peak_bytes'):
            require(name in section,'resource metric missing')
            if section[name] is not None:number(section[name],name)
        require(isinstance(section['sample_count'],int) and section['sample_count']>=0,'invalid resource sample count')
        for field,name in (('pss_bytes','host_pss_peak_bytes'),('rss_bytes','host_rss_sum_peak_bytes'),('locked_bytes','host_locked_sum_peak_bytes')):
            complete=section['host_metric_coverage'][field]['complete_samples']
            require(isinstance(complete,int) and 0<=complete<=section['sample_count'],'invalid host metric coverage')
            require((complete==0)==(section[name] is None),'unavailable host metric relabelled as a value')
        require((section['gpu_complete_samples_with_values']==0)==(section['gpu_physical_used_sum_peak_bytes'] is None),'unavailable GPU metric relabelled as a value')
    allocator={name:number(fit[name],name) for name in ('torch_peak_allocated_bytes','torch_peak_reserved_bytes')}
    require(allocator['torch_peak_reserved_bytes']>=allocator['torch_peak_allocated_bytes'],'allocator reserved smaller than allocated')
    return {'observed_whole_command':observed,'observed_measured_window':measured,
        'trainer_allocator_fit_including_warmup':allocator,
        'boundaries':{'observed':'simultaneous sampled owned-process PSS/RSS/GPU-used sums; observed bounds, not exact peaks',
                      'allocator':'one trainer rank torch allocator during fit, warmup included; not target GPU or physical process total'},
        'cache_controller_final':cleanup.get('stats',{}).get('controller'),
        'cache_final':cleanup.get('stats',{}).get('cache')}


def passed_row(row,record,state,run_dir,seen_runs):
    result=load(run_dir/'result.json');manifest=load(run_dir/'manifest.json')
    require(file_sha(run_dir/'result.json')==record['result_sha256'],'result hash changed')
    require(state['contract']['execute'] is True and result['status']=='passed' and result['passed'] is True and result['source_unchanged'] is True,'case is not actual passed execution')
    require(all(result[key]==row[key] for key in ('group','workload','transport')),'case/result identity differs')
    require(row['group']!='device-off' or row['transport']=='rdma','TCP device result forbidden')
    argv=manifest['argv'];instance=argv[argv.index('--run-dir')+1]
    require(instance not in seen_runs,'same actual run appears in multiple inputs');seen_runs.add(instance)
    row.update(specforge_commit=manifest['specforge']['commit'],sglang_commit=manifest['sglang']['commit'])
    source_check(manifest,load(run_dir/'source-after.json'),state,row)
    row.update(validate_sequence(result,row));row.update(validate_capture(result['capture'],row['measured_consumed_steps']))
    draft=result['initial_draft'];require(len(draft)==1 and draft[0]['sha256'],'one initial draft fingerprint required')
    cleanup=cleanup_check(run_dir,record)
    require(result['resource_sampler']['thread_drained'] is True and not result['resource_sampler']['fatal_errors'],'resource sampler did not drain')
    if row['group']=='device-off':
        native_proof(load(run_dir/'device-proof.json'),state['contract']['boot_id'])
        require(result['device_publication_available'] is True and result['server_stage_attribution']['publication_register_put_unregister_wall_ms']['events']>0,'actual model CUDA-source publication observations missing')
    if row['transport']=='tcp':row['native_unavailable']=unavailable_proof(record,row,state,run_dir.parents[4])
    row.update(status='passed',measurement_eligible=True,source_pass=True,cleanup_pass=True,
        specforge_commit=manifest['specforge']['commit'],sglang_commit=manifest['sglang']['commit'],
        cleanup_scope='complete training publication/consumer/owned-resource drain',comparison_contract_sha256=digest(manifest['comparison_contract']),initial_draft_sha256=draft[0]['sha256'],
        harness_sha256=manifest['harness_sha256'],helpers_sha256=manifest['helpers_sha256'],
        server_stage_attribution=server_stages(result['server_stage_attribution'],row['measured_consumed_steps']),memory=memory_fields(result,cleanup),
        whole_command_wall_s=result.get('whole_command_wall_s'),validation_wall_s=result.get('validation_wall_s'))


def collect_checkpoint(path,matrix,matrix_sha,seen_runs):
    path=Path(path).resolve();state=load(path);root=path.parent;contract=state['contract']
    require(contract['matrix_sha256']==matrix_sha,'checkpoint matrix hash differs; provide its frozen matrix file')
    expected=expected_cases(matrix,contract);records=state['cases'];rows=[]
    unknown=set(records)-set(expected)
    for identifier,case in expected.items():
        record=records.get(identifier);run_dir=root/'runs'/identifier
        row={**case,'checkpoint':str(path),'matrix_status':state['status'],'status':'not_started',
            'measurement_origin':'synthetic_cpu_fixture' if state.get('synthetic_fixture') else 'actual_harness',
            'measurement_eligible':False,'run_dir':str(run_dir),'source_pass':None,'cleanup_pass':None,
            **{key:None for key in ('consumed_steps_total','measured_consumed_steps','sequence_sha256','N_tokens_total','L_tokens_total',
                'token_weighted_L_over_N','recomputed_tokens','training_step_wall_s','controller_window_step_wall_s','memory')},
            'stages':{name:{'total_ms':None,'requests':None,'coverage':None} for name in STAGES}}
        side='candidate' if case['group']=='host-on' else 'official'
        source=state.get('input_verification',{}).get('identities',{}).get(side,{})
        row.update(specforge_commit=source.get('specforge',{}).get('commit'),sglang_commit=source.get('sglang',{}).get('commit'))
        if record is None:
            other=identifier.replace('tcp-','rdma-') if case['transport']=='tcp' else identifier.replace('rdma-','tcp-')
            alternate=records.get(other)
            if case['transport']=='tcp':row['status']='conditional_not_selected' if alternate and alternate.get('status')=='passed' else 'conditional_pending'
            elif case['group']!='device-off' and alternate:
                try:row['native_unavailable']=unavailable_proof(alternate,case,state,root);row['status']='not_run_native_gdr_unavailable'
                except Exception as exc:row.update(status='invalid_data',error=str(exc))
        else:
            row['checkpoint_case_status']=record.get('status');row['checkpoint_stage']=record.get('stage')
            try:
                if record.get('case') is not None:require(record['case']=={**record['case'],**case},'checkpoint case metadata differs')
                status=record.get('status','incomplete');row['status']=status
                if status=='passed':passed_row(row,record,state,run_dir,seen_runs)
                elif status=='unavailable':
                    require(case['group']=='device-off' and case['transport']=='rdma','non-device unavailable case invalid')
                    row['native_unavailable']=unavailable_proof(record,case,state,root)
                    row.update(source_pass=True,cleanup_pass=True,cleanup_scope='native probe only; training unavailable')
                elif status in ('running','failed','incomplete'):
                    if (run_dir/'result.json').exists():
                        partial=load(run_dir/'result.json');row['harness_status']=partial.get('status');row['error']=partial.get('error')
                        if partial.get('status')=='failed':row['status']='failed'
                elif status!='planned':row['status']='incomplete'
            except Exception as exc:row.update(status='invalid_data',measurement_eligible=False,error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    for identifier in sorted(unknown):
        raw=records[identifier].get('case',{})
        rows.append({**raw,'id':identifier,'checkpoint':str(path),'matrix_status':state['status'],'status':'invalid_data',
            'measurement_eligible':False,'measurement_origin':'unvalidated','run_dir':str(root/'runs'/identifier),
            'error':'unknown/unselected checkpoint case identity; never treated as a device/transport measurement'})
    return rows,{'path':str(path),'status':state['status'],'contract':contract,'synthetic_fixture':bool(state.get('synthetic_fixture')),'error':state.get('error')}


def comparisons(rows):
    conditions=defaultdict(list);result=[]
    for row in rows:
        if row['measurement_eligible']:conditions[(row['checkpoint'],row['model'],row['workload'],row['repetition'])].append(row)
    for condition,members in conditions.items():
        groups={row['group']:row for row in members}
        if len(groups)!=len(members):
            result.append({'checkpoint':condition[0],'model':condition[1],'workload':condition[2],'repetition':condition[3],
                'status':'rejected_mismatch','mismatches':['ambiguous same-group observations'],'observed_step_ratio':None,'acceleration_claim_evaluated':False})
            continue
        for left,right in (('host-off','host-on'),('device-off','host-off'),('device-off','host-on')):
            if left not in groups or right not in groups:continue
            a,b=groups[left],groups[right];keys=('transport','comparison_contract_sha256','sequence_sha256','initial_draft_sha256','harness_sha256','helpers_sha256','measurement_origin')
            mismatches=[key for key in keys if a[key]!=b[key]]
            row={'checkpoint':condition[0],'model':condition[1],'workload':condition[2],'repetition':condition[3],
                'reference_group':left,'candidate_group':right,'transport':a['transport'],'acceleration_claim_evaluated':False}
            if mismatches:row.update(status='rejected_mismatch',mismatches=mismatches,observed_step_ratio=None)
            else:row.update(status='matched_observation',observed_step_ratio=a['training_step_wall_s']/b['training_step_wall_s'],
                reference_step_wall_s=a['training_step_wall_s'],candidate_step_wall_s=b['training_step_wall_s'])
            result.append(row)
    return result


def hot_statistics(rows):
    buckets=defaultdict(list)
    keys=('model','workload','transport','group','comparison_contract_sha256','sequence_sha256','initial_draft_sha256',
          'harness_sha256','helpers_sha256','specforge_commit','sglang_commit','measurement_origin')
    for row in rows:
        if row['measurement_eligible'] and row['workload']=='hot':buckets[tuple(row[key] for key in keys)].append(row)
    output=[]
    for identity,members in buckets.items():
        values=[row['training_step_wall_s'] for row in members];multiple=len(values)>=2
        output.append({**dict(zip(keys,identity)),'repetitions':len(values),
            'instances':[{'checkpoint':row['checkpoint'],'repetition':row['repetition'],'run_dir':row['run_dir']} for row in members],
            'mean_step_wall_s':statistics.fmean(values) if multiple else None,
            'sample_variance_step_wall_s2':statistics.variance(values) if multiple else None,
            'sample_stddev_step_wall_s':statistics.stdev(values) if multiple else None,
            'status':'observed_multiple_repetitions' if multiple else 'insufficient_repetitions',
            'acceleration_claim_evaluated':False})
    return output


def summarize(checkpoints,matrix_path):
    matrix=load(matrix_path);all_rows=[];inputs=[];errors=[];seen=set()
    for checkpoint in checkpoints:
        try:
            rows,meta=collect_checkpoint(checkpoint,matrix,file_sha(matrix_path),seen);all_rows.extend(rows);inputs.append(meta)
        except Exception as exc:errors.append({'checkpoint':str(checkpoint),'error':f'{type(exc).__name__}: {exc}'})
    pairs=comparisons(all_rows);counts=dict(Counter(row['status'] for row in all_rows))
    issues=errors+[{'case':row['id'],'checkpoint':row['checkpoint'],'error':row.get('error',row['status'])} for row in all_rows
        if row['status'] in ('invalid_data','failed','running','incomplete','not_started')]
    issues.extend(row for row in pairs if row['status']=='rejected_mismatch')
    real=any(row['measurement_eligible'] and row['measurement_origin']=='actual_harness' for row in all_rows)
    measured=sum(row['measurement_eligible'] for row in all_rows)
    return {'schema_version':1,'tool_sha256':file_sha(__file__),'generated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'matrix':str(Path(matrix_path).resolve()),
        'matrix_sha256':file_sha(matrix_path),'declared_matrix_scope':{
            'models':[row['id'] for row in matrix['models']],
            'base_primary_rdma_cases':sum(len(row['workloads']) for row in matrix['models'])*3,
            'base_conditional_tcp_host_cases':sum(len(row['workloads']) for row in matrix['models'])*2},'checkpoints':inputs,'rows':all_rows,'status_counts':counts,
        'validated_measured_cases':measured,'contains_actual_measurements':real,
        'comparisons':pairs,'hot_statistics':hot_statistics(all_rows),'validation_issues':issues,
        'acceleration_claim_evaluated':False,'performance_conclusion':None,
        'scope_note':'All nonmeasured/failed/conditional cases retained. Ratios are matched observations. Hot sample variance requires multiple independent run instances.',
        'exit_code':2 if issues else (0 if measured else 3)}


def write_outputs(report,prefix):
    prefix=Path(prefix)
    protected=[Path(report['matrix']).resolve()]+[Path(row['path']).resolve() for row in report['checkpoints']]
    for target in (prefix.with_suffix('.json').resolve(),prefix.with_suffix('.csv').resolve()):
        require(target not in protected and not any(target.is_relative_to(Path(row['run_dir']).resolve()) for row in report['rows']),'summary outputs must not overwrite raw case/checkpoint/matrix artifacts')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    identity=('model','workload','transport','group','repetition','status','measurement_origin','specforge_commit','sglang_commit',
        'consumed_steps_total','measured_consumed_steps','sequence_sha256','N_tokens_total','L_tokens_total','token_weighted_L_over_N',
        'recomputed_tokens','training_step_wall_s','controller_window_step_wall_s','source_pass','cleanup_pass')
    memory_columns=('whole_host_pss_peak_bytes','whole_host_rss_sum_peak_bytes','whole_gpu_physical_used_sum_peak_bytes',
        'measured_host_pss_peak_bytes','measured_gpu_physical_used_sum_peak_bytes','trainer_torch_peak_allocated_bytes','trainer_torch_peak_reserved_bytes')
    stage_columns=tuple(name+'_'+field for name in STAGES for field in ('total_ms','requests','coverage'))
    fields=identity+stage_columns+memory_columns+('server_stages_json','memory_coverage_json','error','checkpoint','run_dir')
    with prefix.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in report['rows']:
            out={key:row.get(key) for key in identity}
            for name in STAGES:
                for field in ('total_ms','requests','coverage'):out[name+'_'+field]=row.get('stages',{}).get(name,{}).get(field)
            memory=row.get('memory') or {};whole=memory.get('observed_whole_command',{});window=memory.get('observed_measured_window',{});allocator=memory.get('trainer_allocator_fit_including_warmup',{})
            out.update(whole_host_pss_peak_bytes=whole.get('host_pss_peak_bytes'),whole_host_rss_sum_peak_bytes=whole.get('host_rss_sum_peak_bytes'),
                whole_gpu_physical_used_sum_peak_bytes=whole.get('gpu_physical_used_sum_peak_bytes'),measured_host_pss_peak_bytes=window.get('host_pss_peak_bytes'),
                measured_gpu_physical_used_sum_peak_bytes=window.get('gpu_physical_used_sum_peak_bytes'),trainer_torch_peak_allocated_bytes=allocator.get('torch_peak_allocated_bytes'),
                trainer_torch_peak_reserved_bytes=allocator.get('torch_peak_reserved_bytes'),server_stages_json=row.get('server_stage_attribution'),
                memory_coverage_json={label:{key:section.get(key) for key in ('sample_count','host_memory_comparison_eligible','gpu_memory_comparison_eligible','host_pss_complete_samples','gpu_complete_samples_with_values')} for label,section in (('whole',whole),('measured',window))} if memory else None,error=row.get('error'),checkpoint=row['checkpoint'],run_dir=row['run_dir'])
            writer.writerow({key:('null' if value is None else json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value) for key,value in out.items()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',action='append',required=True,help='matrix-checkpoint.json, repeat for separate runs')
    parser.add_argument('--matrix',required=True,help='the exact matrix JSON identified by checkpoint hash')
    parser.add_argument('--output-prefix',required=True)
    args=parser.parse_args(argv);report=summarize(args.checkpoint,args.matrix);write_outputs(report,args.output_prefix)
    print(json.dumps({'status_counts':report['status_counts'],'validated_measured_cases':report['validated_measured_cases'],
        'contains_actual_measurements':report['contains_actual_measurements'],'validation_issues':len(report['validation_issues']),
        'acceleration_claim_evaluated':False,'json':str(Path(args.output_prefix).with_suffix('.json'))},sort_keys=True))
    return report['exit_code']


if __name__=='__main__':raise SystemExit(main())
