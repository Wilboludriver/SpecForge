"""Small synthetic schema controls; no benchmark execution or GPU dependencies."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('performance_summary',ROOT/'scripts/plana/summarize_performance.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def resources(pss=1000):
    return {'host_pss_peak_bytes':pss,'host_rss_sum_peak_bytes':2000,'host_locked_sum_peak_bytes':0,
        'gpu_physical_used_sum_peak_bytes':4000,'sample_count':2,'gpu_complete_samples_with_values':2,'errors':[],
        'host_memory_comparison_eligible':True,'gpu_memory_comparison_eligible':True,
        'host_metric_coverage':{name:{'complete_samples':2,'partial_samples':0,'unavailable_samples':0,'complete_fraction':1.,'partial_known_sum_peak_bytes':None}
            for name in ('pss_bytes','rss_bytes','locked_bytes')}}


def fixture(directory,repetitions=3):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    matrix={'models':[{'id':'qwen3-4b','workloads':['hot']}],'synthetic_measured_steps':2,'real_measured_steps':2}
    matrix_path=root/'matrix.json';save(matrix_path,matrix)
    identities={side:{'specforge':{'commit':sf},'sglang':{'commit':sg}} for side,sf,sg in (('official','1'*40,'3'*40),('candidate','2'*40,'4'*40))}
    state={'synthetic_fixture':True,'status':'complete','contract':{'matrix_sha256':tool.file_sha(matrix_path),'runner_sha256':'f'*64,
        'models':'all','workloads':'all','hot_repetitions':repetitions,'execute':True,'boot_id':'synthetic-boot'},
        'input_verification':{'identities':identities,'harness_sha256':'a'*64,'helpers_sha256':'b'*64},'cases':{}}
    for case in tool.expected_cases(matrix,state['contract']).values():
        if case['transport']=='tcp':continue
        identifier=case['id'];directory=root/'runs'/identifier;rep=case['repetition'];group=case['group']
        wall={'device-off':1.5,'host-off':2.,'host-on':1.}[group]+(rep-2)*.1
        fps=[tool.digest('warm0'),tool.digest('warm1'),tool.digest('measured0'),tool.digest('measured1')]
        consumed={'steps':[{'step':i+1,'sample_id':f'{identifier}:{i}','content_fingerprint':fp} for i,fp in enumerate(fps)],
            'ordered_fingerprints':fps,'ordered_sha256':tool.digest(fps),'measured_fingerprints':fps[2:],
            'measured_counter':dict(tool.Counter(fps[2:])),'full_order_matches':True,'measured_counter_matches':True}
        stages={name:{'total_ms':float(i+1) if group=='host-on' else None,'requests':2 if group=='host-on' else 0,'coverage':1. if group=='host-on' else 0.}
            for i,name in enumerate(tool.STAGES)}
        side='candidate' if group=='host-on' else 'official';N=2048;L=2046 if group=='host-on' else 0
        result={'group':group,'workload':'hot','transport':'rdma','status':'passed','passed':True,'source_unchanged':True,
            'consumed_order':consumed,'training':{'measured_optimizer_steps':2,'warmup_optimizer_steps':2,
                'optimizer_step_wall_s':wall,'controller_window_optimizer_step_wall_s':wall-.01,'primary_boundary':'synthetic actual callback boundary'},
            'actual_fit':{'completed_steps':4,'warmup_steps':2,'measured_steps':2,'fit_wall_s':wall*4+1,
                'measured_fit_boundary_wall_s':wall*2,'torch_peak_allocated_bytes':3000,'torch_peak_reserved_bytes':5000},
            'capture':{'requests':2,'tokens':N,'reused_tokens':L,'computed_tokens':N-L,'token_weighted_L_over_N':L/N,'stages':stages},
            'initial_draft':[{'sha256':'d'*64}], 'observed_resources':resources(),'measured_window_observed_resources':resources(900),
            'server_stage_attribution':{'publication_register_put_unregister_wall_ms':{'milliseconds':3.,'events':2}},
            'resource_sampler':{'thread_drained':True,'fatal_errors':[]},'device_publication_available':group=='device-off','fixture_repetition':rep}
        manifest={'comparison_contract':{'model':'synthetic-target','data':'same-ordered-input','transport':'rdma'},
            'harness_sha256':'a'*64,'helpers_sha256':'b'*64,'specforge':identities[side]['specforge'],
            'sglang':identities[side]['sglang'],'argv':['--run-dir',str(directory)]}
        save(directory/'result.json',result);save(directory/'manifest.json',manifest)
        save(directory/'source-after.json',{'specforge':manifest['specforge'],'sglang':manifest['sglang']})
        save(directory/'cleanup-final.json',{'remaining_keys':[],'inspected_keys':16,'control_status':404})
        save(directory/'owned-processes-final.json',[{'pid':123,'exit':0}])
        tree=root/'logs'/identifier/'owned-tree-final.json';save(tree,{'remaining':[],'remaining_direct_children':[],'coverage_errors':[]})
        save(directory/'device-proof.json',{'protocol':'rdma','gpu_put':True,'cuda_source':True,'cuda_register_status':0,'put_status':0,
            'fetch_equal':True,'keys_removed':True,'rdma_devices':'mlx5_2','forbidden_overrides':{},'local_memcpy_setting':'0','boot_id':'synthetic-boot'})
        state['cases'][identifier]={'case':case,'status':'passed','stage':'drained','exit':0,'resource_release':{'released':True},
            'owned_tree_report':str(tree),'result_sha256':tool.file_sha(directory/'result.json')}
    checkpoint=root/'matrix-checkpoint.json';save(checkpoint,state)
    return matrix_path,checkpoint


def alter(root,identifier,filename,change,refresh_hash=True):
    path=Path(root)/'runs'/identifier/filename;value=tool.load(path);change(value);save(path,value)
    if filename=='result.json' and refresh_hash:
        checkpoint=Path(root)/'matrix-checkpoint.json';state=tool.load(checkpoint)
        state['cases'][identifier]['result_sha256']=tool.file_sha(path);save(checkpoint,state)


def tcp_fallback(root):
    root=Path(root);checkpoint=root/'matrix-checkpoint.json';state=tool.load(checkpoint)
    source_id='qwen3-4b/hot/r1/rdma-device-off';source=root/'runs'/source_id
    result=tool.load(source/'result.json');result.update(status='unavailable',passed=False,error='DevicePublicationUnavailable: synthetic native register refused')
    save(source/'result.json',result);save(source/'device-proof.json',{'status':'unavailable','error':'synthetic unavailable proof'})
    save(source/'probe-cleanup.json',{'key_removal_authorized':True,'errors':[]})
    evidence={'status':'unavailable','run_dir':str(source),'proof':str(source/'device-proof.json'),'result_sha256':tool.file_sha(source/'result.json'),
        'boot_id':'synthetic-boot','hca':'mlx5_2','writer_gpu':'0'}
    state['cases'][source_id].update(status='unavailable',gdr=evidence,result_sha256=evidence['result_sha256'])
    for group in ('host-off','host-on'):
        old=f'qwen3-4b/hot/r1/rdma-{group}';new=old.replace('rdma-','tcp-');a=root/'runs'/old;b=root/'runs'/new;a.rename(b)
        record=state['cases'].pop(old);record['case'].update(id=new,transport='tcp');record['fallback_native_unavailable']=evidence
        result=tool.load(b/'result.json');result['transport']='tcp';save(b/'result.json',result)
        manifest=tool.load(b/'manifest.json');manifest['comparison_contract']['transport']='tcp';manifest['argv']=['--run-dir',str(b)];save(b/'manifest.json',manifest)
        record['result_sha256']=tool.file_sha(b/'result.json');state['cases'][new]=record
    save(checkpoint,state)


class SummaryTests(unittest.TestCase):
    def test_matched_three_hot_repetitions_and_distinct_memory_boundaries(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp);report=tool.summarize([checkpoint],matrix)
            self.assertEqual(report['exit_code'],0);self.assertEqual(report['validated_measured_cases'],9)
            self.assertEqual(report['status_counts']['conditional_not_selected'],6)
            self.assertFalse(report['contains_actual_measurements']);self.assertFalse(report['acceleration_claim_evaluated'])
            self.assertIsNone(report['performance_conclusion']);self.assertEqual(len(report['comparisons']),9)
            host=next(row for row in report['hot_statistics'] if row['group']=='host-off')
            self.assertAlmostEqual(host['mean_step_wall_s'],2.);self.assertAlmostEqual(host['sample_variance_step_wall_s2'],.01)
            row=next(row for row in report['rows'] if row['status']=='passed')
            self.assertEqual(row['memory']['observed_whole_command']['host_pss_peak_bytes'],1000)
            self.assertEqual(row['memory']['observed_measured_window']['host_pss_peak_bytes'],900)
            self.assertEqual(row['memory']['trainer_allocator_fit_including_warmup']['torch_peak_allocated_bytes'],3000)
            self.assertEqual(row['measured_consumed_steps'],2);self.assertEqual(row['consumed_steps_total'],4)

    def test_single_hot_run_has_no_mean_or_variance(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp,1);report=tool.summarize([checkpoint],matrix)
            self.assertTrue(all(row['mean_step_wall_s'] is None and row['sample_variance_step_wall_s2'] is None for row in report['hot_statistics']))

    def test_missing_or_inconsistent_measurements_never_pass(self):
        defects={
            'missing_consumed':lambda r:r.pop('consumed_order'),
            'missing_step':lambda r:r['consumed_order']['steps'].pop(),
            'wrong_hash':lambda r:r['consumed_order'].update(ordered_sha256='bad'),
            'wrong_counter':lambda r:r['consumed_order']['measured_counter'].clear(),
            'count':lambda r:r['capture'].update(requests=1),
            'tokens':lambda r:r['capture'].update(computed_tokens=0),
            'ratio':lambda r:r['capture'].update(token_weighted_L_over_N=.99),
            'missing_stage':lambda r:r['capture']['stages'].pop('assembly_ms'),
            'missing_as_zero':lambda r:r['capture']['stages']['forward_ms'].update(total_ms=0),
            'wrong_coverage':lambda r:r['capture']['stages']['forward_ms'].update(coverage=.5),
            'no_source':lambda r:r.update(source_unchanged=False),
            'no_fit':lambda r:r.pop('actual_fit'),
            'no_resources':lambda r:r.pop('observed_resources'),
            'bad_allocator':lambda r:r['actual_fit'].update(torch_peak_reserved_bytes=1),
            'sampler_failure':lambda r:r['resource_sampler'].update(fatal_errors=['failed']),
            'nan_server_stage':lambda r:r['server_stage_attribution']['publication_register_put_unregister_wall_ms'].update(milliseconds=float('nan')),
            'missing_server_stage_total':lambda r:r['server_stage_attribution']['publication_register_put_unregister_wall_ms'].pop('milliseconds')}
        identifier='qwen3-4b/hot/r1/rdma-host-off'
        for name,change in defects.items():
            with self.subTest(defect=name),tempfile.TemporaryDirectory(dir=HERE) as tmp:
                matrix,checkpoint=fixture(tmp,1);alter(tmp,identifier,'result.json',change)
                report=tool.summarize([checkpoint],matrix)
                self.assertEqual(report['exit_code'],2)
                self.assertEqual(next(r for r in report['rows'] if r['id']==identifier)['status'],'invalid_data')

    def test_contract_sequence_draft_and_transport_mismatch_comparisons_rejected(self):
        for defect in ('contract','sequence','draft','transport'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory(dir=HERE) as tmp:
                matrix,checkpoint=fixture(tmp,1);identifier='qwen3-4b/hot/r1/rdma-host-on'
                if defect=='contract':alter(tmp,identifier,'manifest.json',lambda m:m['comparison_contract'].update(data='other'))
                elif defect=='draft':alter(tmp,identifier,'result.json',lambda r:r['initial_draft'][0].update(sha256='e'*64))
                elif defect=='transport':
                    tcp_fallback(tmp)
                    # Force a genuine available device beside valid fallback hosts: no cross-transport ratio can be emitted.
                    # Direct comparison negative control below isolates the transport rule.
                    rows=[{'checkpoint':'c','model':'m','workload':'hot','repetition':1,'group':group,
                        'measurement_eligible':True,'transport':transport,'comparison_contract_sha256':'same','sequence_sha256':'same',
                        'initial_draft_sha256':'same','harness_sha256':'same','helpers_sha256':'same','measurement_origin':'synthetic',
                        'training_step_wall_s':wall} for group,transport,wall in (('host-off','rdma',2),('host-on','tcp',1))]
                    pair=tool.comparisons(rows)[0];self.assertEqual(pair['status'],'rejected_mismatch');self.assertIsNone(pair['observed_step_ratio']);continue
                else:
                    def different(r):
                        a=r['consumed_order'];a['ordered_fingerprints']=['other']*4
                        for row in a['steps']:row['content_fingerprint']='other'
                        a.update(ordered_sha256=tool.digest(a['ordered_fingerprints']),measured_fingerprints=['other']*2,measured_counter={'other':2})
                    alter(tmp,identifier,'result.json',different)
                report=tool.summarize([checkpoint],matrix)
                self.assertEqual(report['exit_code'],2)
                self.assertTrue(any(pair['status']=='rejected_mismatch' for pair in report['comparisons']))

    def test_unavailable_preserved_and_tcp_hosts_have_new_contract(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp,1);tcp_fallback(tmp);report=tool.summarize([checkpoint],matrix)
            self.assertEqual(report['exit_code'],0);self.assertEqual(report['status_counts']['unavailable'],1)
            self.assertEqual(report['status_counts']['not_run_native_gdr_unavailable'],2)
            device=next(r for r in report['rows'] if r['group']=='device-off')
            self.assertIsNone(device['training_step_wall_s']);self.assertIsNone(device['N_tokens_total'])
            self.assertEqual(len(report['comparisons']),1);self.assertEqual(report['comparisons'][0]['transport'],'tcp')

    def test_unrelated_failure_cannot_masquerade_as_native_unavailable(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp,1);tcp_fallback(tmp)
            identifier='qwen3-4b/hot/r1/rdma-device-off'
            alter(tmp,identifier,'result.json',lambda r:r.update(error='RuntimeError: model startup'))
            report=tool.summarize([checkpoint],matrix)
            self.assertEqual(report['exit_code'],2);self.assertFalse(any(row['measurement_eligible'] for row in report['rows']))

    def test_failed_incomplete_and_not_started_are_not_suppressed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp,1);state=tool.load(checkpoint)
            off='qwen3-4b/hot/r1/rdma-host-off';on=off.replace('host-off','host-on')
            state['cases'][off]['status']='running';state['status']='failed';state.pop('source_after',None)
            state['cases'].pop(on);save(checkpoint,state)
            alter(tmp,off,'result.json',lambda r:r.update(status='failed',passed=False,error='fixture failure'))
            report=tool.summarize([checkpoint],matrix)
            self.assertEqual(report['exit_code'],2);self.assertEqual(report['status_counts']['failed'],1)
            self.assertEqual(report['status_counts']['not_started'],1)
            self.assertEqual(len(report['rows']),5)

    def test_missing_cleanup_or_changed_source_rejects_pass(self):
        for filename,change in (('cleanup-final.json',lambda r:r.update(remaining_keys=['leak'])),
                                ('source-after.json',lambda r:r['sglang'].update(commit='wrong'))):
            with self.subTest(file=filename),tempfile.TemporaryDirectory(dir=HERE) as tmp:
                matrix,checkpoint=fixture(tmp,1);alter(tmp,'qwen3-4b/hot/r1/rdma-host-off',filename,change)
                self.assertEqual(tool.summarize([checkpoint],matrix)['exit_code'],2)

    def test_null_resources_and_csv_null_stages_remain_null(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp,1)
            def unavailable(r):
                for key in ('observed_resources','measured_window_observed_resources'):
                    section=r[key];section['host_pss_peak_bytes']=None
                    section['host_metric_coverage']['pss_bytes']['complete_samples']=0
                    section['host_memory_comparison_eligible']=False
            alter(tmp,'qwen3-4b/hot/r1/rdma-host-off','result.json',unavailable)
            report=tool.summarize([checkpoint],matrix);self.assertEqual(report['exit_code'],0)
            prefix=Path(tmp)/'table';tool.write_outputs(report,prefix)
            with prefix.with_suffix('.csv').open() as stream:
                row=next(row for row in csv.DictReader(stream) if row['group']=='host-off' and row['status']=='passed')
            self.assertEqual(row['forward_ms_total_ms'],'null');self.assertEqual(row['whole_host_pss_peak_bytes'],'null')
            self.assertEqual(row['forward_ms_coverage'],'0.0')

    def test_result_hash_duplicate_input_and_matrix_identity_are_checked(self):
        for defect in ('result_hash','duplicate','matrix'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory(dir=HERE) as tmp:
                matrix,checkpoint=fixture(tmp,1)
                if defect=='result_hash':alter(tmp,'qwen3-4b/hot/r1/rdma-host-off','result.json',lambda r:r.update(ended_at='modified'),False)
                if defect=='matrix':save(matrix,{**tool.load(matrix),'changed':True})
                report=tool.summarize([checkpoint,checkpoint] if defect=='duplicate' else [checkpoint],matrix)
                self.assertEqual(report['exit_code'],2)

    def test_unknown_tcp_device_and_output_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp,1);state=tool.load(checkpoint)
            state['cases']['qwen3-4b/hot/r1/tcp-device-off']={'status':'passed','case':{'model':'qwen3-4b','workload':'hot','group':'device-off','transport':'tcp','repetition':1}}
            save(checkpoint,state);report=tool.summarize([checkpoint],matrix)
            self.assertEqual(report['exit_code'],2)
            self.assertEqual(next(row for row in report['rows'] if row['id'].endswith('tcp-device-off'))['status'],'invalid_data')
            with self.assertRaises(ValueError):tool.write_outputs(report,Path(tmp)/'runs/qwen3-4b/hot/r1/rdma-host-off/result')
            with self.assertRaises(ValueError):tool.write_outputs(report,matrix.with_suffix(''))

    def test_plans_produce_no_measurements_or_statistics(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            matrix,checkpoint=fixture(tmp,1);state=tool.load(checkpoint);state['contract']['execute']=False;state['status']='planned'
            for record in state['cases'].values():record['status']='planned'
            save(checkpoint,state);report=tool.summarize([checkpoint],matrix)
            self.assertEqual(report['exit_code'],3);self.assertEqual(report['validated_measured_cases'],0)
            self.assertEqual(report['hot_statistics'],[]);self.assertFalse(report['contains_actual_measurements'])


if __name__=='__main__':unittest.main(verbosity=2)
