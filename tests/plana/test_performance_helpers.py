"""CPU-only formal performance planning/accounting contract tests."""
import copy
import importlib.util
import os
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace
from unittest import mock
import json
from pathlib import Path
import tempfile
import unittest

FILE=Path(__file__).resolve().parents[2]/'scripts/plana/performance_helpers.py'
spec=importlib.util.spec_from_file_location('plana_performance_helpers',FILE)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def row(index=0):
    return {'id':str(index),'input_ids':list(range(1024)), 'loss_mask':[0]*512+[1]*512}


def config():
    return {'model':{'target_model_path':'target','draft_checkpoint_path':'draft',
                     'sglang_extra_args':['--enable-deterministic-inference']},
        'data':{'prompts_path':'original','cache_dir':'original'},
        'training':{'learning_rate':.0001},'tracking':{'report_to':'none'},
        'deployment':{'trainer':{'nproc_per_node':1},'disaggregated':{'control_dir':'old','consumer_state_dir':'old',
            'managed_local':{'mooncake':{},'capture_servers':[{'port':30000,'cuda_visible_devices':['0']}],
                             'trainer_cuda_visible_devices':['1']}}}}


class WorkloadTests(unittest.TestCase):
    def build(self,kind='hot',**kw):
        values=dict(warmup_steps=8,measured_steps=64,chunk_tokens=256,row_bytes=30*1024,cache_bytes=64<<20)
        values.update(kw)
        return helpers.build_workload(kind,[row()], [row(i) for i in range(64)],**values)
    def test_feature_keys_are_causal_and_masks_excluded(self):
        first=row();other=row();other['loss_mask']=[1]*1024
        same=helpers.complete_prefix_footprint([first,other],256,2)
        different=row();different['input_ids'][0]=9999
        fork=helpers.complete_prefix_footprint([first,different],256,2)
        self.assertEqual(same['unique_complete_chunks'],4)
        self.assertEqual(fork['unique_complete_chunks'],8)
    def test_over_capacity_is_proven_and_hot_fits_same_budget(self):
        cold=self.build('over-capacity');hot=self.build('hot')
        self.assertGreater(cold['logical_footprint']['feature_bytes'],64<<20)
        self.assertLessEqual(hot['logical_footprint']['feature_bytes'],64<<20)
        self.assertEqual(len(cold['training']),72)
        self.assertTrue(cold['prewarm'])
    def test_real64_preserves_every_original_id_and_mask(self):
        actual=self.build('real64')
        self.assertEqual([x['input_ids'] for x in actual['catalog']],[row(i)['input_ids'] for i in range(64)])
        self.assertEqual([x['loss_mask'] for x in actual['catalog']],[row(i)['loss_mask'] for i in range(64)])
        with self.assertRaises(ValueError):self.build('real64',measured_steps=32)
    def test_cold_has_no_server_prewarm_and_no_prefix_collisions(self):
        actual=self.build('cold')
        self.assertEqual(actual['prewarm'],[])
        self.assertEqual(len({x['input_ids'][0] for x in actual['training']}),72)
    def test_bad_hot_budget_refused(self):
        with self.assertRaises(ValueError):self.build('hot',cache_bytes=1024)


class AccountingTests(unittest.TestCase):
    def test_token_weighted_reuse_and_missing_stage_is_not_zero(self):
        result=helpers.aggregate_capture([
            {'sample_id':'a','attempt_id':'1','N':100,'L':90,'span':[90,100],'timings':{'forward_ms':3.}},
            {'sample_id':'b','attempt_id':'2','N':900,'L':0,'span':[0,900],'timings':{}}])
        self.assertEqual(result['token_weighted_L_over_N'],.09)
        self.assertEqual(result['computed_tokens'],910)
        self.assertIsNone(result['stages']['assembly_ms']['total_ms'])
        self.assertEqual(result['stages']['forward_ms']['coverage'],.5)
    def test_invalid_span_and_duplicate_attempt_fail(self):
        record={'sample_id':'a','attempt_id':'1','N':100,'L':100,'span':[100,100]}
        with self.assertRaises(ValueError):helpers.aggregate_capture([record])
        record.update(L=0,span=[0,100])
        with self.assertRaises(ValueError):helpers.aggregate_capture([record,record])
    def test_real_step_accounting_excludes_warmup(self):
        rows=[{'step':i,'metrics':{'perf/optimizer_step_time_s':float(i)}} for i in range(1,13)]
        actual=helpers.aggregate_training(rows,warmup_steps=4,measured_steps=8)
        self.assertEqual(actual['controller_window_optimizer_step_wall_s'],8.5)
        with self.assertRaises(ValueError):helpers.aggregate_training(rows[:-1],warmup_steps=4,measured_steps=8)
    def test_actual_step_context_not_nonexistent_global_step_keyword(self):
        from types import SimpleNamespace
        self.assertEqual(helpers.optimizer_step_number((),{'ctx':SimpleNamespace(global_step=9)}),10)
        self.assertEqual(helpers.optimizer_step_number((SimpleNamespace(global_step=12),),{}),13)
        with self.assertRaises(ValueError):helpers.optimizer_step_number((),{'global_step':9})
    def test_mask_fingerprint_numeric_representation(self):
        self.assertEqual(helpers.content_fingerprint([1,2],[0,1]),helpers.content_fingerprint([1,2],[0.,1.]))
        with self.assertRaises(ValueError):helpers.content_fingerprint([1],[.5])


class BoundaryTests(unittest.TestCase):
    def test_tcp_never_qualifies_as_device(self):
        with self.assertRaises(ValueError):helpers.isolate_config(config(),run_dir='/tmp/run',
            workload={'training':[row()]*72},group='device-off',transport='tcp',cache_bytes=64<<20,chunk_tokens=256,rdma_devices='mlx5_0')
    def test_actual_gpu_proof_requires_register_fetch_remove_and_no_override(self):
        proof=dict(protocol='rdma',gpu_put=True,cuda_source=True,cuda_register_status=0,put_status=0,
                   fetch_equal=True,keys_removed=True,rdma_devices='mlx5_0',forbidden_overrides={},local_memcpy_setting='0')
        self.assertTrue(helpers.validate_rdma_proof(proof,expected_devices='mlx5_0'))
        for field,value in (('protocol','tcp'),('cuda_source',False),('keys_removed',False),('cuda_register_status',1),('forbidden_overrides',{'MC_FORCE_TCP':'1'})):
            with self.assertRaises(ValueError):helpers.validate_rdma_proof(dict(proof,**{field:value}))
    def test_worker_wrapper_preserves_torchrun_and_config(self):
        argv=['python','-m','torch.distributed.run','--nproc_per_node','1','-m','specforge.cli','train','--config','a.yaml','--role','consumer']
        result=helpers.rewrite_worker_argv(argv,'/tools/run_performance.py')
        self.assertEqual(result[:5],tuple(argv[:5]))
        self.assertEqual(result[-5:],tuple(argv[-5:]))
        with self.assertRaises(ValueError):helpers.rewrite_worker_argv(['arbitrary-server'],'script')
    def test_explicit_prompt_seed_preserved_with_forced_training_seed(self):
        original=config();original['training'].update(seed=99,prompt_seed=17)
        isolated=helpers.isolate_config(original,run_dir='/tmp/seed',workload={'training':[row()]*72},
            group='host-off',transport='tcp',cache_bytes=64<<20,chunk_tokens=256,rdma_devices=None)
        self.assertEqual(isolated['training']['seed'],42)
        self.assertEqual(isolated['training']['prompt_seed'],17)

    def test_deterministic_flag_and_model_identity_survive_mode_switch(self):
        hot={'training':[row()]*72}
        off=helpers.isolate_config(config(),run_dir='/tmp/off',workload=hot,group='host-off',transport='tcp',cache_bytes=64<<20,chunk_tokens=256,rdma_devices=None)
        on=helpers.isolate_config(config(),run_dir='/tmp/on',workload=hot,group='host-on',transport='tcp',cache_bytes=64<<20,chunk_tokens=256,rdma_devices=None)
        self.assertIn('--enable-deterministic-inference',on['model']['sglang_extra_args'])
        self.assertEqual(helpers.comparison_contract(off,'same',{},64<<20,256),helpers.comparison_contract(on,'same',{},64<<20,256))
        self.assertEqual(on['runtime']['producer_concurrency'],1)
        self.assertNotIn('producer_concurrency',on['deployment']['disaggregated'])
    def test_compare_rejects_configuration_change_and_reports_slowdown_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=[]
            for group,wall in (('host-off',1.),('host-on',2.)):
                root=Path(tmp)/group;root.mkdir();roots.append(root)
                (root/'manifest.json').write_text(json.dumps({'comparison_contract':{'data':'same'}}))
                (root/'result.json').write_text(json.dumps({'group':group,'status':'passed','initial_draft':[{'sha256':'same'}],
                    'consumed_order':{'ordered_sha256':'same-order','full_order_matches':True,'measured_counter_matches':True},
                    'training':{'optimizer_step_wall_s':wall}}))
            report=helpers.compare_runs(roots)
            self.assertEqual(report['host_off_over_host_on_step_ratio'],.5)
            self.assertFalse(report['measured_lower_step_time'])
            self.assertFalse(report['acceleration_claim_evaluated'])
            (roots[1]/'manifest.json').write_text(json.dumps({'comparison_contract':{'data':'different'}}))
            with self.assertRaises(ValueError):helpers.compare_runs(roots)


class FullSampleValidationTests(unittest.TestCase):
    def test_full_ref_metadata_and_attention_mask_are_checked(self):
        import sys
        from types import SimpleNamespace
        import torch
        sys.path.insert(0,str(FILE.parent))
        import run_performance as runner
        features={'input_ids':torch.tensor([[1,2,3,4]]),'loss_mask':torch.tensor([[0,1,1,1]]),
                  'attention_mask':torch.ones((1,4),dtype=torch.int64),
                  'hidden_states':torch.zeros((1,4,4)),'target_last_hidden_states':torch.zeros((1,4,2))}
        specs={name:SimpleNamespace(shape=tuple(value.shape),dtype=str(value.dtype).removeprefix('torch.')) for name,value in features.items()}
        ref=SimpleNamespace(num_tokens=4,metadata={'generation':1},feature_specs=specs)
        workload={'fingerprints':[helpers.content_fingerprint([1,2,3,4],[0,1,1,1])]}
        schema={'hidden_size':2,'layers':[1,2],'dtype':'float32'}
        runner._validate_sample(ref,features,workload,schema)
        features['attention_mask'][0,1]=0
        with self.assertRaises(AssertionError):runner._validate_sample(ref,features,workload,schema)
        features['attention_mask'][0,1]=1
        specs['attention_mask'].shape=(1,3)
        with self.assertRaises(AssertionError):runner._validate_sample(ref,features,workload,schema)



def runner_module():
    sys.path.insert(0,str(FILE.parent))
    import run_performance
    return run_performance


def project_root():
    return next(path for path in FILE.parents if (path/'NEXT_SESSION_PROMPT.txt').exists())


def official_module(name, relative):
    path=project_root()/'src/SpecForge-official'/relative
    if not path.is_file():
        raise AssertionError(f'immutable official source required: {path}')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class ActualProducerOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch=official_module('plana_perf_actual_official_launch','specforge/launch.py')

    def workload(self, rows=None):
        if rows is None:
            rows=[dict(row(i),input_ids=[2000+i]+list(range(1,1024))) for i in range(64)]
        return helpers.build_workload('real64',[row()],rows,warmup_steps=8,measured_steps=64,
            chunk_tokens=256,row_bytes=30720,cache_bytes=64<<20)

    def records(self, rows):
        consumed=[];fetches=[]
        for step,item in enumerate(rows,1):
            sid=f'actual-{step:04d}'
            consumed.append({'step':step,'optimizer_stepped':True,'sample_ids':[sid]})
            fetches.append({'sample_id':sid,'content_fingerprint':helpers.content_fingerprint(item['input_ids'],item['loss_mask'])})
        return consumed,fetches

    def test_actual_immutable_shuffle_reproduces57_then_inverse_covers_all64_in_order(self):
        workload=self.workload()
        naive=[workload['training'][index] for index in self.launch._epoch_prompt_indices(workload['training'],0,seed=42)]
        self.assertEqual(len({helpers.content_fingerprint(r['input_ids'],r['loss_mask']) for r in naive[8:]}),57)
        intended=copy.deepcopy(workload['training'])
        helpers.serialize_for_epoch_order(workload,self.launch._epoch_prompt_indices,prompt_seed=42)
        emitted=[workload['training'][i] for i in self.launch._epoch_prompt_indices(workload['training'],0,seed=42)]
        self.assertEqual(emitted,intended)
        report=helpers.audit_consumed_order(*self.records(emitted),workload)
        self.assertEqual(len(report['measured_counter']),64)
        self.assertEqual(len(report['steps']),72)
        self.assertNotEqual(workload['data_sha256'],workload['intended_data_sha256'])
        with self.assertRaises(ValueError):helpers.serialize_for_epoch_order(workload,self.launch._epoch_prompt_indices,prompt_seed=42)

    def test_actual_real64_full_ids_masks_and_counter_preserved(self):
        rows=helpers.load_rows(project_root()/'implementation/data/real-qwen3-4b.jsonl')
        workload=self.workload(rows)
        helpers.serialize_for_epoch_order(workload,self.launch._epoch_prompt_indices,prompt_seed=17)
        emitted=[workload['training'][i] for i in self.launch._epoch_prompt_indices(workload['training'],0,seed=17)]
        report=helpers.audit_consumed_order(*self.records(emitted),workload)
        expected=Counter(helpers.content_fingerprint(r['input_ids'],r['loss_mask']) for r in rows)
        self.assertEqual(report['measured_counter'],dict(expected))
        self.assertEqual(sum(report['measured_counter'].values()),64)

    def test_legitimate_duplicate_originals_preserve_multiplicity(self):
        workload=self.workload();workload['training'][16]=copy.deepcopy(workload['training'][17])
        helpers.serialize_for_epoch_order(workload,self.launch._epoch_prompt_indices,prompt_seed=42)
        report=helpers.audit_consumed_order(*self.records(workload['intended_training']),workload)
        self.assertEqual(max(report['measured_counter'].values()),2)
        self.assertEqual(sum(report['measured_counter'].values()),64)

    def test_reordering_wrong_seed_or_changed_mask_is_rejected(self):
        workload=self.workload();helpers.serialize_for_epoch_order(workload,self.launch._epoch_prompt_indices,prompt_seed=42)
        wrong=[workload['training'][i] for i in self.launch._epoch_prompt_indices(workload['training'],0,seed=43)]
        with self.assertRaises(ValueError):helpers.audit_consumed_order(*self.records(wrong),workload)
        reordered=copy.deepcopy(workload['intended_training']);reordered[8],reordered[9]=reordered[9],reordered[8]
        with self.assertRaisesRegex(ValueError,'order'):helpers.audit_consumed_order(*self.records(reordered),workload)
        changed=copy.deepcopy(workload['intended_training']);changed[8]['loss_mask'][0]=1
        with self.assertRaisesRegex(ValueError,'multiset'):helpers.audit_consumed_order(*self.records(changed),workload)

    def test_missing_duplicated_steps_or_changed_fetch_fail_closed(self):
        workload=self.workload();helpers.serialize_for_epoch_order(workload,self.launch._epoch_prompt_indices,prompt_seed=42)
        consumed,fetches=self.records(workload['intended_training'])
        with self.assertRaises(ValueError):helpers.audit_consumed_order(consumed[:-1],fetches,workload)
        with self.assertRaises(ValueError):helpers.audit_consumed_order(consumed+[consumed[0]],fetches,workload)
        with self.assertRaises(ValueError):helpers.audit_consumed_order(consumed,fetches+[dict(fetches[0],content_fingerprint='changed')],workload)

    def test_compare_refuses_different_actual_order_despite_matching_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=[]
            for group,digest in (('host-off','order-A'),('host-on','order-B')):
                root=Path(tmp)/group;root.mkdir();roots.append(root)
                helpers.write_json(root/'manifest.json',{'comparison_contract':{'same':True}})
                helpers.write_json(root/'result.json',{'group':group,'status':'passed','initial_draft':[{'sha256':'same'}],
                    'training':{'optimizer_step_wall_s':1.},'consumed_order':{'ordered_sha256':digest,
                    'full_order_matches':True,'measured_counter_matches':True}})
            with self.assertRaisesRegex(ValueError,'sample order'):helpers.compare_runs(roots)


class OwnedProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch=official_module('plana_perf_actual_official_launch_plan','specforge/launch_plan.py')
        cls.runner=runner_module()

    def test_actual_supervisor_interrupted_or_timedout_probe_drains_all_owned_sessions(self):
        # The actual immutable supervisor is invoked; only process I/O/preflight are faked.
        for failure in (KeyboardInterrupt('injected interruption'),subprocess.TimeoutExpired('probe',.25)):
            with self.subTest(failure=type(failure).__name__),tempfile.TemporaryDirectory() as tmp:
                processes=[];terminate_calls=[];probe_waits=[]
                class FakeProcess:
                    def __init__(self,argv,**kwargs):
                        self.pid=700000+len(processes);self.argv=argv;self.status=None
                        self.detached=kwargs.get('start_new_session',False);processes.append(self)
                    def poll(self):return self.status
                    def wait(self,timeout=None):
                        if '--rdma-probe' in self.argv:
                            probe_waits.append(timeout);raise failure
                        return self.status
                def terminate(items,**kwargs):
                    terminate_calls.extend(p.pid for p in items)
                    for process in items:process.status=-15
                def stop(started,**kwargs):terminate([p for _,p in started],**kwargs)
                command=lambda label,argv:SimpleNamespace(label=label,argv=tuple(argv),env={})
                service=lambda label,phase:SimpleNamespace(command=command(label,('fake',label)),phase=phase,
                    log_path=str(Path(tmp)/'logs'/f'{label}.log'))
                plan=SimpleNamespace(kind='managed_supervisor',managed_root=tmp,services=(service('mooncake',0),service('capture-server-0',1)),
                    commands=(),shutdown_grace_s=.01)
                def ready(service,process,started):
                    if service.phase==1:
                        self.runner.run_owned_probe(FakeProcess,['python','--rdma-probe'],env={},terminate=terminate,timeout_s=.25,grace_s=.01)
                sampler=mock.Mock();sampler.stop.return_value={'thread_drained':True}
                with mock.patch.object(self.launch,'_terminate_processes',side_effect=terminate),mock.patch.object(self.launch,'_stop_services',side_effect=stop):
                    try:
                        with self.assertRaises(type(failure)):
                            self.launch.run_commands(plan,popen=FakeProcess,managed_preflight=lambda _:None,readiness_waiter=ready)
                    finally:
                        self.runner.finish_owned_run(self.launch,stop,[],processes,sampler,tmp,grace_s=.01)
                self.assertEqual(len(processes),3)
                self.assertTrue(all(p.detached for p in processes))
                self.assertTrue(all(p.poll() is not None for p in processes))
                self.assertEqual(probe_waits,[.25])
                self.assertNotIn(os.getpid(),terminate_calls)
                sampler.stop.assert_called_once()

    def test_final_cleanup_attempts_live_extra_even_if_service_stop_raises(self):
        extra=SimpleNamespace(pid=700003,poll=lambda:None)
        sampler=mock.Mock()
        def terminate(items,**kwargs):
            for p in items:p.poll=lambda:-15
        launch=SimpleNamespace(_terminate_processes=mock.Mock(side_effect=terminate))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'services'):
                self.runner.finish_owned_run(launch,mock.Mock(side_effect=PermissionError('stop')),[('service',extra)],
                    [extra],sampler,tmp,grace_s=.01)
            launch._terminate_processes.assert_called_once();sampler.stop.assert_called_once()
            self.assertEqual(json.loads((Path(tmp)/'owned-processes-final.json').read_text())[0]['exit'],-15)

    def test_probe_finally_closes_sink_store_and_both_keys_after_failure(self):
        calls=[]
        sink=SimpleNamespace(_executor=SimpleNamespace(shutdown=lambda **_:calls.append('sink-drained')))
        store=SimpleNamespace(release=lambda _:calls.append('reader-released'),drain_sample_removals=lambda _:calls.append('removal-drained'),
            _tkey=lambda sid,g,name:name,_store_remove=lambda key,**_:calls.append(key) or True,
            _store_exists=lambda _:False,close=lambda:calls.append('store-closed'))
        self.runner._cleanup_probe_resources(sink,store,'handle','probe',('aux','last'),writer_finished=True,read_started=True,read_finished=True)
        self.assertEqual(calls,['sink-drained','reader-released','removal-drained','aux','last','store-closed'])
        calls.clear();sink._executor.shutdown=mock.Mock(side_effect=RuntimeError('drain'))
        with self.assertRaisesRegex(RuntimeError,'sink drain'):
            self.runner._cleanup_probe_resources(sink,store,None,'probe',('aux','last'),writer_finished=True)
        self.assertNotIn('aux',calls);self.assertNotIn('last',calls);self.assertIn('store-closed',calls)
        calls.clear();sink._executor.shutdown=lambda **_:calls.append('sink-drained')
        with self.assertRaisesRegex(RuntimeError,'drain unproven'):
            self.runner._cleanup_probe_resources(sink,store,None,'probe',('aux','last'),writer_finished=True,read_started=True,read_finished=False)
        self.assertEqual(calls,['sink-drained','store-closed'])

    def test_failed_native_writer_close_never_authorizes_key_removal(self):
        calls=[]
        sink=SimpleNamespace(_executor=SimpleNamespace(shutdown=lambda **_:None),_store=SimpleNamespace(close=lambda:7))
        store=SimpleNamespace(close=lambda:calls.append('receiver-closed'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'probe-cleanup.json'
            with self.assertRaisesRegex(RuntimeError,'native writer transport close failed'):
                self.runner._cleanup_probe_resources(sink,store,None,'probe',('aux','last'),
                    report_path=path,store_id='owned-proof')
            report=json.loads(path.read_text())
            self.assertFalse(report['key_removal_authorized'])
            self.assertEqual(report['physical_keys'],['owned-proof/probe/g1/aux','owned-proof/probe/g1/last'])
            self.assertEqual(report['store_id'],'owned-proof')
            self.assertEqual(calls,['receiver-closed'])

    def test_probe_nonfinite_timeout_rejected_before_spawn(self):
        for timeout in (float('inf'),float('nan'),0.,-1.):
            spawn=mock.Mock()
            with self.assertRaises(ValueError):
                self.runner.run_owned_probe(spawn,['probe'],env={},terminate=mock.Mock(),timeout_s=timeout,grace_s=.01)
            spawn.assert_not_called()

    def test_parent_rdma_environment_is_explicit_and_restored(self):
        with mock.patch.dict(os.environ,{'MC_STORE_MEMCPY':'1'}):
            with self.runner.scoped_environment({'MC_STORE_MEMCPY':'0'}):self.assertEqual(os.environ['MC_STORE_MEMCPY'],'0')
            self.assertEqual(os.environ['MC_STORE_MEMCPY'],'1')


class ResourceCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runner=runner_module()

    def resources(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resources.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return self.runner._load_resource_samples(path)

    def sample(self,host,**kwargs):
        return {'monotonic':1.,'host':host,'gpu':[],'sample_interval_s':.1,**kwargs}

    def test_unavailable_peaks_are_null_and_not_comparable(self):
        report=self.resources([self.sample([{'pss_bytes':None,'rss_bytes':None,'locked_bytes':None}])])
        for field in ('host_pss_peak_bytes','host_rss_sum_peak_bytes','host_locked_sum_peak_bytes','gpu_physical_used_sum_peak_bytes'):
            self.assertIsNone(report[field])
        self.assertEqual(report['host_metric_coverage']['pss_bytes']['unavailable_samples'],1)
        self.assertFalse(report['host_memory_comparison_eligible'])

    def test_partial_known_sum_separate_from_complete_peak(self):
        a={'pss_bytes':4,'rss_bytes':8,'locked_bytes':0};b={'pss_bytes':None,'rss_bytes':None,'locked_bytes':None}
        report=self.resources([self.sample([a,b]),self.sample([a],owned_process_count=2,monotonic=1.4)])
        self.assertIsNone(report['host_pss_peak_bytes'])
        self.assertEqual(report['host_metric_coverage']['pss_bytes']['partial_known_sum_peak_bytes'],4)
        self.assertEqual(report['host_metric_coverage']['pss_bytes']['partial_samples'],2)
        self.assertAlmostEqual(report['observed_interval_s']['max'],.4)
        self.assertFalse(report['host_memory_comparison_eligible'])

    def test_complete_true_zero_and_observed_intervals_preserved(self):
        host=[{'pss_bytes':4,'rss_bytes':8,'locked_bytes':0}]
        report=self.resources([self.sample(host),self.sample(host,monotonic=1.7)])
        self.assertEqual(report['host_pss_peak_bytes'],4)
        self.assertEqual(report['host_locked_sum_peak_bytes'],0)
        self.assertTrue(report['host_memory_comparison_eligible'])
        self.assertAlmostEqual(report['observed_interval_s']['max'],.7)

    def test_permission_denied_sample_is_recorded_without_silent_thread_death(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler=helpers.ProcessSampler(Path(tmp)/'resource.jsonl')
            def owned(errors):return {123:'fake-owned'}
            with (mock.patch.object(sampler,'_owned',side_effect=owned),
                  mock.patch.object(helpers,'read_process_sample',side_effect=PermissionError('denied')),
                  mock.patch.object(helpers.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=''))):
                sampler._sample_once()
            record=json.loads(sampler.output.read_text())
            self.assertEqual(record['owned_process_count'],1)
            self.assertEqual(record['host'],[])
            self.assertIn('PermissionError',record['errors'][0]['error'])
            report=self.runner._load_resource_samples(sampler.output)
            self.assertIsNone(report['host_pss_peak_bytes'])
            self.assertFalse(report['host_memory_comparison_eligible'])

    def test_unexpected_sampler_thread_error_is_surfaced_at_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler=helpers.ProcessSampler(Path(tmp)/'resource.jsonl')
            with mock.patch.object(sampler,'_sample_once',side_effect=PermissionError('injected fatal')):
                sampler.start();sampler.thread.join(1)
                with self.assertRaisesRegex(RuntimeError,'PermissionError'):sampler.stop()
            report=self.runner._load_resource_samples(sampler.output)
            self.assertEqual(report['fatal_sampler_errors'],1)
            self.assertIsNone(report['host_pss_peak_bytes'])


if __name__=='__main__':unittest.main()
