"""CPU-only formal performance planning/accounting contract tests."""
import importlib.util
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
    def test_deterministic_flag_and_model_identity_survive_mode_switch(self):
        hot={'training':[row()]*72}
        off=helpers.isolate_config(config(),run_dir='/tmp/off',workload=hot,group='host-off',transport='tcp',cache_bytes=64<<20,chunk_tokens=256,rdma_devices=None)
        on=helpers.isolate_config(config(),run_dir='/tmp/on',workload=hot,group='host-on',transport='tcp',cache_bytes=64<<20,chunk_tokens=256,rdma_devices=None)
        self.assertIn('--enable-deterministic-inference',on['model']['sglang_extra_args'])
        self.assertEqual(helpers.comparison_contract(off,'same',{},64<<20,256),helpers.comparison_contract(on,'same',{},64<<20,256))
    def test_compare_rejects_configuration_change_and_reports_slowdown_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=[]
            for group,wall in (('host-off',1.),('host-on',2.)):
                root=Path(tmp)/group;root.mkdir();roots.append(root)
                (root/'manifest.json').write_text(json.dumps({'comparison_contract':{'data':'same'}}))
                (root/'result.json').write_text(json.dumps({'group':group,'status':'passed','initial_draft':[{'sha256':'same'}],
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


if __name__=='__main__':unittest.main()
