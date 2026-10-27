"""CPU controls for actual PID mismatch, device totals, and final sampling."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/plana'))
import performance_helpers as helpers
import run_performance as runner


def host(pid=123,birth=77,pss=10):
    return dict(pid=pid,start_time=birth,pss_bytes=pss,rss_bytes=pss,locked_bytes=0,
                read_errors=[],sources={'status':True,'smaps_rollup':True})


def sample(monotonic=1.,pss=10,**values):
    return dict(monotonic=monotonic,host=[host(pss=pss)],gpu=[],owned_process_count=1,
                owned_tree_complete=True,gpu_query_complete=True,sample_interval_s=.1,errors=[],**values)


class PhysicalSamplerTests(unittest.TestCase):
    def collect(self,compute='',devices='0, GPU-zero, 636\n',*,birth=77,previous=None):
        with tempfile.TemporaryDirectory() as tmp:
            sampler=helpers.ProcessSampler(Path(tmp)/'resources.jsonl',assigned_gpus=('0',))
            if previous is not None:sampler.identities[123]=previous
            def query(argv,**kwargs):
                output=devices if '--query-gpu=index,uuid,memory.used' in argv else compute
                return SimpleNamespace(returncode=0,stdout=output,stderr='')
            with (mock.patch.object(sampler,'_owned',return_value={123:'owned'}),
                  mock.patch.object(helpers,'read_process_sample',return_value=host()),
                  mock.patch.object(helpers,'process_start_time',return_value=birth),
                  mock.patch.object(helpers.subprocess,'run',side_effect=query)):
                sampler._sample_once()
            return json.loads(sampler.output.read_text()),runner._load_resource_samples(sampler.output)

    def test_actual_diagnostic_pid_mismatch_is_retained_and_never_owned(self):
        row,report=self.collect('1823624, 626, GPU-zero\n')
        self.assertEqual(row['gpu_query']['stdout'],'1823624, 626, GPU-zero\n')
        self.assertEqual(row['gpu_query']['unmatched_processes'][0]['pid'],1823624)
        self.assertEqual(row['gpu_query']['availability'],'no_owned_compute_process_values')
        self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
        self.assertEqual(report['assigned_device_physical_total_sum_peak_bytes'],636<<20)
        self.assertIsNone(report['assigned_device_attributable_owned_peak_bytes'])
        self.assertFalse(report['assigned_device_ownership_proven'])
        self.assertEqual(report['gpu_unavailable_reason_counts'],{'no_owned_compute_process_values':1})

    def test_empty_enumeration_and_nonidle_device_never_prove_ownership(self):
        row,report=self.collect()
        self.assertEqual(row['gpu_query']['availability'],'no_reported_compute_processes')
        self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
        self.assertEqual(report['assigned_device_physical_total_sum_peak_bytes'],636<<20)
        self.assertFalse(row['assigned_gpu_devices'][0]['ownership_proven'])

    def test_numeric_pid_collision_is_not_namespace_identity_proof(self):
        row,report=self.collect('123, 626, GPU-zero\n')
        self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
        self.assertEqual(report['assigned_device_physical_total_sum_peak_bytes'],636<<20)
        self.assertEqual(row['gpu_query']['same_numeric_pid_candidates'][0]['physical_used_bytes'],626<<20)
        self.assertFalse(row['gpu_query']['pid_namespace_identity_proven'])
        self.assertTrue(row['assigned_gpu_devices'][0]['all_reported_compute_pids_numerically_match_owned_tree'])
        self.assertFalse(row['assigned_gpu_devices'][0]['ownership_proven'])

    def test_mixed_matching_and_unmatched_nvml_rows_never_form_a_complete_owned_sum(self):
        row,_=self.collect('123, 20, GPU-zero\n1823624, 606, GPU-zero\n')
        self.assertEqual(len(row['gpu_query']['same_numeric_pid_candidates']),1)
        self.assertEqual(len(row['gpu_query']['unmatched_processes']),1)
        self.assertEqual(row['gpu'],[])
        self.assertEqual(row['errors'],[])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resources.jsonl'
            helpers.append_json(path,row)
            helpers.append_json(path,dict(row,monotonic=row['monotonic']+.1))
            report=runner._load_resource_samples(path)
        self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
        self.assertFalse(report['gpu_memory_comparison_eligible'])
        self.assertTrue(report['host_memory_comparison_eligible'])
        self.assertEqual(report['assigned_device_physical_total_sum_peak_bytes'],636<<20)

    def test_reused_pid_cannot_be_attributed(self):
        for kwargs in ({'birth':88},{'previous':88}):
            with self.subTest(kwargs=kwargs):
                row,report=self.collect('123, 626, GPU-zero\n',**kwargs)
                self.assertFalse(row['gpu_query_complete'])
                self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
                self.assertTrue(row['errors'])

    def test_unavailable_process_memory_never_becomes_zero_or_device_fallback(self):
        row,report=self.collect('123, [N/A], GPU-zero\n')
        self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
        self.assertEqual(report['assigned_device_physical_total_sum_peak_bytes'],636<<20)
        self.assertEqual(row['gpu_query']['reported_processes'][0]['memory_raw'],'[N/A]')
        self.assertFalse(row['gpu_query_complete'])

    def test_unavailable_device_memory_is_null(self):
        row,report=self.collect(devices='0, GPU-zero, [N/A]\n')
        self.assertFalse(row['gpu_device_query_complete'])
        self.assertIsNone(report['assigned_device_physical_total_sum_peak_bytes'])
        self.assertTrue(any(e['phase']=='gpu_device' for e in row['errors']))

    def test_malformed_process_row_is_not_silently_ignored(self):
        row,report=self.collect('unexpected warning masquerading as data\n')
        self.assertFalse(row['gpu_query_complete'])
        self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
        self.assertIn('malformed',row['errors'][0]['error'])

    def test_assignment_excludes_unassigned_gpu_totals(self):
        row,report=self.collect(devices='0, GPU-zero, 636\n7, GPU-seven, 99999\n')
        self.assertEqual([r['index'] for r in row['assigned_gpu_devices']],['0'])
        self.assertEqual(report['assigned_device_physical_total_sum_peak_bytes'],636<<20)

    def test_query_failure_and_bounded_timeout_preserve_errors_and_nulls(self):
        with tempfile.TemporaryDirectory() as tmp:
            sampler=helpers.ProcessSampler(Path(tmp)/'resources.jsonl',assigned_gpus=('0',))
            failure=SimpleNamespace(returncode=7,stdout='',stderr='permission denied')
            with (mock.patch.object(sampler,'_owned',return_value={123:'owned'}),
                  mock.patch.object(helpers,'read_process_sample',return_value=host()),
                  mock.patch.object(helpers.subprocess,'run',side_effect=[failure,subprocess.TimeoutExpired('nvidia-smi',2)]) as query):
                sampler._sample_once()
            row=json.loads(sampler.output.read_text());report=runner._load_resource_samples(sampler.output)
            self.assertEqual(row['gpu_query']['returncode'],7)
            self.assertEqual(row['gpu_query']['stderr'],'permission denied')
            self.assertEqual({e['phase'] for e in row['errors']},{'gpu','gpu_device'})
            self.assertTrue(all(call.kwargs['timeout']==2 for call in query.call_args_list))
            self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])
            self.assertIsNone(report['assigned_device_physical_total_sum_peak_bytes'])

    def test_process_birth_change_during_memory_read_is_rejected(self):
        with (mock.patch.object(helpers,'process_start_time',side_effect=(77,88)),
              mock.patch.object(Path,'read_text',return_value='VmRSS: 1 kB\n')):
            with self.assertRaisesRegex(ProcessLookupError,'start time changed'):
                helpers.read_process_sample(123)

    def test_duplicate_device_identity_and_missing_assignment_fail_closed(self):
        for text in ('0, GPU-zero, 636\n1, GPU-zero, 2\n','1, GPU-one, 636\n'):
            with self.subTest(text=text):
                row,report=self.collect(devices=text)
                self.assertFalse(row['gpu_device_query_complete'])
                self.assertIsNone(report['assigned_device_physical_total_sum_peak_bytes'])

    def test_assigned_indices_include_target_and_draft_but_reject_ambiguous_ids(self):
        config={'deployment':{'disaggregated':{'managed_local':{
            'trainer_cuda_visible_devices':['2'],'capture_servers':[{'cuda_visible_devices':['0','1']}]
        }}}}
        self.assertEqual(helpers.assigned_gpu_indices(config),('0','1','2'))
        config['deployment']['disaggregated']['managed_local']['trainer_cuda_visible_devices']=['GPU-ambiguous']
        with self.assertRaises(ValueError):helpers.assigned_gpu_indices(config)


class FinalResourceTests(unittest.TestCase):
    def report(self,rows):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resources.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            return runner._load_resource_samples(path)

    def test_per_device_peaks_do_not_fabricate_a_simultaneous_sum(self):
        rows=[]
        for time,values in enumerate(((100,400),(300,200)),1):
            rows.append(sample(monotonic=time,assigned_gpu_indices=['0','1'],gpu_device_query_complete=True,
                assigned_gpu_devices=[{'index':str(i),'gpu_uuid':f'GPU-{i}','physical_total_used_bytes':value}
                                      for i,value in enumerate(values)]))
        report=self.report(rows)
        self.assertEqual(report['assigned_device_physical_total_peak_bytes']['0']['peak_bytes'],300)
        self.assertEqual(report['assigned_device_physical_total_peak_bytes']['1']['peak_bytes'],400)
        self.assertEqual(report['assigned_device_physical_total_sum_peak_bytes'],500)
        self.assertTrue(report['assigned_device_total_observation_complete'])
        self.assertIsNone(report['gpu_physical_used_sum_peak_bytes'])

    def test_partial_device_data_remains_partial_and_preserves_available_device(self):
        row=sample(assigned_gpu_indices=['0','1'],gpu_device_query_complete=False,assigned_gpu_devices=[
            {'index':'0','gpu_uuid':'GPU-0','physical_total_used_bytes':100},
            {'index':'1','gpu_uuid':'GPU-1','physical_total_used_bytes':None}])
        report=self.report([row])
        self.assertIsNone(report['assigned_device_physical_total_sum_peak_bytes'])
        self.assertEqual(report['assigned_device_physical_total_peak_bytes']['0']['peak_bytes'],100)
        self.assertFalse(report['assigned_device_total_observation_complete'])

    def test_gpu_assignment_or_uuid_changes_are_rejected(self):
        a=sample(assigned_gpu_indices=['0'],gpu_device_query_complete=True,
                 assigned_gpu_devices=[{'index':'0','gpu_uuid':'GPU-0','physical_total_used_bytes':100}])
        b=json.loads(json.dumps(a));b['assigned_gpu_indices']=['1']
        with self.assertRaisesRegex(AssertionError,'set changed'):self.report([a,b])
        b=json.loads(json.dumps(a));b['assigned_gpu_devices'][0]['gpu_uuid']='GPU-replaced'
        with self.assertRaisesRegex(AssertionError,'UUID changed'):self.report([a,b])

    def test_real_finalizer_includes_late_teardown_sample_and_preserves_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'resources.jsonl'
            helpers.append_json(path,sample(monotonic=3.,pss=10))
            events=root/'events';events.mkdir()
            (events/'training-steps-123.jsonl').write_text(
                json.dumps({'step':8,'monotonic':2.})+'\n'+json.dumps({'step':40,'monotonic':4.})+'\n')
            result={'actual_fit':{}}
            def stop_sampler():
                helpers.append_json(path,sample(monotonic=6.,pss=900))
                return {'thread_drained':True,'fatal_errors':[]}
            runner.finish_owned_run(SimpleNamespace(_terminate_processes=lambda *a,**k:None),
                lambda *a,**k:None,[],[],SimpleNamespace(stop=stop_sampler),root,grace_s=.01,
                result=result,workload={'training_warmup_steps':8,'measured_steps':32})
            self.assertEqual(result['observed_resources']['sample_count'],2)
            self.assertEqual(result['observed_resources']['host_pss_peak_bytes'],900)
            self.assertEqual(result['measured_window_observed_resources']['sample_count'],1)
            self.assertEqual(result['measured_window_observed_resources']['host_pss_peak_bytes'],10)
            self.assertEqual(result['measured_window_observed_resources']['window_monotonic'],(2.,4.))

    def test_teardown_exit_race_is_retained_and_keeps_whole_host_ineligible(self):
        a=sample();b=sample(monotonic=2.,pss=None)
        b['errors']=[{'phase':'host','error':'ProcessLookupError: exited'}]
        report=self.report([a,b])
        self.assertEqual(report['host_metric_coverage']['pss_bytes']['complete_samples'],1)
        self.assertFalse(report['host_memory_comparison_eligible'])
        self.assertEqual(report['host_pss_peak_bytes'],10)
        self.assertTrue(report['errors'])

    def test_undrained_sampler_never_gets_final_observations(self):
        result={}
        sampler=SimpleNamespace(stop=mock.Mock(side_effect=RuntimeError('not drained')))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'not drained'):
                runner.finish_owned_run(SimpleNamespace(_terminate_processes=lambda *a,**k:None),
                    lambda *a,**k:None,[],[],sampler,tmp,grace_s=.01,result=result,workload={})
        self.assertNotIn('observed_resources',result)
        self.assertIn('drain unproven',result['resource_observation_error'])


if __name__=='__main__':unittest.main()
