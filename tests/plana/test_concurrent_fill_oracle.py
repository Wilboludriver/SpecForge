"""CPU-only receipt negative controls and real CPU hook mechanics; no GPU proof."""
import copy
from concurrent.futures import Future,ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

TOOLS=Path(__file__).resolve().parents[2]/'scripts/plana'
sys.path.insert(0,str(TOOLS))
from actual_faults import atomic_json,read_json,wait_for
from actual_faults.concurrent_fill import (KIND,_Observer,assert_receipts,execution_wire,guarded_cpu_copy,
    install,key_wire,pair_arm,token_digest,validate_config)


def fixture():
    arm={'kind':KIND,'case_id':'pair','actors':{'A':{'rid':'rid-A','sample_id':'run:A'},'B':{'rid':'rid-B','sample_id':'run:B'}},
         'N':1280,'token_sha256':token_digest(range(1280)),'gate_timeout_s':2}
    identities={actor:{'server_epoch':'epoch','request_id':'rid-'+actor,'attempt_id':'attempt-'+actor,'execution_epoch':0} for actor in ('A','B')}
    rows=[]
    def add(event,actor,rank=0,**kw):
        sequence=1+sum(row['rank']==rank for row in rows)
        row={'event':event,'actor':actor,'rank':rank,'seq':sequence,'case_id':'pair','execution':dict(identities[actor]),**kw}
        rows.append(row);return row
    def key(start):return {'namespace':'ns','digest':'key-'+str(start),'parent_digest':'parent-'+str(start),'start':start,'end':start+256}
    missing={'state':'MISSING','writer_active':False,'fill_ids':[],'features':[],
             'cache_bytes':0,'capacity_bytes':64<<20,'chunk_bytes':256*36864}
    writing={**missing,'state':'WRITING','writer_active':True,'fill_ids':['fill-0'],'cache_bytes':256*36864}
    ready={**missing,'state':'READY','features':['aux','last_hidden'],'cache_bytes':1280*36864}
    for actor in ('A','B'):
        for rank in (0,1):
            add('cf_admitted',actor,rank,member=True,plan={'H':0,'L':0,'N':1280,'span':[0,1280],
                'execution_epoch':0,'plan_digest':'plan-'+actor},token_sha256=arm['token_sha256'],sample_id=arm['actors'][actor]['sample_id'])
            add('cf_forward',actor,rank,batch_size=1,effective_running=1,a_finished=actor=='B',a_effective=False)
            add('cf_forward_return',actor,rank)
        if actor=='A':
            data={'key':key(0),'features_cuda':True,'producer_event':True,'workers':1,'future_pending':True,
                  'N':1280,'L':0,'span':[0,1280],'token_sha256':arm['token_sha256'],'schema_signature':'schema',
                  'reservation':{'device_bytes':1280*36864,'pinned_bytes':1280*36864,'host_bytes':1280*36864+20000}}
            add('cf_a_submitted','A',**data)
            add('cf_copy_start','A',worker=123,thread='spec-feature-capture_0')
            add('cf_reserve','A',key=key(0),before=copy.deepcopy(missing),fill_id='fill-0',reason='RESERVED',owner=dict(identities['A']))
            add('cf_aux_copy_done','A',key=key(0),fill_id='fill-0',copied=['aux'],before=copy.deepcopy(writing),worker=123,thread='spec-feature-capture_0')
            add('cf_partial_gate','A',key=key(0),fill_id='fill-0')
            # The real B lookup occurs inside its admission, before the admission-return receipt.
            add('cf_lookup_writing','B',key=key(0),H=0,gate_held=True,before=copy.deepcopy(writing))
        else:
            add('cf_b_submitted','B',**data,gate_held=True,b_copy_started=False,before=copy.deepcopy(writing),
                controller={'pending_copies':2,'output_reservations':2,'output_bytes':2*1280*36864+40000})
            add('cf_b_producer_ready','B',producer_ready=True)
            add('cf_gate_release','B',reason='verified_b_submit')
    for start in range(0,1280,256):
        if start:add('cf_reserve','A',key=key(start),before=copy.deepcopy(missing),fill_id='fill-'+str(start),reason='RESERVED',owner=dict(identities['A']))
        add('cf_publish_ready','A',key=key(start),fill_id='fill-'+str(start),published=True,
            after=copy.deepcopy(ready),copied=['aux','last_hidden'],worker=123,thread='spec-feature-capture_0')
    add('cf_copy_start','B',worker=123,thread='spec-feature-capture_0')
    for start in range(0,1280,256):
        add('cf_reserve','B',key=key(start),before=copy.deepcopy(ready),fill_id=None,reason='READY',owner=dict(identities['B']))
    return arm,rows


def selected(rows,event,actor=None,rank=0):
    return next(row for row in rows if row['event']==event and row['rank']==rank and (actor is None or row['actor']==actor))


class ReceiptTests(unittest.TestCase):
    def test_complete_natural_overlap_passes(self):
        arm,rows=fixture();result=assert_receipts(rows,arm)
        self.assertTrue(result['natural_overlap']);self.assertEqual(result['ready_skips'],5)
        self.assertFalse(result['partial_ready_visible'])

    def test_each_required_boundary_and_both_tp_ranks_are_required(self):
        arm,rows=fixture()
        for index,row in enumerate(rows):
            if row['event'] not in ('cf_admitted','cf_forward','cf_forward_return','cf_a_submitted','cf_aux_copy_done',
                'cf_partial_gate','cf_lookup_writing','cf_b_submitted','cf_b_producer_ready','cf_gate_release','cf_copy_start','cf_reserve','cf_publish_ready'):continue
            with self.subTest(event=row['event'],actor=row['actor'],rank=row['rank']):
                with self.assertRaises(AssertionError):assert_receipts(rows[:index]+rows[index+1:],arm)

    def test_repeated_first_chunk_commit_cannot_prove_all_five_keys(self):
        arm,rows=fixture()
        commits=[r for r in rows if r['event']=='cf_publish_ready']
        first=copy.deepcopy(commits[0])
        for row in commits:
            row['key']=copy.deepcopy(first['key'])
            row['fill_id']=first['fill_id']
        with self.assertRaisesRegex(AssertionError,'every distinct reserved content key'):
            assert_receipts(rows,arm)

    def test_meaningful_scalar_negative_controls(self):
        arm,original=fixture()
        changes=[('http_only',lambda r:r.remove(selected(r,'cf_b_submitted','B'))),
            ('late_submit',lambda r:selected(r,'cf_b_submitted','B').update(seq=999)),
            ('wrong_key',lambda r:selected(r,'cf_b_submitted','B')['key'].update(digest='other')),
            ('partial_ready',lambda r:selected(r,'cf_aux_copy_done','A')['before'].update(state='READY')),
            ('published_aux_early',lambda r:selected(r,'cf_aux_copy_done','A')['before'].update(features=['aux'])),
            ('not_writer_active',lambda r:selected(r,'cf_lookup_writing','B')['before'].update(writer_active=False)),
            ('lookup_crosses_writing',lambda r:selected(r,'cf_lookup_writing','B').update(H=256)),
            ('gate_not_held',lambda r:selected(r,'cf_b_submitted','B').update(gate_held=False)),
            ('no_cpu_copy',lambda r:selected(r,'cf_aux_copy_done','A').update(copied=[])),
            ('wrong_owner',lambda r:selected(r,'cf_reserve','B')['owner'].update(attempt_id='foreign')),
            ('same_attempt',lambda r:selected(r,'cf_b_submitted','B')['execution'].update(attempt_id='attempt-A')),
            ('other_epoch',lambda r:selected(r,'cf_b_submitted','B')['execution'].update(server_epoch='old')),
            ('peer_plan',lambda r:selected(r,'cf_admitted','B',1)['plan'].update(L=256)),
            ('not_admitted',lambda r:selected(r,'cf_admitted','B',1).update(member=False)),
            ('engine_two',lambda r:selected(r,'cf_forward','B').update(effective_running=2)),
            ('A_still_running',lambda r:selected(r,'cf_forward','B').update(a_finished=False,a_effective=True)),
            ('CPU_span',lambda r:selected(r,'cf_b_submitted','B').update(features_cuda=False)),
            ('missing_event',lambda r:selected(r,'cf_b_submitted','B').update(producer_event=False)),
            ('more_workers',lambda r:selected(r,'cf_b_submitted','B').update(workers=2)),
            ('B_started_early',lambda r:selected(r,'cf_b_submitted','B').update(b_copy_started=True)),
            ('no_pending',lambda r:selected(r,'cf_b_submitted','B')['controller'].update(pending_copies=1)),
            ('missing_charge',lambda r:selected(r,'cf_b_submitted','B')['reservation'].update(device_bytes=0)),
            ('no_output_overlap',lambda r:selected(r,'cf_b_submitted','B')['controller'].update(output_reservations=1)),
            ('fake_ready',lambda r:selected(r,'cf_b_producer_ready','B').update(producer_ready=False)),
            ('early_release',lambda r:selected(r,'cf_gate_release','B').update(reason='b_submit_error')),
            ('capacity_skip',lambda r:selected(r,'cf_reserve','B').update(reason='CAPACITY_OR_OTHER')),
            ('writing_skip_claim',lambda r:selected(r,'cf_reserve','B').update(reason='WRITING')),
            ('second_committer',lambda r:selected(r,'cf_publish_ready','A').update(actor='B')),
            ('partial_features',lambda r:selected(r,'cf_publish_ready','A')['after'].update(features=['aux'])),
            ('different_worker',lambda r:selected(r,'cf_copy_start','B').update(worker=456)),
            ('capacity_pressure',lambda r:selected(r,'cf_lookup_writing','B')['before'].update(capacity_bytes=1))]
        for name,change in changes:
            rows=copy.deepcopy(original)
            with self.subTest(defect=name):
                change(rows)
                with self.assertRaises(AssertionError):assert_receipts(rows,arm)


class HookMechanicsTests(unittest.TestCase):
    def state(self,directory,cache):
        records=[]
        def emit(event,**values):records.append({'event':event,'rank':0,'seq':len(records)+1,**values})
        bridge=SimpleNamespace(cache=cache,controller=None,epoch='epoch')
        return SimpleNamespace(directory=Path(directory),bridge=bridge,scheduler=SimpleNamespace(),rank=0,gates={},emit=emit),records

    def runtime(self):
        import torch
        from sglang.srt.spec_feature_cache import PrefixFeatureCache,FeatureSchema,FeatureSpec,ExecutionIdentity
        schema=FeatureSchema((FeatureSpec('aux',2,'bfloat16',(1,)),FeatureSpec('last_hidden',1,'bfloat16',())))
        cache=PrefixFeatureCache(schema,256,64<<20)
        return torch,cache,ExecutionIdentity('epoch','rid-A','attempt-A',0)

    def test_real_cpu_fill_copy_is_held_and_cannot_partially_lookup(self):
        torch,cache,execution=self.runtime()
        with tempfile.TemporaryDirectory() as tmp:
            arm,_=fixture();atomic_json(Path(tmp)/'arm.json',arm)
            state,records=self.state(tmp,cache);observer=_Observer(state);observer.executions['A']=execution_wire(execution)
            key=cache.block_keys('ns',tuple(range(1280)))[0]
            fill=observer.reserve(cache.reserve_fill,key,execution)
            source={'aux':torch.arange(512,dtype=torch.float32).reshape(256,2).to(torch.bfloat16),
                    'last_hidden':torch.ones(256,1,dtype=torch.bfloat16)}
            old=torch.Tensor.copy_
            def wrapped(target,source,*a,**kw):return guarded_cpu_copy(old,target,source,*a,**kw)
            with mock.patch.object(torch.Tensor,'copy_',wrapped),ThreadPoolExecutor(max_workers=1,thread_name_prefix='spec-feature-capture') as pool:
                future=pool.submit(observer.publish,cache.publish_ready,fill,source)
                wait_for(lambda:(Path(tmp)/'fill-partial-pair.json').exists(),1,'real CPU partial fill')
                self.assertFalse(future.done());self.assertEqual(cache.stats()['writing_blocks'],1)
                lease=cache.lookup_and_pin('ns',tuple(range(1280)),1280)
                self.assertEqual(lease.coverage_end,0);cache.release_lease(lease)
                self.assertEqual(observer.observation(key)['features'],[])
                observer.auto_release=True;observer.gate.set();self.assertTrue(future.result(2))
            self.assertEqual(cache.stats()['ready_blocks'],1);self.assertEqual(cache.stats()['fill_reservations'],0)
            self.assertEqual(next(r for r in records if r['event']=='cf_publish_ready')['copied'],['aux','last_hidden'])

    def test_B_submit_validation_failure_still_releases_gate_after_original_enqueue(self):
        torch,cache,_=self.runtime()
        from sglang.srt.spec_feature_cache import ExecutionIdentity
        with tempfile.TemporaryDirectory() as tmp:
            arm,_=fixture();atomic_json(Path(tmp)/'arm.json',arm)
            state,records=self.state(tmp,cache);observer=_Observer(state)
            plan=SimpleNamespace(execution=ExecutionIdentity('epoch','rid-B','attempt-B',0),namespace='ns',tokens=tuple(range(1280)),
                N=1280,L=0,schema_signature='schema')
            span=SimpleNamespace(start=0,end=1280,ready_event=SimpleNamespace(query=lambda:True),
                features={'aux':torch.zeros(1280,2),'last_hidden':torch.zeros(1280,1)})
            future=Future();future.set_running_or_notify_cancel();original=mock.Mock(return_value=future)
            with self.assertRaisesRegex(AssertionError,'real full-range CUDA'):observer.submit(original,plan,span)
            original.assert_called_once_with(plan,span)
            self.assertTrue(observer.gate.is_set());self.assertFalse(observer.auto_release)
            self.assertEqual(selected(records,'cf_gate_release','B')['reason'],'b_submit_error')
            self.assertFalse(read_json(Path(tmp)/'concurrent-fill-error.json')['overlap_covered'])

    def test_B_receipt_or_metadata_failure_cannot_strand_A_gate(self):
        torch,cache,_=self.runtime()
        from sglang.srt.spec_feature_cache import ExecutionIdentity
        for cause in ('original-submit', 'metadata'):
            with tempfile.TemporaryDirectory() as tmp:
                arm,_=fixture();atomic_json(Path(tmp)/'arm.json',arm)
                state,_=self.state(tmp,cache);observer=_Observer(state)
                plan=SimpleNamespace(execution=ExecutionIdentity('epoch','rid-B','attempt-B',0),namespace='ns',tokens=tuple(range(1280)),
                    N=1280,L=0,schema_signature='schema')
                original=mock.Mock(side_effect=RuntimeError('actual submit failed'))
                state.emit=mock.Mock(side_effect=OSError('receipt write failed'))
                if cause=='metadata':
                    plan.tokens=None
                with self.subTest(cause=cause),self.assertRaisesRegex(OSError,'receipt write failed'):
                    observer.submit(original,plan,SimpleNamespace())
                self.assertTrue(observer.gate.is_set());self.assertFalse(observer.auto_release)
                if cause=='original-submit':original.assert_called_once()
                else:original.assert_not_called()

    def test_inactive_copy_wrapper_preserves_existing_semantics(self):
        import torch
        old=torch.Tensor.copy_;calls=[]
        def original(target,source,*a,**kw):calls.append((a,kw));return old(target,source,*a,**kw)
        target=torch.empty(4);source=torch.arange(4,dtype=torch.float32)
        returned=guarded_cpu_copy(original,target,source,non_blocking=False)
        self.assertIs(returned,target);self.assertTrue(torch.equal(target,source));self.assertEqual(len(calls),1)

    def test_pair_arm_rejects_aliases_and_wrong_actor_layout(self):
        for defect in ('same_rid','same_sample','extra_actor','no_tokens','short'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                arm,_=fixture()
                if defect=='same_rid':arm['actors']['B']['rid']=arm['actors']['A']['rid']
                if defect=='same_sample':arm['actors']['B']['sample_id']=arm['actors']['A']['sample_id']
                if defect=='extra_actor':arm['actors']['G']={'rid':'g','sample_id':'g'}
                if defect=='no_tokens':arm['token_sha256']=''
                if defect=='short':arm['N']=1024
                atomic_json(Path(tmp)/'arm.json',arm)
                with self.assertRaises(AssertionError):pair_arm(tmp)

    def test_dedicated_config_scope_is_enforced(self):
        config={'kinds':[KIND],'model':{'name':'qwen36-a3b'},'engine_kwargs':{'tp_size':2,'max_running_requests':1,
                'spec_feature_chunk_tokens':256,'spec_feature_cache_bytes':64<<20}}
        validate_config(config)
        for key,value in (('tp_size',1),('max_running_requests',2),('spec_feature_chunk_tokens',384),('spec_feature_cache_bytes',1<<30)):
            bad=copy.deepcopy(config);bad['engine_kwargs'][key]=value
            with self.assertRaises(ValueError):validate_config(bad)
        with self.assertRaises(ValueError):validate_config({**config,'kinds':[KIND,'d2h_partial']})

    def test_existing_one_actor_fault_guard_ignores_new_pair_kind(self):
        from actual_faults.rank_hooks import RankState
        with tempfile.TemporaryDirectory() as tmp:
            arm,_=fixture();atomic_json(Path(tmp)/'arm.json',arm)
            state=RankState.__new__(RankState);state.directory=Path(tmp)
            self.assertIsNone(state.arm_for(request_id='rid-A'))
            self.assertIsNone(state.arm_for(req=SimpleNamespace(rid='rid-B',spec_capture={'sample_id':'run:B'})))
            atomic_json(Path(tmp)/'arm.json',{'case_id':'old','kind':'d2h_partial','rid':'old-rid','sample_id':'old-sample'})
            self.assertEqual(state.arm_for(request_id='old-rid')['kind'],'d2h_partial')


class DedicatedDriverTests(unittest.TestCase):
    def run_dedicated(self, failure=False):
        from actual_faults import driver,launcher,scenarios
        import actual_faults.concurrent_fill as module
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/'native.jsonl'
            data.write_text(''.join(json.dumps({'id':name,'input_ids':list(range(1280)),'loss_mask':[1]*1280})+'\n'
                for name in ('cold-1024','extension-1280','same-tail-different-history')))
            config={'kinds':[KIND],'cases_file':str(data)}
            client=SimpleNamespace(timeout=1,cleanup=mock.Mock(),close=mock.Mock(),submit=mock.Mock())
            server=SimpleNamespace(_base_url='http://unused',shutdown=mock.Mock())
            def concurrent(*args):
                client.submit.assert_not_called()  # Any common golden/prewarm here invalidates the fixture.
                self.assertEqual(len(args[2]['input_ids']),1280)
                if failure:raise AssertionError('natural overlap not covered')
                return {'kind':KIND,'passed':True}
            with (mock.patch.object(driver,'source_manifest',return_value={}),
                  mock.patch.object(driver,'wait_for',return_value=True),
                  mock.patch.object(launcher,'start',return_value=server),
                  mock.patch.object(launcher,'execute',return_value=None),
                  mock.patch.object(scenarios,'Client',return_value=client),
                  mock.patch.object(module,'run_case',side_effect=concurrent)):
                result=driver.run(config,Path(tmp)/'run')
            client.cleanup.assert_called_once();client.close.assert_called_once();server.shutdown.assert_called_once()
            client.submit.assert_not_called()
            return result

    def test_dedicated_branch_precedes_all_common_golden_or_warmup(self):
        result=self.run_dedicated()
        self.assertTrue(result['passed']);self.assertTrue(result['source_unchanged'])
        self.assertEqual(len(result['cases']),1)

    def test_unexercised_overlap_is_failure_and_still_uses_outer_cleanup(self):
        result=self.run_dedicated(failure=True)
        self.assertFalse(result['passed']);self.assertIn('natural overlap not covered',result['error'])



if __name__=='__main__':unittest.main()
