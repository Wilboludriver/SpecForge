"""CPU negative controls for unique run keys and genuine cancel-event receipts."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/plana'))
from actual_faults.scenarios import run_namespace, assert_cuda_cancel_receipts


class IdentityAndGateTests(unittest.TestCase):
    def test_unique_outer_runs_and_restart_children_never_share_namespace(self):
        paths=['/tmp/run1/engine','/tmp/run2/engine','/tmp/run1/engine/S1','/tmp/run1/engine/S2']
        values=[run_namespace(path) for path in paths]
        self.assertEqual(len(set(values)),len(values))
        self.assertEqual(values[0],run_namespace(paths[0]))

    def evidence(self):
        marker={'pending':True,'stage':'after_real_aux_d2h','issued':['aux'],'event_handle':13}
        rows=[{'event':'delayed_real_cuda_event','seq':1,'monotonic_ns':11,
               'event_pending':True,'stage':marker['stage'],'issued':['aux'],
               'event_handle':13,'producer_event_handle':12,'source_pointers':{'aux':31,'last_hidden':32}},
              {'event':'cancel_event_observation','seq':2,'monotonic_ns':22,'event_pending':True,'event_handle':13},
              {'event':'cuda_gate_released','seq':4,'monotonic_ns':44,'reason':'observed_cancel'},
              {'event':'cancel_control_applied','seq':3,'monotonic_ns':33,'state':'DRAINING','writer_drained':False}]
        for row in rows:row.update(case_id='c',rank=0)
        return rows,marker

    def test_requires_same_real_event_still_pending_at_cancel_and_owned_release(self):
        rows,marker=self.evidence()
        assert_cuda_cancel_receipts(rows,'c',marker)
        for defect in ('expired','wrong_event','producer_event','watchdog','early_release','wrong_stage','no_dma','no_sources','other_rank','duplicate','no_invalidation','premature_drained'):
            rows,marker=self.evidence()
            if defect=='expired':rows[1]['event_pending']=False
            elif defect=='wrong_event':rows[1]['event_handle']=99
            elif defect=='producer_event':rows[0]['producer_event_handle']=13
            elif defect=='watchdog':rows[2]['reason']='watchdog'
            elif defect=='early_release':rows[2]['seq']=0
            elif defect=='wrong_stage':marker['stage']='producer_only'
            elif defect=='no_dma':rows[0]['issued']=[]
            elif defect=='no_sources':rows[0]['source_pointers'].pop('last_hidden')
            elif defect=='other_rank':rows[1]['rank']=1
            elif defect=='no_invalidation':rows[3]['state']='MATERIALIZING'
            elif defect=='premature_drained':rows[3]['writer_drained']=True
            else:rows.append(copy.deepcopy(rows[1]))
            with self.subTest(defect=defect),self.assertRaises(AssertionError):
                assert_cuda_cancel_receipts(rows,'c',marker)


if __name__=='__main__':unittest.main()
