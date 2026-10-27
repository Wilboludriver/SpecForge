"""CPU failure receipts must preserve the original cause and fixture ownership."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/plana'))
from actual_faults import atomic_json
from actual_faults.rank_hooks import RankState
from actual_faults.scenarios import command


class CommandTests(unittest.TestCase):
    def test_rank_command_failure_is_acknowledged_without_killing_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=object.__new__(RankState);state.directory=Path(tmp);state.rank=1;state.last_command=None
            state.emit=mock.Mock();state._apply_command=mock.Mock(side_effect=AssertionError('actual missing checkpoint'))
            atomic_json(Path(tmp)/'command.json',{'id':'cmd','action':'pin_kv'})
            state.command()
            row=json.loads((Path(tmp)/'ack-cmd-rank1.json').read_text())
            self.assertIn('actual missing checkpoint',row['error'])
            self.assertIn('AssertionError',row['traceback'])
            self.assertEqual(row['rank'],1)
            state.command();state._apply_command.assert_called_once()

    def test_driver_rejects_rank_error_after_receiving_all_rank_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in (0,1):
                row={'rank':rank,'action':'pin_kv'}
                if rank==1:row['error']='actual missing checkpoint'
                atomic_json(Path(tmp)/f'ack-cmd-rank{rank}.json',row)
            with mock.patch('actual_faults.scenarios.uuid.uuid4',return_value=SimpleNamespace(hex='cmd')):
                with self.assertRaisesRegex(RuntimeError,'actual missing checkpoint'):
                    command(tmp,'pin_kv',tp_size=2,timeout=.1)

    def test_failed_actual_release_retains_ownership_and_success_removes_it(self):
        state=object.__new__(RankState);receipt=object()
        release=mock.Mock(side_effect=RuntimeError('pool release failed'))
        state.scheduler=SimpleNamespace(tree_cache=SimpleNamespace(dec_lock_ref=release))
        state.fixture_pins={'owned':('kv',7,receipt)}
        with self.assertRaisesRegex(RuntimeError,'pool release failed'):state._release_fixture_pin('owned')
        self.assertEqual(state.fixture_pins['owned'],('kv',7,receipt))
        release.side_effect=None;state._release_fixture_pin('owned')
        self.assertFalse(state.fixture_pins)
        self.assertEqual(release.call_args_list,[mock.call(7,receipt),mock.call(7,receipt)])

    def test_release_all_clears_only_fixture_owned_kv_and_hs(self):
        state=object.__new__(RankState)
        kv=mock.Mock();hs=mock.Mock()
        state.scheduler=SimpleNamespace(tree_cache=SimpleNamespace(dec_lock_ref=kv))
        state.bridge=SimpleNamespace(cache=SimpleNamespace(release_lease=hs))
        state.fixture_pins={'kv':('kv',7,'receipt'),'hs':('hs','lease')}
        with tempfile.TemporaryDirectory() as tmp,mock.patch('actual_faults.rank_hooks.engine_snapshot',return_value={}):
            state.directory=Path(tmp);state.rank=0
            state._apply_command({'id':'cmd','action':'release_fixture_pins'})
        kv.assert_called_once_with(7,'receipt');hs.assert_called_once_with('lease')
        self.assertFalse(state.fixture_pins)
    def test_empty_real_HS_lease_is_retained_on_failed_pin_then_released(self):
        from sglang.srt.spec_feature_cache import PrefixFeatureCache,FeatureSchema,FeatureSpec
        schema=FeatureSchema((FeatureSpec('aux',2,'bfloat16',(1,)),FeatureSpec('last_hidden',2,'bfloat16',())))
        cache=PrefixFeatureCache(schema,2,4096)
        state=object.__new__(RankState)
        state.bridge=SimpleNamespace(cache=cache,namespace='ns',writer=True)
        state.scheduler=SimpleNamespace();state.fixture_pins={};state.last_command=None
        state.rank=0;state.emit=mock.Mock()
        with tempfile.TemporaryDirectory() as tmp,mock.patch('actual_faults.rank_hooks.engine_snapshot',return_value={}):
            state.directory=Path(tmp)
            atomic_json(Path(tmp)/'command.json',{'id':'pin','action':'pin_hs','tokens':[1,2],'pin_id':'owned'})
            state.command()
            self.assertEqual(cache.stats()['active_leases'],1)
            self.assertIn('owned',state.fixture_pins)
            self.assertIn('no actual READY coverage',json.loads((Path(tmp)/'ack-pin-rank0.json').read_text())['error'])
            atomic_json(Path(tmp)/'command.json',{'id':'release','action':'release_fixture_pins'})
            state.command()
            self.assertEqual(cache.stats()['active_leases'],0);self.assertFalse(state.fixture_pins)
            self.assertNotIn('error',json.loads((Path(tmp)/'ack-release-rank0.json').read_text()))

    def test_ordinary_warm_requires_bounded_actual_hit_and_no_HS_change(self):
        from actual_faults.scenarios import Client
        row={'input_ids':[1,2,3,4]}
        base={'cache':{'ready_blocks':1,'ready_bytes':8,'cache_bytes':8}}
        for defect in ('none','no-hit','capture','N-bound','HS-change'):
            with tempfile.TemporaryDirectory() as tmp:
                client=object.__new__(Client);client.base_url='http://not-called';client.post=mock.Mock()
                client.timeout=1;client.directory=Path(tmp)
                probes=[{'meta_info':{'cached_tokens':0}},{'meta_info':{'cached_tokens':2}}]
                if defect=='no-hit':probes[1]['meta_info']['cached_tokens']=0
                elif defect=='capture':probes[1]['meta_info']['spec_capture']={}
                elif defect=='N-bound':probes[1]['meta_info']['cached_tokens']=4
                client.transport=mock.Mock(side_effect=probes)
                states=[base,base,base,base]
                if defect=='HS-change':states[-1]={'cache':dict(base['cache'],ready_blocks=0)}
                client.cleanup=mock.Mock(side_effect=states)
                with mock.patch('specforge.inference.capture_protocol.CaptureProtocol') as protocol:
                    protocol.return_value.capability.return_value={'capture_namespace':'ns'}
                    if defect=='none':
                        self.assertEqual([r['cached_tokens'] for r in client.warm_ordinary_checkpoint(row,'test')],[0,2])
                    else:
                        with self.subTest(defect=defect),self.assertRaises(AssertionError):
                            client.warm_ordinary_checkpoint(row,'test')
                self.assertEqual(client.transport.call_count,2)
                for call in client.transport.call_args_list:
                    body=call.kwargs['json_body']
                    self.assertEqual(body['input_ids'],row['input_ids'])
                    self.assertEqual(body['extra_key'],'ns');self.assertNotIn('spec_capture',body)


if __name__=='__main__':unittest.main()
