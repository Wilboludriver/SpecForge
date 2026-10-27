"""Exercise driver control flow; mocked I/O here is not actual-model evidence."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/plana'))
from actual_faults import driver


class TeardownTests(unittest.TestCase):
    def check_path(self, concurrent, defect):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'run';cases=root/'cases.jsonl'
            cases.write_text('\n'.join(json.dumps({'id':name,'input_ids':[1000,1001],
                'loss_mask':[0,1]}) for name in ('cold-1024','extension-1280','same-tail-different-history'))+'\n')
            events=[];stopped=False;cleanups=0
            config={'cases_file':str(cases),'model':{'hybrid':False},
                    'kinds':['same_content_fill_overlap'] if concurrent else []}
            def cleanup():
                nonlocal cleanups
                cleanups+=1;events.append('cleanup')
                if stopped:raise RuntimeError('HTTP cleanup after script stop')
                if defect=='cleanup' and (concurrent or cleanups==3):
                    raise RuntimeError('unknown cleanup drain')
                return {}
            def close():
                events.append('close')
                if stopped:raise RuntimeError('client close after script stop')
            def finish_script(timeout):
                nonlocal stopped
                self.assertTrue((directory/'driver-stop.json').exists())
                stopped=True;events.append('script_stop')
                if defect=='script':raise RuntimeError('script failed at exit')
            future=SimpleNamespace(done=lambda:stopped,result=finish_script)
            pool=SimpleNamespace(submit=lambda *a,**k:future,
                shutdown=lambda **k:events.append('pool_shutdown'))
            store=SimpleNamespace(_lock=threading.Lock(),_capture_fences={},
                _capture_pending_begin={},health=lambda:{'capture_fences':0})
            client=SimpleNamespace(tp_size=1,timeout=1,store=store,cleanup=cleanup,close=close,
                submit=lambda *a,**k:(None,None),fetch=lambda *a,**k:({},None))
            server=SimpleNamespace(_base_url='http://unused',
                shutdown=lambda:events.append('server_shutdown'))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(driver,'source_manifest',return_value={'source':'fixed'}))
                stack.enter_context(mock.patch.object(driver,'wait_for'))
                stack.enter_context(mock.patch.object(driver,'ThreadPoolExecutor',return_value=pool))
                stack.enter_context(mock.patch('actual_faults.launcher.start',return_value=server))
                stack.enter_context(mock.patch('actual_faults.scenarios.Client',return_value=client))
                stack.enter_context(mock.patch('actual_faults.scenarios.command',return_value=[]))
                stack.enter_context(mock.patch('actual_faults.concurrent_fill.run_case',return_value={'passed':True}))
                result=driver.run(config,directory)
            self.assertEqual(result['passed'],defect=='none')
            self.assertLess(events.index('close'),events.index('script_stop'))
            self.assertLess(max(i for i,e in enumerate(events) if e=='cleanup'),events.index('script_stop'))
            self.assertLess(events.index('script_stop'),events.index('server_shutdown'))
            self.assertEqual(events.count('script_stop'),1)
            self.assertEqual(json.loads((directory/'result.json').read_text())['passed'],result['passed'])
            if defect=='cleanup':
                self.assertIn('unknown cleanup drain',result['cleanup_error'])
                self.assertTrue((directory/'unresolved-ownership.json').exists())
            elif defect=='script':
                self.assertIn('script failed at exit',result['script_stop_error'])
                self.assertTrue((directory/'script-failure.txt').exists())

    def test_normal_path_drains_before_stopping(self):self.check_path(False,'none')
    def test_concurrent_path_drains_before_stopping(self):self.check_path(True,'none')
    def test_normal_cleanup_failure_remains_failed(self):self.check_path(False,'cleanup')
    def test_concurrent_cleanup_failure_remains_failed(self):self.check_path(True,'cleanup')
    def test_normal_script_failure_remains_failed(self):self.check_path(False,'script')
    def test_concurrent_script_failure_remains_failed(self):self.check_path(True,'script')


if __name__=='__main__':unittest.main()
