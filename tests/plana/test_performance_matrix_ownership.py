"""CPU process ownership tests; no GPU, model, transport or services."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

FILE=Path(__file__).resolve().parents[2]/'scripts/plana/run_performance_matrix.py'
spec=importlib.util.spec_from_file_location('performance_matrix_ownership',FILE)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class OwnershipTests(unittest.TestCase):
    def test_existing_child_refuses_boundary_without_signalling(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        try:
            with mock.patch.object(os,'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError,'already contains children'):
                    runner.enter_owned_boundary()
                kill.assert_not_called()
            self.assertIsNone(child.poll())
        finally:
            child.terminate();child.wait(timeout=5)

    def test_fast_double_fork_survives_leader_then_is_adopted_and_drained(self):
        previous=runner.enter_owned_boundary()
        child=None;tree=None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                marker=Path(tmp)/'grandchild'
                code='''import os,sys,time
pid=os.fork()
if pid==0:
 os.setsid()
 leaf=os.fork()
 if leaf==0:
  open(sys.argv[1],'w').write(str(os.getpid()))
  time.sleep(30)
  os._exit(0)
 os._exit(0)
os.waitpid(pid,0)
os._exit(0)
'''
                child=subprocess.Popen([sys.executable,'-c',code,str(marker)],start_new_session=True)
                child.wait(timeout=5)
                deadline=time.monotonic()+5
                while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
                leaf=int(marker.read_text())
                # Construct only AFTER the original leader has exited: ordinary
                # ancestry polling can no longer discover this detached leaf.
                tree=runner.OwnedTree(child)
                self.assertIn(leaf,[row['pid'] for row in tree.live()])
                report=tree.drain(.1)
                self.assertFalse(runner.child_pids(os.getpid()))
                self.assertIn(leaf,[row['pid'] for row in report['reaped_orphans']])
        finally:
            if child is not None:
                if tree is None:tree=runner.OwnedTree(child)
                tree.drain(.1)
            runner.leave_owned_boundary(previous)

    def test_adopted_zombie_is_terminal_and_reaped_not_killed(self):
        previous=runner.enter_owned_boundary()
        child=None;tree=None
        try:
            code='''import os
if os.fork()==0:os._exit(0)
os._exit(0)
'''
            child=subprocess.Popen([sys.executable,'-c',code],start_new_session=True)
            child.wait(timeout=5)
            time.sleep(.05)
            with mock.patch.object(os,'killpg') as kill:
                tree=runner.OwnedTree(child)
                report=tree.drain(.1)
                kill.assert_not_called()
            self.assertTrue(report['reaped_orphans'])
            self.assertFalse(runner.child_pids(os.getpid()))
        finally:
            if child is not None:
                if tree is None:tree=runner.OwnedTree(child)
                tree.drain(.1)
            runner.leave_owned_boundary(previous)

    def test_child_spawned_during_term_is_discovered_after_leader_exit(self):
        previous=runner.enter_owned_boundary();child=tree=None
        try:
            code='''import os,signal,time

def stop(sig,frame):
 if os.fork()==0:
  os.setsid()
  signal.signal(signal.SIGTERM,signal.SIG_DFL)
  time.sleep(30)
  os._exit(0)
 os._exit(0)
signal.signal(signal.SIGTERM,stop)
print('ready',flush=True)
while True:time.sleep(1)
'''
            child=subprocess.Popen([sys.executable,'-c',code],start_new_session=True,stdout=subprocess.PIPE)
            self.assertEqual(child.stdout.readline(),b'ready\n')
            tree=runner.OwnedTree(child)
            report=tree.drain(.2)
            self.assertGreaterEqual(len(report['identities']),2)
            self.assertFalse(runner.child_pids(os.getpid()))
        finally:
            if child is not None:
                if tree is None:tree=runner.OwnedTree(child)
                tree.drain(.1)
                child.stdout.close()
            runner.leave_owned_boundary(previous)

    def test_runner_and_matrix_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix=Path(tmp)/'matrix.json';matrix.write_text('{}')
            contract={'runner_sha256':runner.file_sha(FILE),'matrix_sha256':runner.file_sha(matrix)}
            runner.verify_runner_contract(contract,matrix)
            matrix.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'frozen matrix/runner changed'):
                runner.verify_runner_contract(contract,matrix)
            matrix.write_text('{}');contract['runner_sha256']='0'*64
            with self.assertRaisesRegex(RuntimeError,'frozen matrix/runner changed'):
                runner.verify_runner_contract(contract,matrix)

    def test_subreaper_failure_never_launches(self):
        fake=mock.Mock();fake.prctl.return_value=-1
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner.ctypes,'CDLL',return_value=fake), mock.patch.object(subprocess,'Popen') as spawn:
            with self.assertRaises(OSError):
                runner.run_child(['unused'],{},Path(tmp)/'log',timeout_s=1,grace_s=1,checkpoint=lambda x:None)
            spawn.assert_not_called()


if __name__=='__main__':unittest.main()
