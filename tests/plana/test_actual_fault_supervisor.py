"""No service/GPU starts: owned cleanup failures must not skip other groups."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

FILE = Path(__file__).resolve().parents[2] / 'scripts/plana/run_actual_faults.py'
spec = importlib.util.spec_from_file_location('actual_fault_supervisor', FILE)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class OwnedCleanupTests(unittest.TestCase):
    def test_driver_cleanup_error_does_not_skip_mooncake(self):
        events = []
        child = SimpleNamespace(poll=lambda: None)
        services = [object()]
        def terminate(processes, **kwargs):
            self.assertEqual(processes, [child])
            events.append('driver')
            raise RuntimeError('injected group teardown failure')
        def stop(started, **kwargs):
            self.assertIs(started, services)
            events.append('mooncake')
        errors = supervisor.finish_owned_processes(child, services, terminate, stop)
        self.assertEqual(events, ['driver', 'mooncake'])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['phase'], 'fault-driver teardown')

    def test_exited_leader_still_has_owned_descendant_cleanup(self):
        child = SimpleNamespace(poll=lambda: 0)
        def terminate(processes, **kwargs):
            self.assertEqual(processes, [])
            self.assertEqual(kwargs['exited_group_leaders'], [child])
        errors = supervisor.finish_owned_processes(child, [], terminate, lambda *a, **kw: None)
        self.assertEqual(errors, [])

    def test_both_cleanup_errors_are_retained(self):
        def fail(*args, **kwargs):
            raise RuntimeError('injected teardown failure')
        errors = supervisor.finish_owned_processes(SimpleNamespace(poll=lambda: None), [], fail, fail)
        self.assertEqual([row['phase'] for row in errors], ['fault-driver teardown', 'Mooncake teardown'])


if __name__ == '__main__':
    unittest.main()
