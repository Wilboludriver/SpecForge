"""The opt-in overlap experiment keeps CPU producer/lease ownership sequential."""
import os
import unittest
from unittest import mock

import torch

from tests.test_runtime import test_hs_runtime as fixtures


class PrefixOverlapCPUContractTest(unittest.TestCase):
    plan = fixtures.ControllerTests.plan
    submit = fixtures.ControllerTests.submit
    take = fixtures.ControllerTests.take

    def setUp(self):
        self.flag = mock.patch.dict(os.environ, {'SPECFORGE_HS_OVERLAP_PREFIX_COPY':'1'})
        self.flag.start()
        fixtures.ControllerTests.setUp(self)

    def tearDown(self):
        fixtures.ControllerTests.tearDown(self)
        self.flag.stop()

    def test_cpu_input_waits_for_producer_before_prefix_and_allocates_once(self):
        fixtures.populate(self.cache, tuple(range(8)))
        plan = self.plan(n=9, L=5)
        gate = fixtures.GateEvent()
        copy = self.cache.copy_prefix_into
        with mock.patch.object(self.cache, 'copy_prefix_into', wraps=copy) as prefix:
            future = self.submit(plan, gate)
            try:
                self.assertTrue(gate.entered.wait(2))
                self.assertEqual(prefix.call_count, 0)
                self.assertEqual(self.cache.stats()['active_leases'], 1)
                self.assertEqual(self.controller.stats()['output_bytes'], 9*fixtures.schema().row_bytes)
            finally:
                gate.finished.set()
            sample = self.take(future)
            self.assertEqual(prefix.call_count, 1)
        for name, tensor in sample.features.items():
            self.assertTrue(torch.equal(tensor, fixtures.features(9)[name]))
        self.assertNotIn('p3_v1_timing_schema_version', sample.timings)
        self.assertIn('d2h_wait_ms', sample.timings)
        self.assertIn('assembly_ms', sample.timings)
        self.assertEqual(self.cache.stats()['active_leases'], 0)


if __name__ == '__main__':
    unittest.main()
