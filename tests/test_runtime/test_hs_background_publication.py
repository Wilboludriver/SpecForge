"""CPU ownership gates for the opt-in publication lane; no CUDA/TP simulation."""
import os
import threading
import time
import unittest
from concurrent.futures import Future
from unittest import mock

from tests.test_runtime import test_hs_plugin as fixtures
from specforge_sglang_capture.hs.publication import PublicationLane


class BackgroundPublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.HSPluginTests()
        self.fixture.setUp()
        self.env = mock.patch.dict(os.environ, {"SPECFORGE_HS_BACKGROUND_PUBLISH": "1",
            "SPECFORGE_HS_PUBLISH_TICKETS": "8", "SPECFORGE_HS_PUBLISH_BATCH": "8",
            "SPECFORGE_HS_PUBLISH_WAIT_MS": "1"})
        self.env.start()

    def tearDown(self):
        for runtime in self.fixture.runtimes:
            if runtime.publication_lane is not None and runtime.quarantined_publications:
                runtime.publication_lane.close()
        self.fixture.tearDown()
        self.env.stop()

    def prepare(self, hs, name):
        req = self.fixture.request(hs, name)
        self.fixture.admit(hs, req)
        return req

    def wait(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(.001)

    def transfer(self, deferred, actual):
        try:
            value = actual.result(2)
        except Exception as exc:
            deferred.set_exception(exc)
        else:
            deferred.set_result(value)

    def delayed_first(self, hs):
        actual, deferred, real = hs.controller.submit, Future(), []
        def submit(*args, **kwargs):
            future = actual(*args, **kwargs)
            if not real:
                real.append(future)
                return deferred
            return future
        return mock.patch.object(hs.controller, "submit", submit), deferred, real

    def test_capacity_fallback_is_permanent_and_does_not_block_other_job(self):
        with mock.patch.dict(os.environ, {"SPECFORGE_HS_PUBLISH_TICKETS": "1"}):
            hs = self.fixture.runtime()
        first = self.prepare(hs, "late")
        second = self.prepare(hs, "fallback")
        patch, deferred, real = self.delayed_first(hs)
        with patch:
            one = self.fixture.forward(hs, [first], [8]).hs_jobs[0]
            two = self.fixture.forward(hs, [second], [8]).hs_jobs[0]
        self.assertIsNotNone(one.publication_ticket)
        self.assertIsNone(two.publication_ticket)
        self.fixture.spin(hs, two.output.done)
        self.assertFalse(one.output.done())
        self.assertEqual(hs.publication_lane.stats()["live_tickets"], 1)
        self.transfer(deferred, real[0])
        self.fixture.spin(hs, one.output.done)
        self.assertEqual(hs.publication_lane.stats()["attached"], 1)
        self.assertEqual(hs.publication_lane.stats()["fallback"], 1)
        self.assertEqual(hs.publication_lane.stats()["live_role_bytes"], 0)
        self.assertEqual(len(hs.capture.sink._store.values), 6)

    def test_pre_enqueue_cancel_late_materialize_has_no_publication(self):
        hs = self.fixture.runtime()
        req = self.prepare(hs, "cancel-late")
        patch, deferred, real = self.delayed_first(hs)
        with patch:
            job = self.fixture.forward(hs, [req], [8]).hs_jobs[0]
        hs.deferred[-1][1].cancel()
        hs.poll()
        self.assertIn(job.execution, hs.jobs)
        self.assertEqual(hs.publication_lane.stats()["live_tickets"], 1)
        self.transfer(deferred, real[0])
        self.fixture.spin(hs, lambda: not hs.jobs)
        self.assertEqual(hs.capture.sink._store.values, {})
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)
        self.assertEqual(hs.publication_lane.stats()["live_tickets"], 0)

    def test_post_enqueue_cancel_waits_real_sink_and_cleans_exact_keys(self):
        gate = threading.Event()
        store = fixtures._Store(gate)
        hs = self.fixture.runtime(store=store)
        req = self.prepare(hs, "cancel-put")
        job = self.fixture.forward(hs, [req], [8]).hs_jobs[0]
        self.assertTrue(store.entered.wait(2))
        hs.deferred[-1][1].cancel()
        hs.poll()
        self.assertFalse(job.output.done())
        self.assertGreater(hs.controller.stats()["output_bytes"], 0)
        self.assertTrue(store.registered)
        gate.set()
        self.fixture.spin(hs, lambda: not hs.jobs)
        self.assertFalse(store.values)
        self.assertFalse(store.registered)
        self.assertEqual(hs.publication_lane.stats()["live_role_bytes"], 0)

    def test_done_unobserved_retains_budget_and_main_is_only_certificate_owner(self):
        hs = self.fixture.runtime()
        req = self.prepare(hs, "unobserved")
        job = self.fixture.forward(hs, [req], [8]).hs_jobs[0]
        self.wait(lambda: (handback := hs.publication_lane.observe(job.publication_ticket))
                  is not None and handback.sink is not None and handback.sink.done())
        self.assertFalse(job.output.done())
        self.assertFalse(job.entry.transfer_certificate)
        self.assertGreater(hs.controller.stats()["output_bytes"], 0)
        self.assertEqual(hs.publication_lane.stats()["live_tickets"], 1)
        hs.deferred[-1][1].cancel()
        self.fixture.spin(hs, lambda: not hs.jobs)
        self.assertFalse(job.entry.transfer_certificate)
        self.assertFalse(hs.capture.sink._store.values)
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)

    def test_cancel_during_submit_waits_source_alias_handback(self):
        hs = self.fixture.runtime()
        req = self.prepare(hs, "cancel-commit")
        entered, allow = threading.Event(), threading.Event()
        actual = hs.capture.sink.submit_samples
        def gated(samples):
            entered.set()
            if not allow.wait(3):
                raise RuntimeError("test submit gate timed out")
            return actual(samples)
        with mock.patch.object(hs.capture.sink, "submit_samples", gated):
            job = self.fixture.forward(hs, [req], [8]).hs_jobs[0]
            self.assertTrue(entered.wait(2))
            hs.deferred[-1][1].cancel()
            hs.poll()
            self.assertIsNone(hs.publication_lane.observe(job.publication_ticket))
            self.assertFalse(job.output.done())
            self.assertGreater(hs.controller.stats()["output_bytes"], 0)
            allow.set()
            self.fixture.spin(hs, lambda: not hs.jobs)
        self.assertFalse(hs.capture.sink._store.values)
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)

    def test_partial_put_drains_and_unknown_put_retains_ticket(self):
        for status in (-1, -800):
            with self.subTest(status=status):
                store = fixtures._Store(status=status)
                hs = self.fixture.runtime(store=store)
                req = self.prepare(hs, str(status))
                job = self.fixture.forward(hs, [req], [8]).hs_jobs[0]
                self.fixture.spin(hs, lambda: not hs.jobs)
                self.assertFalse(job.entry.transfer_certificate)
                self.assertIsNotNone(job.output.exception())
                if status == -1:
                    self.assertFalse(store.registered)
                    self.assertFalse(store.values)
                    self.assertEqual(hs.controller.stats()["output_bytes"], 0)
                    self.assertEqual(hs.publication_lane.stats()["live_tickets"], 0)
                else:
                    self.assertTrue(store.registered)
                    self.assertGreater(hs.controller.stats()["output_bytes"], 0)
                    self.assertEqual(hs.publication_lane.stats()["live_tickets"], 1)
                    with self.assertRaisesRegex(RuntimeError, "quarantined"):
                        hs.shutdown()
                    self.assertFalse(hs.publication_lane.stats()["thread_alive"])

    def test_shared_batch_cancel_preserves_sibling_and_diagnostic_identity(self):
        gate = threading.Event()
        store = fixtures._Store(gate)
        with mock.patch.dict(os.environ, {"SPECFORGE_HS_DIAGNOSTICS": "1",
                                         "SPECFORGE_HS_PUBLISH_WAIT_MS": "20"}):
            hs = self.fixture.runtime(store=store)
        reqs = [self.prepare(hs, name) for name in ("cancel", "sibling")]
        jobs = self.fixture.forward(hs, reqs, [8, 8], wait_materialize=True).hs_jobs
        try:
            self.assertTrue(store.entered.wait(2))
            self.wait(lambda: all(hs.publication_lane.observe(job.publication_ticket) for job in jobs))
            hs.poll()
            self.assertIs(jobs[0].sink, jobs[1].sink)
            self.assertEqual(jobs[0].sink_count, 2)
            hs.deferred[0][1].cancel()
        finally:
            gate.set()
        self.fixture.spin(hs, lambda: not hs.jobs)
        result = jobs[1].output.result()
        self.assertTrue(all(key in store.values for key in result["physical_keys"].values()))
        self.assertFalse(any(jobs[0].spec["reservation_token"] in key for key in store.values))
        self.assertTrue(jobs[1].diagnostic["p0_v1_publication_lane"])
        self.assertEqual(jobs[1].diagnostic["p0_v1_sink_batch_samples"], 2)
        self.assertLessEqual(jobs[1].diagnostic["p0_v1_runtime_sink_enqueue_ns"],
                             jobs[1].diagnostic["p0_v1_owner_observed_ns"])
        self.assertEqual(hs.publication_lane.stats()["live_role_bytes"], 0)

    def test_unknown_enqueue_acceptance_retains_runtime_owner_and_ticket(self):
        gate = threading.Event()
        store = fixtures._Store(gate)
        hs = self.fixture.runtime(store=store)
        req = self.prepare(hs, "unknown-accept")
        actual, accepted = hs.capture.sink._executor.submit, []
        def uncertain(*args, **kwargs):
            future = actual(*args, **kwargs)
            accepted.append(future)
            if not store.entered.wait(2):
                raise RuntimeError("test native reader did not start")
            raise RuntimeError("accepted then failed")
        try:
            with mock.patch.object(hs.capture.sink._executor, "submit", uncertain):
                job = self.fixture.forward(hs, [req], [8]).hs_jobs[0]
                self.fixture.spin(hs, lambda: bool(hs.quarantined_publications))
            self.assertFalse(job.entry.transfer_certificate)
            self.assertIsNotNone(job.sample)
            self.assertTrue(store.registered)
            self.assertGreater(hs.controller.stats()["output_bytes"], 0)
            self.assertEqual(hs.publication_lane.stats()["live_tickets"], 1)
        finally:
            gate.set()
        accepted[0].result(2)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            hs.shutdown()

    def test_shutdown_waits_late_materialization_and_joins_lane(self):
        hs = self.fixture.runtime()
        req = self.prepare(hs, "shutdown")
        patch, deferred, real = self.delayed_first(hs)
        with patch:
            self.fixture.forward(hs, [req], [8])
        timer = threading.Timer(.02, lambda: self.transfer(deferred, real[0]))
        timer.start()
        hs.shutdown()
        timer.join()
        self.assertFalse(hs.jobs)
        self.assertFalse(hs.publication_lane.stats()["thread_alive"])
        self.assertEqual(hs.controller.stats()["output_bytes"], 0)
        self.assertEqual(hs.publication_lane.stats()["live_tickets"], 0)

    def test_role_byte_limit_is_independent_of_ticket_count(self):
        lane = PublicationLane(object(), max_tickets=4, max_bytes=100, batch_samples=4)
        try:
            ticket = lane.try_attach(1, {}, Future(), 60)
            self.assertIsNotNone(ticket)
            self.assertIsNone(lane.try_attach(2, {}, Future(), 41))
            self.assertEqual(lane.stats()["live_tickets"], 1)
            self.assertEqual(lane.stats()["live_role_bytes"], 60)
            lane.revoke(ticket)
            self.assertEqual(lane.stats()["live_role_bytes"], 60)
            lane.acknowledge(ticket)
            self.assertEqual(lane.stats()["live_role_bytes"], 0)
        finally:
            lane.close()

    def test_lane_coalesces_ready_siblings_and_freezes_spec(self):
        class Sink:
            def __init__(self):
                self.batch = []
            def submit_samples(self, samples):
                self.batch.append(tuple(item[0]["id"] for item in samples))
                future = Future()
                future.set_result([])
                return future
        sink = Sink()
        lane = PublicationLane(sink, max_tickets=4, max_bytes=400, batch_samples=4, batch_wait_ms=20)
        try:
            futures = [Future() for _ in range(4)]
            tickets = []
            for index, future in enumerate(futures):
                spec = {"id": index}
                tickets.append(lane.try_attach(index, spec, future, 100))
                spec["id"] = -1
            self.assertIsNone(lane.try_attach(5, {}, Future(), 1))
            for future in futures:
                future.set_result(type("Sample", (), {"features": {"aux": None, "last_hidden": None}})())
            self.wait(lambda: all(lane.observe(ticket) for ticket in tickets))
            self.assertEqual(len(sink.batch), 1)
            self.assertEqual(set(sink.batch[0]), set(range(4)))
            self.assertEqual(lane.stats()["live_role_bytes"], 400)
            for ticket in tickets:
                lane.acknowledge(ticket)
            self.assertEqual(lane.stats()["live_role_bytes"], 0)
        finally:
            lane.close()


if __name__ == "__main__":
    unittest.main()
