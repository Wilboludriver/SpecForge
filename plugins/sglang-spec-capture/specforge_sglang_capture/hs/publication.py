"""Bounded publication authorization, independent of scheduler and TP state.

Only immutable descriptors and existing materialization Futures cross this lane.
The controller remains the sole physical allocation accountant. Ticket bytes are
an additional role limit, retained until the scheduler acknowledges terminal work.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import Future
from types import MappingProxyType

from .attempts import scalar_error


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("publication descriptor must contain only JSON scalars")


@dataclasses.dataclass(frozen=True)
class PublicationDescriptor:
    execution: object
    spec: object
    materialize: Future
    host_bytes: int


@dataclasses.dataclass(frozen=True)
class Handback:
    sink: Future | None = None
    index: int = 0
    count: int = 0
    enqueue_ns: int = 0
    publish_started: float = 0.0
    error: BaseException | None = None


@dataclasses.dataclass(eq=False)
class PublicationTicket:
    descriptor: PublicationDescriptor
    phase: str = "waiting"
    handback: Handback | None = None


class PublicationLane:
    def __init__(self, sink, *, max_tickets=8, max_bytes, batch_samples=8,
                 batch_wait_ms=1.0):
        if not 1 <= max_tickets <= 4096 or max_bytes <= 0:
            raise ValueError("invalid publication ticket capacity")
        if not 1 <= batch_samples <= max_tickets or not 0 <= batch_wait_ms <= 100:
            raise ValueError("invalid publication batch bounds")
        self._sink = sink
        self._max_tickets, self._max_bytes = max_tickets, max_bytes
        self._batch_samples, self._batch_wait = batch_samples, batch_wait_ms / 1000
        self._condition = threading.Condition()
        self._tickets = {}
        self._bytes = 0
        self._closing = self._poisoned = False
        self._inflight = None
        self._counts = dict(attached=0, fallback=0, batches=0, samples=0,
                            max_live_tickets=0, max_live_bytes=0)
        self._thread = threading.Thread(target=self._run,
            name="hs-publication-lane", daemon=True)
        try:
            self._thread.start()
        except BaseException:
            with self._condition:
                self._closing = True
                self._condition.notify_all()
            if self._thread.is_alive():
                self._thread.join()
            raise

    def _wake(self, _future=None):
        with self._condition:
            self._condition.notify_all()

    def try_attach(self, execution, spec, materialize, host_bytes):
        # Main calls once, only after consume authorization. An unsuccessful
        # attach permanently selects the original main-poll publication owner.
        with self._condition:
            if (self._closing or self._poisoned or len(self._tickets) >= self._max_tickets
                    or host_bytes > self._max_bytes - self._bytes):
                self._counts["fallback"] += 1
                return None
            ticket = PublicationTicket(PublicationDescriptor(
                execution, _freeze(spec), materialize, host_bytes))
            self._tickets[ticket] = None
            self._bytes += host_bytes
            self._counts["attached"] += 1
            self._counts["max_live_tickets"] = max(self._counts["max_live_tickets"], len(self._tickets))
            self._counts["max_live_bytes"] = max(self._counts["max_live_bytes"], self._bytes)
        materialize.add_done_callback(self._wake)
        self._wake()
        return ticket

    def revoke(self, ticket):
        with self._condition:
            # This lock also protects waiting -> committed, the enqueue
            # linearization point. A committed reader must return its handback.
            if ticket.phase == "waiting":
                ticket.phase = "handed"
                ticket.handback = Handback(error=RuntimeError("publication authorization revoked"))
            self._condition.notify_all()

    def observe(self, ticket):
        with self._condition:
            return ticket.handback if ticket.phase == "handed" else None

    def acknowledge(self, ticket):
        with self._condition:
            if ticket not in self._tickets:
                raise RuntimeError("publication ticket acknowledged twice")
            if ticket.phase != "handed":
                raise RuntimeError("publication source reader has not drained")
            future = ticket.handback.sink
            if future is not None:
                if not future.done():
                    raise RuntimeError("publication writer has not drained")
                error = self._error(future)
                if error is not None and getattr(error, "capture_writer_drained", True) is False:
                    raise RuntimeError("cannot acknowledge unknown writer drain")
            self._tickets.pop(ticket)
            self._bytes -= ticket.descriptor.host_bytes
            self._condition.notify_all()

    @staticmethod
    def _error(future):
        try:
            return future.exception()
        except BaseException as exc:
            return scalar_error(exc)

    def poisoned(self):
        with self._condition:
            if self._inflight is not None and self._inflight.done():
                error = self._error(self._inflight)
                if error is not None and getattr(error, "capture_writer_drained", True) is False:
                    self._poisoned = True
            if self._poisoned:
                for ticket in self._tickets:
                    self.revoke(ticket)
            return self._poisoned

    def stats(self):
        with self._condition:
            return {**self._counts, "live_tickets": len(self._tickets), "live_role_bytes": self._bytes,
                    "ticket_limit": self._max_tickets, "role_byte_limit": self._max_bytes,
                    "batch_samples": self._batch_samples, "batch_wait_ms": self._batch_wait * 1000,
                    "poisoned": self._poisoned, "thread_alive": self._thread.is_alive()}

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join()

    def _submit(self, tickets):
        """Return only handles/scalars; this entire frame dies before handback."""
        samples, selected, failures = [], [], {}
        sample = None
        try:
            for ticket in tickets:
                try:
                    sample = ticket.descriptor.materialize.result()
                    samples.append((ticket.descriptor.spec, sample.features["aux"],
                                    sample.features["last_hidden"]))
                    selected.append(ticket)
                except BaseException as exc:
                    failures[ticket] = Handback(error=scalar_error(exc))
            enqueue_ns, started = time.monotonic_ns(), time.perf_counter()
            if selected:
                try:
                    future = self._sink.submit_samples(samples)
                except BaseException as exc:
                    future = Future()
                    future.set_exception(scalar_error(exc))
                for index, ticket in enumerate(selected):
                    failures[ticket] = Handback(future, index, len(selected), enqueue_ns, started)
                return failures, future, len(selected)
            return failures, None, 0
        finally:
            # The sink owns its own list. Never expose a release-capable
            # handback while this frame still holds output tensor aliases.
            samples.clear()
            sample = None

    def _run(self):
        while True:
            with self._condition:
                while True:
                    if self._closing:
                        return
                    if self._inflight is not None and self._inflight.done():
                        error = self._error(self._inflight)
                        if error is not None and getattr(error, "capture_writer_drained", True) is False:
                            self._poisoned = True
                        self._inflight = None
                    if self._poisoned:
                        for ticket in self._tickets:
                            self.revoke(ticket)
                    ready = [ticket for ticket in self._tickets
                             if ticket.phase == "waiting" and ticket.descriptor.materialize.done()]
                    if ready and self._inflight is None and not self._poisoned:
                        # At most one bounded aggregation wait per batch. This
                        # may admit newly ready siblings; it never waits for all.
                        if self._batch_wait:
                            self._condition.wait(self._batch_wait)
                        if self._closing:
                            return
                        ready = [ticket for ticket in self._tickets
                                 if ticket.phase == "waiting" and ticket.descriptor.materialize.done()]
                        ready = ready[:self._batch_samples]
                        if ready:
                            for ticket in ready:
                                ticket.phase = "committed"
                            break
                    self._condition.wait()
            handbacks, future, count = self._submit(ready)
            # _submit frame (and every temporary tensor alias) has ended.
            with self._condition:
                for ticket, handback in handbacks.items():
                    ticket.handback, ticket.phase = handback, "handed"
                self._inflight = future
                self._counts["batches"] += bool(count)
                self._counts["samples"] += count
                self._condition.notify_all()
            if future is not None:
                future.add_done_callback(self._wake)
