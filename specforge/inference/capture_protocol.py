"""Version-one capture control client and serializable sample writer fences."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field


class CaptureControlRejected(RuntimeError):
    """The server explicitly rejected the operation without granting a permit."""


def request_digest(tokens, spec):
    value = {"tokens": list(tokens), "features": spec.get("features", {}),
             "passthrough": spec.get("passthrough", [])}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def http_control(url, json_body, timeout):
    import requests
    response = requests.post(url, json=json_body, timeout=timeout)
    response.raise_for_status()
    return response.json()


class CaptureProtocol:
    def __init__(self, base_url, post, timeout):
        self.url = base_url.rstrip('/') + '/spec_capture/control'
        self.post = post
        self.timeout = timeout
        self.negotiated = False

    def call(self, body):
        result = self.post(self.url, json_body={"protocol_version": 1, **body}, timeout=self.timeout)
        if not isinstance(result, dict) or result.get('error'):
            raise RuntimeError('capture control failed: ' + repr(result))
        return result

    def capability(self):
        try:
            result = self.call({'action': 'capability'})
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status == 404 and not self.negotiated:
                return None
            raise
        if result.get('protocol_version') != 1:
            if self.negotiated:
                raise RuntimeError('negotiated capture protocol disappeared')
            return None
        required = ('server_epoch', 'capture_namespace', 'capture_signature', 'schema', 'cache_enabled')
        if any(name not in result for name in required):
            raise RuntimeError('incomplete capture capability')
        self.negotiated = True
        return result

    def begin(self, capability, spec, tokens, attempt_id):
        identity = {name: spec[name] for name in ('store_id', 'sample_id', 'gen')}
        reply = self.call({'action': 'begin', **identity, 'server_epoch': capability['server_epoch'],
                           'attempt_id': attempt_id, 'request_digest': request_digest(tokens, spec)})
        for name, value in {**identity, 'attempt_id': attempt_id, 'server_epoch': capability['server_epoch']}.items():
            if reply.get(name) != value:
                raise RuntimeError('capture begin identity mismatch: ' + name)
        if not reply.get('reservation_token'):
            raise RuntimeError('capture begin omitted server reservation')
        return {**identity, 'protocol_version': 1, 'server_epoch': capability['server_epoch'],
                'attempt_id': attempt_id, 'reservation_token': reply['reservation_token'],
                'control_url': self.url, 'timeout_s': self.timeout}


@dataclass
class WriterFence:
    """A sample's writer identity; serializable fields travel on every SampleRef."""
    wire: dict
    names: tuple
    post: object = http_control
    writer_drained: bool = False
    removed: bool = False
    acknowledged: bool = False
    adopted: bool = False
    cleanup_owner: str = field(default_factory=lambda: uuid.uuid4().hex)
    cleanup_token: str | None = None
    remote_retired: bool = False
    lock: object = field(default_factory=threading.RLock)

    @property
    def token(self):
        return self.wire['reservation_token']

    def control(self, action, **values):
        body = {name: self.wire[name] for name in ('server_epoch', 'store_id', 'sample_id', 'gen', 'attempt_id', 'reservation_token')}
        try:
            result = self.post(self.wire['control_url'], json_body={'protocol_version': 1, 'action': action, **body, **values},
                               timeout=min(float(self.wire.get('timeout_s', 30.0)), 30.0))
        except Exception as exc:
            if getattr(exc, 'capture_control_rejected', False) or getattr(getattr(exc, 'response', None), 'status_code', None) == 409:
                raise CaptureControlRejected(str(exc)) from None
            raise
        if isinstance(result, dict) and result.get('error'):
            raise CaptureControlRejected('capture fence control rejected: ' + str(result['error']))
        if not isinstance(result, dict):
            raise RuntimeError('capture fence control returned invalid response')
        if any(result.get(name) != value for name, value in body.items()):
            raise RuntimeError('capture fence identity changed')
        return result

    def drain(self, *, cancel=True):
        if self.writer_drained:
            return
        deadline = time.monotonic() + float(self.wire.get('timeout_s', 30.0))
        while True:
            result = self.control('resolve', cancel=cancel)
            if result.get('writer_drained'):
                self.writer_drained = True
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('capture writer has not drained; sample retained')
            time.sleep(0.01)

    def acquire_read(self, reader_owner):
        reply = self.control('read_acquire', reader_owner=reader_owner)
        if not reply.get('reader_token') or reply.get('reader_owner') != reader_owner:
            raise RuntimeError('server did not grant exact read ownership')
        return reply['reader_token']

    def release_read(self, reader_owner, reader_token):
        reply = self.control('read_release', reader_owner=reader_owner, reader_token=reader_token)
        if not reply.get('reader_released') or reply.get('reader_token') != reader_token:
            raise RuntimeError('server did not release exact read ownership')

    def acquire_cleanup(self):
        # Cached writer_drained is never permission to delete deterministic g1 keys.
        self.drain(cancel=True)
        reply = self.control('cleanup_acquire', cleanup_owner=self.cleanup_owner)
        if reply.get('state') == 'RETIRED':
            self.remote_retired = self.removed = self.acknowledged = True
            return
        if not reply.get('cleanup_token') or reply.get('cleanup_owner') != self.cleanup_owner:
            raise RuntimeError('server did not grant exact cleanup ownership')
        self.cleanup_token = reply['cleanup_token']

    def acknowledge(self):
        if self.acknowledged:
            return
        if not self.writer_drained or not self.removed:
            raise RuntimeError('cleanup acknowledgement before writer/remove drain')
        if not self.cleanup_token:
            raise RuntimeError('cleanup acknowledgement without server permit')
        result = self.control('cleanup_done', cleanup_token=self.cleanup_token)
        if result.get('state') != 'RETIRED':
            raise RuntimeError('capture cleanup was not retired')
        self.acknowledged = True
