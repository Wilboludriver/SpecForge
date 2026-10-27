"""Test-only real CUDA stream gate, released by cancel or a bounded watchdog."""
from __future__ import annotations
import ctypes
import threading


class CudaCopyGate:
    def __init__(self, device, stream, *, timeout_s, emit):
        self.device=device;self.stream=stream;self.emit=emit
        self.timeout_s=timeout_s
        self.flag=self.release_stream=self.initialized=self.watchdog=None
        self._lock=threading.Lock();self._released=threading.Event()
        self.reason=None;self.error=None
        self.driver=ctypes.CDLL('libcuda.so.1')
        for name in ('cuStreamWaitValue32_v2','cuStreamWriteValue32_v2'):
            function=getattr(self.driver,name)
            function.argtypes=(ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint)
            function.restype=ctypes.c_int

    def arm(self):
        """Caller must persist this owner before any CUDA allocation or wait."""
        import torch
        # Created on the controller's stream, so zero initialization precedes
        # the native wait. Keep this storage alive through the real DMA drain.
        self.flag=torch.zeros(1,device=self.device,dtype=torch.int32)
        self.release_stream=torch.cuda.Stream(device=self.device)
        self.initialized=torch.cuda.Event()
        self.initialized.record(self.stream)
        self.release_stream.wait_event(self.initialized)
        self.flag.record_stream(self.release_stream)
        self.watchdog=threading.Thread(target=self._watch, args=(self.timeout_s,),
                                       name='actual-cuda-gate-watchdog',daemon=True)
        self.watchdog.start()
        try:
            # CU_STREAM_WAIT_VALUE_EQ: a real device-side wait, no CPU Event
            # is used as a substitute for the observed pending CUDA event.
            self._check(self.driver.cuStreamWaitValue32_v2(int(self.stream.cuda_stream),self.flag.data_ptr(),1,1),'wait')
        except BaseException:
            # Even an unsuccessful native arm may have uncertain side effects.
            # Keep this owner retained and attempt a bounded safety release.
            self.release('arm_error')
            raise

    @staticmethod
    def _check(code, operation):
        if code:
            raise RuntimeError(f'CUDA native stream {operation} failed with driver status {code}')

    def _watch(self, timeout_s):
        if not self._released.wait(timeout_s):
            try:self.release('watchdog')
            except BaseException as exc:
                self.error=repr(exc)
                self.emit('cuda_gate_watchdog_failed',error=self.error)

    def release(self, reason):
        import torch
        with self._lock:
            if self._released.is_set():
                return self.reason
            try:
                with torch.cuda.device(self.device):
                    self._check(self.driver.cuStreamWriteValue32_v2(
                        int(self.release_stream.cuda_stream),self.flag.data_ptr(),1,0),'release')
                self.reason=reason
                self.emit('cuda_gate_released',reason=reason,flag_pointer=self.flag.data_ptr())
            except BaseException as exc:
                self.error=repr(exc)
                raise
            finally:
                self._released.set()
            return self.reason

    def close_after_drain(self):
        if not self._released.is_set() or self.error:
            raise AssertionError('CUDA test gate lacks a proven successful release')
        self.watchdog.join(timeout=1)
        if self.watchdog.is_alive():
            raise AssertionError('CUDA test gate watchdog remains active')
        self.release_stream.synchronize()
        self.flag=None
