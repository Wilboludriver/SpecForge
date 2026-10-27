"""Use official scripted server/context with explicit root-assigned ports."""
from __future__ import annotations
import multiprocessing
import os
from pathlib import Path
import time
from actual_faults import atomic_json


def scheduler_entry(*args, **kwargs):
    from actual_faults.rank_hooks import install
    from sglang.srt.managers.scheduler import run_scheduler_process
    restore = install(os.environ['PLANA_ACTUAL_FAULT_DIR'])
    try:
        return run_scheduler_process(*args, **kwargs)
    finally:
        restore()


def server_entry(engine_kwargs):
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    launch_server(ServerArgs(**engine_kwargs), run_scheduler_process_func=scheduler_entry)


def start(config, directory):
    """No automatic free-port selection and no service other than owned server."""
    import zmq
    from sglang.srt.environ import envs
    from sglang.test.scripted_runtime.http_server import ScriptedHttpServer
    directory = Path(directory)
    allocation = config['allocation']
    kwargs = dict(config['engine_kwargs'])
    if kwargs.get('port') != allocation['http_port'] or kwargs.get('nccl_port') != allocation['nccl_port']:
        raise ValueError('HTTP/NCCL args must match explicit root allocation')
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    endpoint = f'tcp://127.0.0.1:{allocation["dispatch_port"]}'
    socket.bind(endpoint)
    oob = directory / 'scripted-runtime-oob.json'
    oob.write_text('')
    process = multiprocessing.get_context('spawn').Process(target=server_entry, args=(kwargs,), name='plana-owned-actual-fault-server')
    old_environment = {}
    overrides = {'PLANA_ACTUAL_FAULT_DIR': str(directory), **config['mooncake_env']}
    for name, value in overrides.items():
        old_environment[name] = os.environ.get(name)
        os.environ[name] = str(value)
    try:
        with (envs.SGLANG_TEST_SCRIPTED_RUNTIME.override(True),
              envs.SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR.override(endpoint),
              envs.SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH.override(str(oob)),
              envs.SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY.override(str(Path(__file__).parent.parent))):
            process.start()
    finally:
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    server = ScriptedHttpServer(ctx=ctx, socket=socket, server_process=process,
                                out_of_band_error_path=oob, http_port=allocation['http_port'])
    atomic_json(directory / 'owned-server.json', {'pid': process.pid, 'http_port': allocation['http_port'],
                                                 'dispatch_port': allocation['dispatch_port'], 'nccl_port': allocation['nccl_port']})
    try:
        server._await_handshake()
        server._await_http_ready()
    except BaseException:
        server._teardown()
        raise
    return server


def drive(ctx, directory):
    """One official driver script; external clients perform normal HTTP requests."""
    from actual_faults.rank_hooks import _STATES
    directory = Path(directory)
    state = next(item for item in _STATES if item.scheduler is ctx.scheduler)
    atomic_json(directory / 'driver-ready.json', {'ready': True, 'rank': state.rank})
    try:
        while not (directory / 'driver-stop.json').exists():
            state.command()
            yield
        state.emit('driver_stop', fully_idle=ctx.is_fully_idle)
        if not ctx.is_fully_idle or state.bridge.jobs or state.fixture_pins or state.cuda_gates:
            raise AssertionError('driver stopped while engine/capture work remains')
    finally:
        for gate in state.gates.values():
            gate.set()  # Owned test barriers cannot strand worker teardown.


def execute(server, function, *, args, timeout_s):
    """Official RunScript protocol with bounded poll slices for owned teardown.

    ScriptedHttpServer.execute_script uses one long poll. A short poll permits
    detecting an owned server crash without concurrently using its ZMQ socket.
    """
    from sglang.srt.managers.io_struct import sock_recv, sock_send, wrap_as_pickle
    from sglang.test.scripted_runtime.io_struct import RunScript, ScriptFailed, ScriptSucceeded
    path = function.__module__ + ':' + function.__qualname__
    sock_send(server._socket, wrap_as_pickle(RunScript(fn_path=path, args=args)))
    deadline = time.monotonic() + timeout_s
    while not server._socket.poll(100):
        if not server._server_process.is_alive():
            raise RuntimeError('owned scripted server died while executing ' + path)
        if time.monotonic() >= deadline:
            raise TimeoutError('actual fault script exceeded its root-configured suite deadline')
    reply = sock_recv(server._socket)
    if isinstance(reply, ScriptFailed):
        raise AssertionError(reply.traceback)
    if not isinstance(reply, ScriptSucceeded):
        raise RuntimeError('unexpected official script reply: ' + repr(reply))
