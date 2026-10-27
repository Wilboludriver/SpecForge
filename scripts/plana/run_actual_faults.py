#!/usr/bin/env python3
"""Own a managed Mooncake service and the explicitly allocated fault driver."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import signal
import subprocess
import sys
import time
import traceback


def finish_owned_processes(child, started, terminate, stop_services):
    """A failure stopping one owned group cannot skip the remaining services."""
    errors = []
    if child is not None:
        try:
            if child.poll() is None:
                terminate([child], grace_s=10)
            else:
                terminate([], grace_s=10, exited_group_leaders=[child])
        except BaseException as exc:
            errors.append({'phase': 'fault-driver teardown', 'error': repr(exc)})
    try:
        stop_services(started, grace_s=10)
    except BaseException as exc:
        errors.append({'phase': 'Mooncake teardown', 'error': repr(exc)})
    return errors


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'owned fault supervisor received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--service-template', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    from actual_faults.driver import validate_config, source_manifest
    from specforge.config import load_config
    from specforge.application import resolve_run
    from specforge.launch_plan import (build_launch_plan, _spawn_service,
                                      _wait_for_service, _stop_services,
                                      _terminate_processes)

    config = validate_config(json.loads(Path(args.config).read_text()))
    source = source_manifest(config)
    python_bin = str(Path(sys.executable).parent)
    os.environ['PATH'] = python_bin + os.pathsep + os.environ.get('PATH', '')
    mooncake_binary = shutil.which('mooncake_master')
    if mooncake_binary is None or str(Path(mooncake_binary).parent) != python_bin:
        raise ValueError('managed Mooncake must come from the fixed Python environment bin directory')
    template = load_config(args.service_template)
    resolved = resolve_run(template)
    plan = build_launch_plan(resolved.config, algorithm=resolved.algorithm,
                             config_path=args.service_template, env=dict(os.environ))
    service = next(service for service in plan.services if service.command.label == 'mooncake')
    capture = next(service for service in plan.services if service.command.label.startswith('capture-server-'))
    for key in ('MOONCAKE_PROTOCOL', 'MOONCAKE_MASTER_SERVER_ADDR', 'MOONCAKE_METADATA_SERVER'):
        if config['mooncake_env'][key] != capture.command.env[key]:
            raise ValueError('fault configuration and managed service differ: ' + key)
    service_ports = [int(arg.split('=', 1)[1]) for arg in service.command.argv
                     if arg.startswith(('--rpc_port=', '--http_metadata_server_port=', '--metrics_port='))]
    allocated = [config['allocation'][key] for key in ('http_port', 'dispatch_port', 'nccl_port')]
    if len(service_ports) != 3 or len(set(service_ports + allocated)) != 6:
        raise ValueError('fault server and Mooncake require six distinct explicit ports')
    manifest = {'source': source, 'config': config, 'service_template': args.service_template,
                'python': sys.executable, 'mooncake_binary': mooncake_binary,
                'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'mooncake_service': dataclasses.asdict(service),
                'explicit_ports': service_ports + allocated,
                'scope': 'plan only unless --execute; actual capture faults, no training/performance'}
    if not args.execute:
        print(json.dumps(manifest, indent=2, default=str))
        return 0
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['allocation']['cuda_visible_devices']:
        raise ValueError('CUDA_VISIBLE_DEVICES differs from root allocation')
    for port in manifest['explicit_ports']:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (root / name).write_text(json.dumps(value, indent=2, default=str) + '\n')
    write('supervisor-manifest.json', manifest)
    write('config.json', config)
    service = dataclasses.replace(service, log_path=str(root / 'mooncake.log'))
    started = []
    child = None
    result = {'passed': False, 'owned_pids': []}
    try:
        process = _spawn_service(service, popen=subprocess.Popen)
        started.append((service, process))
        result['owned_pids'].append({'role': 'mooncake', 'pid': process.pid})
        _wait_for_service(service, process, started)
        with (root / 'driver.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'actual_faults.driver',
                '--config', str(root / 'config.json'), '--execute',
                '--run-dir', str(root / 'engine')], env=dict(os.environ),
                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        result['owned_pids'].append({'role': 'fault-driver', 'pid': child.pid})
        write('owned-processes.json', result['owned_pids'])
        deadline = time.monotonic() + float(config.get('suite_timeout_s', 1800)) + 600
        while child.poll() is None:
            if process.poll() is not None:
                raise RuntimeError('owned Mooncake service exited during the fault suite')
            if time.monotonic() >= deadline:
                raise TimeoutError('owned fault driver exceeded bounded startup/suite deadline')
            time.sleep(.2)
        result['driver_exit'] = child.returncode
        result['passed'] = child.returncode == 0 and json.loads((root/'engine/result.json').read_text())['passed']
    except BaseException as exc:
        result['error'] = repr(exc)
        (root/'failure.txt').write_text(traceback.format_exc())
    finally:
        result['cleanup_errors'] = finish_owned_processes(child, started, _terminate_processes, _stop_services)
        try:
            result['source_unchanged'] = source == source_manifest(config)
            result['supervisor_unchanged'] = manifest['supervisor_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        except BaseException as exc:
            result['source_unchanged'] = False
            result['source_check_error'] = repr(exc)
        result['passed'] = bool(result['passed'] and result['source_unchanged']
                                and result.get('supervisor_unchanged') and not result['cleanup_errors'])
        result['owned_process_exits'] = [{'pid': p.pid, 'exit': p.poll()} for _, p in started]
        if child is not None:
            result['owned_process_exits'].append({'pid': child.pid, 'exit': child.poll()})
        result['passed'] = result['passed'] and all(row['exit'] is not None for row in result['owned_process_exits'])
        write('supervisor-result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
