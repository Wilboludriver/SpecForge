#!/usr/bin/env python3
"""Plan by default. --execute is only for a later explicit root resource assignment.

Every parent harness command uses the fixed venv PATH. Actual training, full
sample validation and measurement remain in the unchanged integrated harness.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import time

PROJECT=next(parent for parent in Path(__file__).resolve().parents
             if (parent / "NEXT_SESSION_PROMPT.txt").is_file())
FORBIDDEN=('MC_FORCE_TCP','MC_MS_AUTO_DISC','MC_USE_TENT','MC_USE_TEV1')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def parent_env(matrix,source):
    env=dict(os.environ)
    env.update(PATH=matrix['fixed_bin']+os.pathsep+env.get('PATH',''),CUDA_VISIBLE_DEVICES='',
        TMPDIR='/workdir/spectrain/temp',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=os.pathsep.join((source['specforge'],str(Path(source['sglang'])/'python'))))
    master=shutil.which('mooncake_master',path=env['PATH'])
    expected=Path(matrix['fixed_bin'])/'mooncake_master'
    if master is None or Path(master).absolute()!=expected.absolute():
        raise RuntimeError(f'fixed-venv mooncake_master required, resolved {master}')
    return env,{'PATH':env['PATH'],'mooncake_master':master,'python':matrix['python'],
                'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],'PYTHONPATH':env['PYTHONPATH']}


def verify_inputs(matrix):
    checks={matrix['harness']:matrix['harness_sha256'],str(Path(matrix['harness']).with_name('performance_helpers.py')):matrix['helpers_sha256'],
        matrix['native_cases']:matrix['native_sha256'],matrix['real_manifest']:matrix['real_manifest_sha256']}
    for model in matrix['models']:
        checks[model['config']]=model['config_sha256'];checks[model['real_cases']]=model['real_sha256']
        checks.update({row['path']:row['sha256'] for row in model['model_references'].values()})
    for path,expected in checks.items():
        if file_sha(path)!=expected:raise RuntimeError(f'frozen input changed: {path}')
    env,provenance=parent_env(matrix,matrix['sources']['candidate'])
    identities={}
    for name,source in matrix['sources'].items():
        identities[name]={}
        for kind,commit_key in (('specforge','sf_commit'),('sglang','sg_commit')):
            path=source[kind]
            def git(*args):return subprocess.check_output(['git','-C',path,*args],env=env,text=True).strip()
            commit=git('rev-parse','HEAD');dirty=git('status','--porcelain')
            if commit!=source[commit_key] or dirty:raise RuntimeError(f'frozen {name} {kind} identity changed/dirty')
            identities[name][kind]={'root':git('rev-parse','--show-toplevel'),'commit':commit,'branch':git('branch','--show-current'),'dirty':False}
    return {'identities':identities,'parent_environment':provenance,'input_sha256':checks,
            'harness_sha256':matrix['harness_sha256'],'helpers_sha256':matrix['helpers_sha256']}


def verify_runner_contract(contract, matrix_path):
    for key,path in (('runner_sha256',__file__),('matrix_sha256',matrix_path)):
        if file_sha(path)!=contract[key]:
            raise RuntimeError('frozen matrix/runner changed: '+str(path))
    return {'runner_sha256':contract['runner_sha256'],'matrix_sha256':contract['matrix_sha256']}


def cases(matrix,models,workloads,hot_repetitions=1,include_fallback=False):
    rows=[]
    for model in matrix['models']:
        if models is not None and model['id'] not in models:continue
        for workload in model['workloads']:
            if workloads is not None and workload not in workloads:continue
            for repetition in range(1,(hot_repetitions if workload=='hot' else 1)+1):
                for group in matrix['groups']:
                    for transport in (('rdma','tcp') if include_fallback and group!='device-off' else ('rdma',)):
                        rows.append({'model':model['id'],'workload':workload,'group':group,'transport':transport,
                            'repetition':repetition,'conditional_fallback':transport=='tcp',
                            'id':f"{model['id']}/{workload}/r{repetition}/{transport}-{group}",
                            'measured_steps':matrix['real_measured_steps'] if workload=='real64' else matrix['synthetic_measured_steps']})
    return rows


def harness_command(matrix,case,run_dir,*,execute=False,rdma_proof=None):
    model=next(row for row in matrix['models'] if row['id']==case['model'])
    source=matrix['sources']['candidate' if case['group']=='host-on' else 'official']
    env,provenance=parent_env(matrix,source)
    argv=[matrix['python'],matrix['harness'],'--config',model['config'],'--native-cases',matrix['native_cases'],
        '--real-cases',model['real_cases'],'--run-dir',str(run_dir),'--group',case['group'],'--workload',case['workload'],
        '--transport',case['transport'],'--specforge-source',source['specforge'],'--sglang-source',source['sglang'],
        '--expected-specforge',source['sf_commit'],'--expected-sglang',source['sg_commit'],
        '--warmup-steps',str(matrix['warmup_steps']),'--steps',str(case['measured_steps']),
        '--cache-bytes',str(matrix['cache_bytes']),'--chunk-tokens',str(matrix['chunk_tokens']),
        '--rdma-probe-timeout-s',str(matrix['bounds']['rdma_probe_timeout_s'])]
    if 'polling_policy' in matrix:
        from performance_helpers import validate_polling_policy
        for field,value in validate_polling_policy(matrix['polling_policy']).items():
            argv.extend(['--'+field.replace('_','-'),str(value)])
    if case['transport']=='rdma':
        argv.extend(['--rdma-devices',matrix['rdma_devices']])
        if execute and case['group']!='device-off':
            if rdma_proof is None:raise RuntimeError('host RDMA requires a successful actual device proof')
            argv.extend(['--rdma-proof',str(rdma_proof)])
    if execute:argv.append('--execute')
    return argv,env,provenance


def proc_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'start_time':int(fields[19]),'pgid':os.getpgid(pid)}
    except (FileNotFoundError,ProcessLookupError):return None


def child_pids(pid):
    # fork/spawn may run on any thread, not just the process leader.
    result=set()
    try:
        tasks=list(Path(f'/proc/{pid}/task').iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return result
    for task in tasks:
        try:
            result.update(int(value) for value in (task/'children').read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def enter_owned_boundary():
    """Serial runner is the adopter for all descendants of its next child.

    Fail before launching if the boundary already has children. The only
    asynchronous child created inside a case is its harness; helper commands
    outside this boundary are synchronous and waited before entry.
    """
    if sys.platform != 'linux':
        raise RuntimeError('owned descendant adoption requires Linux subreaper')
    existing=child_pids(os.getpid())
    if existing:
        raise RuntimeError(f'owned boundary already contains children; no launch or signals: {sorted(existing)}')
    libc=ctypes.CDLL(None, use_errno=True)
    previous=ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot read previous child subreaper')
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot establish child subreaper')
    enabled=ctypes.c_int()
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) != 0 or enabled.value != 1:
        libc.prctl(36, previous.value, 0, 0, 0)
        raise RuntimeError('child subreaper could not be verified')
    return previous.value


def leave_owned_boundary(previous):
    if child_pids(os.getpid()):
        raise RuntimeError('cannot leave subreaper boundary while owned children remain')
    libc=ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, previous, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot restore previous child subreaper')


class OwnedTree:
    """Own one serial subreaper scope, including previously unseen double forks."""
    def __init__(self,process):
        self.process=process;self.identities={};self.errors=[];self.reaped=[]
        self.adopter=os.getpid();self.refresh()
    def refresh(self):
        pending=[self.process.pid];seen=set()
        # Reparented detached descendants cannot escape this dedicated adopter.
        pending.extend(child_pids(self.adopter))
        pending.extend(row['pid'] for row in self.live())
        while pending:
            pid=pending.pop()
            if pid in seen:continue
            seen.add(pid);row=proc_identity(pid)
            if row is None:continue
            previous=self.identities.get(pid)
            if previous is not None and previous['start_time']!=row['start_time']:
                self.errors.append(f'PID birth identity changed inside active boundary: {pid}')
                continue
            self.identities[pid]=row
            try:pending.extend(child_pids(pid))
            except (PermissionError, OSError) as exc:
                self.errors.append(f'descendant coverage unavailable: {pid}: {exc}')
        # Popen alone reaps its own leader. Adopted orphan zombies have no
        # executing resource, but are recorded and reaped by their new parent.
        for pid,expected in list(self.identities.items()):
            if pid==self.process.pid:continue
            row=proc_identity(pid)
            if (row is not None and row['start_time']==expected['start_time']
                    and row['state']=='Z' and row['ppid']==self.adopter):
                try:
                    reaped,status=os.waitpid(pid,os.WNOHANG)
                    if reaped:self.reaped.append({**row,'wait_status':status})
                except ChildProcessError:pass
    def live(self):
        rows=[]
        for pid,expected in self.identities.items():
            current=proc_identity(pid)
            if (current is not None and current['start_time']==expected['start_time']
                    and current.get('state')!='Z'):rows.append(current)
        return rows
    def signal_groups(self,signum):
        own_group=os.getpgrp()
        groups={row['pgid'] for row in self.live()}
        if own_group in groups or any(group<=1 for group in groups):raise RuntimeError('refusing to signal the matrix/root process group')
        for group in groups:
            try:os.killpg(group,signum)
            except ProcessLookupError:pass
    def drain(self,grace_s):
        deadline=time.monotonic()+grace_s
        kill_deadline=deadline+5
        while True:
            self.process.poll();self.refresh()
            direct=child_pids(self.adopter)
            live=self.live()
            if not direct and not live and self.process.poll() is not None:break
            now=time.monotonic()
            if now>=kill_deadline:break
            if live:self.signal_groups(signal.SIGTERM if now<deadline else signal.SIGKILL)
            time.sleep(.05)
        self.process.poll();self.refresh()
        remaining_direct=sorted(child_pids(self.adopter))
        if self.live() or remaining_direct or self.errors:
            raise RuntimeError(f'owned process tree not proven drained: {self.live()}, direct={remaining_direct}, errors={self.errors}')
        return {'identities':list(self.identities.values()),'reaped_orphans':self.reaped,
                'remaining':[],'remaining_direct_children':[], 'coverage_errors':[],
                'ownership_boundary':'verified Linux child subreaper; initially empty; one asynchronous harness'}


def emergency_drain(process,tree,grace_s,cause):
    """Inspection failure still attempts the known session and verified descendants."""
    errors=[f'ownership inspection/drain failed: {type(cause).__name__}: {cause}']
    known=dict(getattr(tree,'identities',{}));signalled=[]
    def verified():
        # The empty, exclusive subreaper boundary was established before Popen.
        # Newly adopted descendants remain owned even if tracker setup failed.
        try:
            pending=list(child_pids(os.getpid()));seen=set()
            while pending:
                pid=pending.pop()
                if pid in seen:continue
                seen.add(pid);row=proc_identity(pid)
                if row is not None:
                    old=known.get(pid)
                    if old is None or old['start_time']==row['start_time']:
                        known[pid]=row;pending.extend(child_pids(pid))
        except (OSError,ValueError) as exc:
            errors.append(f'adopted descendant inspection unavailable: {exc}')
        live=[]
        for pid,expected in known.items():
            try:
                row=proc_identity(pid)
                if row is not None and row['start_time']==expected['start_time'] and row.get('state')!='Z':live.append(row)
            except (OSError,ValueError) as exc:errors.append(f'identity unavailable {pid}: {exc}')
        return live
    def send(signum):
        groups={row['pgid'] for row in verified()}
        # Popen's unreaped running child is still the exact new session we created,
        # even when /proc cannot be read. Never use an unverified stale leader PID.
        if process.poll() is None:groups.add(process.pid)
        for group in groups:
            if group<=1 or group in (os.getpgrp(),os.getpid()):
                errors.append(f'refused root/invalid process group {group}');continue
            try:os.killpg(group,signum);signalled.append({'pgid':group,'signal':signum})
            except ProcessLookupError:pass
            except OSError as exc:errors.append(f'group signal failed {group}: {exc}')
    send(signal.SIGTERM)
    try:process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:pass
    deadline=time.monotonic()+grace_s
    while verified() and time.monotonic()<deadline:time.sleep(.1)
    if process.poll() is None or verified():send(signal.SIGKILL)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:errors.append('owned harness did not exit after emergency cleanup')
    return {'drain_proven':False,'emergency_cleanup':True,'child_pid':process.pid,'remaining_verified':verified(),
            'signalled':signalled,'coverage_errors':errors,'identities':list(known.values())}


def run_child(argv,env,log_path,*,timeout_s,grace_s,checkpoint):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    process=tree=selector=None
    previous_subreaper=enter_owned_boundary()
    try:
        # Everything after a successful spawn is within the ownership finally,
        # including tracker construction and any asynchronous signal delivery.
        process=subprocess.Popen(argv,cwd=PROJECT,env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        tree=OwnedTree(process)
        checkpoint({'child_pid':process.pid,'owned_session':True,'argv':argv,'stage':'running'})
        selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
        started=time.monotonic()
        with log_path.open('wb') as log:
            while selector.get_map():
                tree.refresh()
                if time.monotonic()-started>timeout_s:raise TimeoutError(f'bounded harness wait exceeded {timeout_s}s')
                for key,_ in selector.select(.2):
                    data=os.read(key.fileobj.fileno(),65536)
                    if data:
                        log.write(data);log.flush();sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
                    else:selector.unregister(key.fileobj)
                if process.poll() is not None and not selector.get_map():break
            return process.wait(timeout=5)
    finally:
        if process is None:
            leave_owned_boundary(previous_subreaper)
        else:
            if selector is not None:selector.close()
            process.stdout.close()
            failure=None
            try:
                if tree is None:raise RuntimeError('ownership tracker construction failed')
                report=tree.drain(grace_s)
            except BaseException as exc:
                failure=exc
                report=emergency_drain(process,tree,grace_s,exc)
            if failure is None:
                try:leave_owned_boundary(previous_subreaper)
                except BaseException as exc:
                    failure=exc;report['boundary_restore_error']=repr(exc)
            save(log_path.with_name('owned-tree-final.json'),report)
            checkpoint({'child_pid':process.pid,'stage':'drained' if failure is None else 'cleanup-failed',
                        'exit':process.poll(),'owned_tree_report':str(log_path.with_name('owned-tree-final.json'))})
            if failure is not None:
                raise RuntimeError('owned tree drain could not be proven; emergency cleanup attempted') from failure


def port_snapshot(ports):
    busy=[]
    # Match official managed_preflight's strict raw bind, including TIME_WAIT
    # and bound-but-not-listening sockets. Hold all probes until the full check.
    with ExitStack() as probes:
        for port in ports:
            probe=probes.enter_context(socket.socket(socket.AF_INET,socket.SOCK_STREAM))
            try:probe.bind(('0.0.0.0',port))
            except OSError as exc:busy.append({'port':port,'error':str(exc)})
        for source in ('/proc/net/tcp','/proc/net/tcp6'):
            for line in Path(source).read_text().splitlines()[1:]:
                fields=line.split();port=int(fields[1].split(':')[1],16)
                if fields[3]=='0A' and port in ports and not any(row['port']==port for row in busy):busy.append({'port':port,'error':'LISTEN in '+source})
    return busy


def gpu_snapshot(gpus,env):
    def query(kind,fields):
        result=subprocess.run(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],env=env,
            text=True,capture_output=True,timeout=10,check=True)
        return [[value.strip() for value in line.split(',')] for line in result.stdout.splitlines() if line.strip()]
    devices=query('gpu','index,uuid,memory.used');mapping={row[1]:row[0] for row in devices}
    selected=[{'index':row[0],'uuid':row[1],'memory_used_mib':int(row[2])} for row in devices if row[0] in gpus]
    if len(selected)!=len(gpus):raise RuntimeError('assigned GPU visibility/physical memory unavailable')
    processes=[{'gpu':mapping.get(row[0]),'pid':int(row[1])} for row in query('compute-apps','gpu_uuid,pid') if mapping.get(row[0]) in gpus]
    return {'devices':selected,'processes':processes}


def resource_gate(matrix,model,env,*,baseline=None):
    gpu=gpu_snapshot(set(model['target_gpus']+model['draft_gpus']),env);busy=port_snapshot(matrix['resources']['ports'])
    excessive=[]
    if baseline is not None:
        initial={row['index']:row['memory_used_mib'] for row in baseline['gpu']['devices']}
        excessive=[row for row in gpu['devices'] if row['memory_used_mib']>initial[row['index']]+matrix['resources']['memory_release_slack_mib']]
    return {'gpu':gpu,'busy_ports':busy,'memory_above_baseline':excessive,'time':time.time(),
        'released':not gpu['processes'] and not busy and not excessive}


def wait_released(matrix,model,env,baseline):
    deadline=time.monotonic()+matrix['bounds']['release_timeout_s']
    while True:
        audit=resource_gate(matrix,model,env,baseline=baseline)
        if audit['released']:return audit
        if time.monotonic()>=deadline:raise RuntimeError('assigned GPUs/ports not released: '+json.dumps(audit))
        time.sleep(.5)


def typed_gdr_unavailable(case,result,proof,cleanup):
    return (case['group']=='device-off' and case['transport']=='rdma' and result.get('status')=='unavailable'
        and result.get('passed') is False and result.get('source_unchanged') is True
        and result.get('error','').startswith('DevicePublicationUnavailable:') and proof.get('status')=='unavailable'
        and cleanup.get('key_removal_authorized') is True and not cleanup.get('errors'))


def validate_case(matrix,case,run_dir,execute,status):
    result=load(run_dir/'result.json');manifest=load(run_dir/'manifest.json')
    if not result.get('source_unchanged') or manifest['harness_sha256']!=matrix['harness_sha256'] or manifest['helpers_sha256']!=matrix['helpers_sha256']:
        raise RuntimeError('source/harness immutability check failed')
    if not execute:
        if status!=0 or result['status']!='planned' or result.get('passed'):raise RuntimeError('resource-free planning failed')
    elif status!=0 or result.get('status')!='passed' or result.get('passed') is not True:
        proof_path=run_dir/'device-proof.json';cleanup_path=run_dir/'probe-cleanup.json'
        if proof_path.exists() and cleanup_path.exists() and typed_gdr_unavailable(case,result,load(proof_path),load(cleanup_path)):
            return {'status':'unavailable','typed_proof':str(proof_path),'result':result,'manifest':manifest}
        raise RuntimeError('actual harness failed: '+str(result.get('error',result)))
    if execute:
        audit=result['consumed_order']
        if not audit['full_order_matches'] or not audit['measured_counter_matches']:raise RuntimeError('actual optimizer input audit failed')
        exits=load(run_dir/'owned-processes-final.json')
        if any(row['exit'] is None for row in exits):raise RuntimeError('harness retained an owned child')
        cleanup=load(run_dir/'cleanup-final.json')
        if cleanup.get('remaining_keys') or not cleanup.get('inspected_keys'):raise RuntimeError('full training key cleanup not proven')
        if case['group']=='device-off':
            sys.path.insert(0,str(Path(matrix['harness']).parent))
            from performance_helpers import validate_rdma_proof
            proof=load(run_dir/'device-proof.json');validate_rdma_proof(proof,expected_devices=matrix['rdma_devices'])
            if proof.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise RuntimeError('device proof boot mismatch')
            if result.get('device_publication_available') is not True:raise RuntimeError('actual CUDA publication proof missing')
            stages=result['server_stage_attribution'].get('publication_register_put_unregister_wall_ms',{})
            # The actual harness separately validates CUDA sources for every measured model sink record.
            if not stages.get('events'):raise RuntimeError('actual model CUDA publication sink observations missing')
    return {'status':'passed' if execute else 'planned','result':result,'manifest':manifest}


def self_test(matrix):
    import tempfile
    from unittest import mock
    rows=cases(matrix,None,None,include_fallback=True)
    workload_count=sum(len(model['workloads']) for model in matrix['models'])
    assert len(rows)==workload_count*5 and sum(row['transport']=='rdma' for row in rows)==workload_count*3
    assert len(cases(matrix,{'qwen3-4b'},None))==12
    assert len(cases(matrix,{'qwen3-4b'},{'hot'},3))==9
    assert all(row['group']!='device-off' for row in rows if row['transport']=='tcp')
    env,provenance=parent_env(matrix,matrix['sources']['official'])
    assert env['PATH'].split(os.pathsep)[0]==matrix['fixed_bin']
    assert provenance['mooncake_master']==str(Path(matrix['fixed_bin'])/'mooncake_master')
    case=rows[0];argv,_,_=harness_command(matrix,case,Path('/tmp/not-executed'))
    assert '--execute' not in argv and '--steps' in argv and argv[argv.index('--steps')+1]=='32'
    result={'status':'unavailable','passed':False,'source_unchanged':True,'error':'DevicePublicationUnavailable: native CUDA registration failed'}
    assert typed_gdr_unavailable(case,result,{'status':'unavailable'},{'key_removal_authorized':True,'errors':[]})
    for change in ({'status':'failed'},{'error':'RuntimeError: model startup failed'},{'source_unchanged':False},{'passed':True}):
        assert not typed_gdr_unavailable(case,{**result,**change},{'status':'unavailable'},{'key_removal_authorized':True,'errors':[]})
    assert not typed_gdr_unavailable({**case,'transport':'tcp'},result,{'status':'unavailable'},{'key_removal_authorized':True,'errors':[]})
    assert not typed_gdr_unavailable(case,result,{'status':'unavailable'},{'key_removal_authorized':False,'errors':[]})
    # A foreign/root process group is never signalled, even under injected ownership corruption.
    tree=OwnedTree.__new__(OwnedTree);tree.live=lambda:[{'pgid':os.getpgrp()}]
    with mock.patch.object(os,'killpg') as kill:
        try:tree.signal_groups(signal.SIGTERM)
        except RuntimeError:pass
        else:raise AssertionError('root process group accepted')
        kill.assert_not_called()
    with tempfile.TemporaryDirectory(dir=PROJECT/'implementation/configs/performance') as tmp:
        save(Path(tmp)/'checkpoint.json',{'stage':'planned'})
        assert load(Path(tmp)/'checkpoint.json')['stage']=='planned'
        # A checkpoint failure immediately after Popen must still drain the owned child.
        child=type('FakeChild',(),{'pid':700001,'stdout':type('Pipe',(),{'close':lambda self:None})(),
            'poll':lambda self:0})()
        owner=mock.Mock();owner.drain.return_value={'remaining':[]}
        with mock.patch.object(subprocess,'Popen',return_value=child) as spawn,mock.patch(__name__+'.OwnedTree',return_value=owner):
            try:run_child(['fake'],{},Path(tmp)/'harness.log',timeout_s=1,grace_s=1,
                checkpoint=mock.Mock(side_effect=RuntimeError('checkpoint failure')))
            except RuntimeError:pass
            else:raise AssertionError('checkpoint failure ignored')
            owner.drain.assert_called_once_with(1)
            assert spawn.call_args.kwargs['start_new_session'] is True
        # /proc/drain failure must attempt the known session and any still-verifiable nested group.
        alive={'value':True};calls=[]
        child=type('FakeChild',(),{'pid':700001,'poll':lambda self:None if alive['value'] else 0,
            'wait':lambda self,timeout:alive.update(value=False) or 0})()
        owner=type('Owner',(),{'identities':{700002:{'pid':700002,'start_time':123,'pgid':700002}}})()
        def identity(pid):return {'pid':pid,'start_time':123,'pgid':pid} if alive['value'] else None
        with mock.patch(__name__+'.proc_identity',side_effect=identity),mock.patch.object(os,'killpg',side_effect=lambda group,sig:calls.append((group,sig))):
            report=emergency_drain(child,owner,0,PermissionError('injected /proc failure'))
        assert {group for group,_ in calls}=={700001,700002}
        assert report['drain_proven'] is False and report['emergency_cleanup'] and not report['remaining_verified']
        # Busy foreign GPU/port observations stop the next case; no signal/cleanup is attempted.
        model=matrix['models'][0]
        with mock.patch(__name__+'.gpu_snapshot',return_value={'devices':[],'processes':[{'pid':999,'gpu':'0'}]}),mock.patch(__name__+'.port_snapshot',return_value=[]):
            assert resource_gate(matrix,model,env)['released'] is False
        fake_dir=Path(tmp)/'failed';fake_dir.mkdir()
        save(fake_dir/'manifest.json',{'harness_sha256':matrix['harness_sha256'],'helpers_sha256':matrix['helpers_sha256']})
        save(fake_dir/'result.json',{'source_unchanged':True,'status':'failed','passed':False,'error':'model startup'})
        try:validate_case(matrix,case,fake_dir,True,2)
        except RuntimeError:pass
        else:raise AssertionError('unrelated model failure treated as unavailable')
    print('CPU self-test passed: matrix cardinality/bounds, fixed PATH, typed fallback negative controls, root-group guard, atomic checkpoints; no children/services/GPU queries.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',default=str(PROJECT/'implementation/configs/performance/matrix.json'))
    parser.add_argument('--run-root');parser.add_argument('--models',default='qwen3-4b',help='comma-separated model ids, or all')
    parser.add_argument('--workloads',default='all');parser.add_argument('--hot-repetitions',type=int,default=1)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--expected-specforge',help='explicit later root-frozen candidate SHA; default is matrix snapshot')
    parser.add_argument('--expected-sglang',help='explicit later root-frozen candidate SHA; harness byte hashes still fixed')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv);matrix=load(args.matrix)
    if args.expected_specforge:matrix['sources']['candidate']['sf_commit']=args.expected_specforge
    if args.expected_sglang:matrix['sources']['candidate']['sg_commit']=args.expected_sglang
    if args.self_test:self_test(matrix);return 0
    if args.run_root is None:parser.error('--run-root is required')
    if args.hot_repetitions<1:parser.error('positive repetition count required')
    models=None if args.models=='all' else set(args.models.split(','));workloads=None if args.workloads=='all' else set(args.workloads.split(','))
    if models is not None and not models<={row['id'] for row in matrix['models']}:parser.error('unknown model')
    if workloads is not None and not workloads<={'cold','hot','over-capacity','real64'}:parser.error('unknown workload')
    selection=cases(matrix,models,workloads,args.hot_repetitions,include_fallback=not args.execute)
    if not selection:parser.error('empty matrix selection')
    run_root=Path(args.run_root).resolve();checkpoint_path=run_root/'matrix-checkpoint.json'
    contract={'matrix_sha256':file_sha(args.matrix),'runner_sha256':file_sha(__file__),'models':args.models,
        'workloads':args.workloads,'hot_repetitions':args.hot_repetitions,'execute':args.execute,
        'expected_candidate_sources':matrix['sources']['candidate'],
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip() if args.execute else None}
    if args.resume:
        state=load(checkpoint_path)
        if state['contract']!=contract:raise RuntimeError('resume contract differs')
        if any(row.get('status') not in ('passed','planned','unavailable') for row in state['cases'].values()):raise RuntimeError('nonterminal/failing checkpoint requires resource investigation; not silently resumed')
    else:
        run_root.mkdir(parents=True,exist_ok=False);state={'contract':contract,'cases':{},'status':'planning','acceleration_claim_evaluated':False}
    save(checkpoint_path,state)
    try:
        verify_runner_contract(contract,args.matrix)
        state['input_verification']=verify_inputs(matrix)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');save(checkpoint_path,state);return 2
    save(checkpoint_path,state)
    lock=None;active_case=None
    previous_handlers={}
    def interrupted(signum,_frame):
        for current in previous_handlers:signal.signal(current,signal.SIG_IGN)
        raise KeyboardInterrupt(f'matrix signal {signum}')
    if args.execute:
        if any(name in os.environ for name in FORBIDDEN):raise RuntimeError('forbidden inherited transport override')
        lock=(Path(args.matrix).parent/'assigned-resource.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):previous_handlers[signum]=signal.signal(signum,interrupted)
    try:
        gdr={};contracts={};observed={};initial={}
        for case in selection:
            verify_runner_contract(contract,args.matrix)
            active_case=case;identifier=case['id'];run_dir=run_root/'runs'/identifier
            model=next(row for row in matrix['models'] if row['id']==case['model'])
            if args.execute and case['group']=='device-off' and gdr.get(case['model'],{}).get('status')=='unavailable':
                state['cases'][identifier]={'status':'unavailable','inherited_typed_proof':gdr[case['model']],
                    'reason':'same frozen writer GPU/HCA/boot already proved native GDR unavailable; no TCP device relabel'}
                save(checkpoint_path,state);continue
            if args.execute and case['group']!='device-off':
                available=gdr.get(case['model'])
                if available is None:raise RuntimeError('host comparison lacks preceding actual device availability proof')
                if available['status']=='unavailable':
                    case={**case,'transport':'tcp','conditional_fallback':True,'id':identifier.replace('rdma-','tcp-')}
                    identifier=case['id'];run_dir=run_root/'runs'/identifier;active_case=case
            existing=state['cases'].get(identifier)
            if existing is not None:
                if existing.get('result_sha256') and file_sha(run_dir/'result.json')!=existing['result_sha256']:raise RuntimeError('resumed result changed')
                if existing['status']=='unavailable':gdr[case['model']]=existing['gdr'];continue
                checked=validate_case(matrix,case,run_dir,args.execute,0)
            else:
                verify_inputs(matrix)
                proof=gdr.get(case['model'],{}).get('proof')
                command,env,provenance=harness_command(matrix,case,run_dir,execute=args.execute,rdma_proof=proof)
                baseline=None
                if args.execute:
                    baseline=resource_gate(matrix,model,env)
                    if not baseline['released']:raise RuntimeError('assigned resources busy before case: '+json.dumps(baseline))
                record={'status':'running','case':case,'argv':command,'parent_environment':provenance,'baseline':baseline,'started_at':time.time(),
                    'fallback_native_unavailable':gdr.get(case['model']) if case['transport']=='tcp' else None}
                state['cases'][identifier]=record;state['status']='executing' if args.execute else 'planning';save(checkpoint_path,state)
                def update(values):record.update(values);save(checkpoint_path,state)
                print(f"[{identifier}] {'EXECUTE' if args.execute else 'RESOURCE-FREE PLAN'}",flush=True)
                try:
                    status=run_child(command,env,run_root/'logs'/identifier/'harness.log',
                        timeout_s=matrix['bounds']['execute_timeout_s' if args.execute else 'plan_timeout_s'],
                        grace_s=matrix['bounds']['shutdown_grace_s'],checkpoint=update)
                finally:
                    if args.execute:
                        record['resource_release']=wait_released(matrix,model,env,baseline);save(checkpoint_path,state)
                checked=validate_case(matrix,case,run_dir,args.execute,status)
                record.update(status=checked['status'],result_sha256=file_sha(run_dir/'result.json'),ended_at=time.time())
                if checked['status']=='unavailable':
                    unavailable={'status':'unavailable','proof':checked['typed_proof'],'run_dir':str(run_dir),
                        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'hca':matrix['rdma_devices'],
                        'writer_gpu':model['target_gpus'][0],'result_sha256':record['result_sha256']}
                    record['gdr']=unavailable;gdr[case['model']]=unavailable;save(checkpoint_path,state);continue
                save(checkpoint_path,state)
            manifest=checked['manifest'];key=(case['model'],case['workload'],case['repetition'],case['transport'])
            value=digest(manifest['comparison_contract'])
            if key in contracts and contracts[key]!=value:raise RuntimeError('matched group planned contracts differ')
            contracts[key]=value
            if args.execute:
                result=checked['result'];sequence=result['consumed_order']['ordered_sha256'];draft=result['initial_draft']
                if len(draft)!=1:raise RuntimeError('one initial draft required')
                if key in observed and observed[key]!=sequence:raise RuntimeError('actual optimizer order differs between matched groups')
                if case['model'] in initial and initial[case['model']]!=draft[0]['sha256']:raise RuntimeError('initial pretrained draft differs')
                observed[key]=sequence;initial[case['model']]=draft[0]['sha256']
                if case['group']=='device-off':gdr[case['model']]={'status':'available','proof':str(run_dir/'device-proof.json')}
            state['matched_contracts']={'/'.join(map(str,key)):value for key,value in contracts.items()};save(checkpoint_path,state)
        state['runner_contract_after']=verify_runner_contract(contract,args.matrix)
        state['source_after']=verify_inputs(matrix);state['status']='complete' if args.execute else 'planned'
        state['ended_at']=time.time();save(checkpoint_path,state)
        return 0
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',failed_case=active_case,ended_at=time.time())
        save(checkpoint_path,state);print(state['error'],file=sys.stderr,flush=True);return 2
    finally:
        for signum,handler in previous_handlers.items():signal.signal(signum,handler)
        if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':raise SystemExit(main())
