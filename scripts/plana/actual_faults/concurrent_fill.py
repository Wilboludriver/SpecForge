"""Test-only natural same-content overlap; one real capture worker, no extra fill."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
import weakref
from actual_faults import atomic_json, read_json, wait_for

KIND = 'same_content_fill_overlap'
_LOCAL = threading.local()


def token_digest(tokens):
    return hashlib.sha256(json.dumps(list(tokens),separators=(',',':')).encode()).hexdigest()


def key_wire(key):
    return {name:getattr(key,name) for name in ('namespace','digest','parent_digest','start','end')}


def execution_wire(execution):
    return {name:getattr(execution,name) for name in ('server_epoch','request_id','attempt_id','execution_epoch')}


def validate_config(config):
    if config.get('kinds') != [KIND]:raise ValueError('same-content fill requires its own fresh dedicated run')
    args=config['engine_kwargs']
    if config['model']['name']!='qwen36-a3b' or args['tp_size']!=2:
        raise ValueError('this bounded fixture requires the selected actual A3B TP2 model')
    if args.get('spec_feature_chunk_tokens')!=256 or args.get('spec_feature_cache_bytes')!=64<<20:
        raise ValueError('same-content fixture requires explicit C256 and64MiB cache')
    if args.get('max_running_requests')!=1:raise ValueError('single-request engine bound must remain one')
    gate=float(config.get('fill_gate_timeout_s',45))
    if not 5<=gate<=45:raise ValueError('fill gate must be bounded between5 and45 seconds')


def pair_arm(directory):
    arm=read_json(Path(directory)/'arm.json',{})
    if arm.get('kind')!=KIND:return None
    actors=arm.get('actors',{})
    if set(actors)!={'A','B'} or any(set(actor)!={'rid','sample_id'} for actor in actors.values()):
        raise AssertionError('pair arm must bind exact A/B rid and sample_id')
    if any(len({actor[name] for actor in actors.values()})!=2 for name in ('rid','sample_id')):
        raise AssertionError('pair actors require distinct requests and sample keys')
    if arm.get('N')!=1280 or not arm.get('token_sha256'):raise AssertionError('pair content/range identity missing')
    return arm


class _Observer:
    def __init__(self,state):
        self.state=state;self.bridge=state.bridge;self.scheduler=state.scheduler
        self.requests={};self.executions={};self.plans={};self.gate=threading.Event()
        self.target_fill=None;self.partial=False;self.b_copy_started=False
        self.auto_release=False;self.gate_used=False

    def actor(self,rid,sample_id=None):
        arm=pair_arm(self.state.directory)
        if arm is None:return None,None
        matches=[name for name,value in arm['actors'].items() if value['rid']==rid and (sample_id is None or value['sample_id']==sample_id)]
        if len(matches)>1:raise AssertionError('ambiguous pair actor')
        return (matches[0] if matches else None),arm

    def emit(self,event,actor,arm,**values):
        self.state.emit(event,case_id=arm['case_id'],actor=actor,
            execution=self.executions.get(actor),**values)

    def observation(self,key):
        cache=self.bridge.cache
        with cache._lock:
            entry=cache._entries.get(key)
            fills=[record for record in cache._fills.values() if record.entry is entry] if entry is not None else []
            return {'state':entry.state if entry is not None else 'MISSING',
                'writer_active':any(record.writer_active for record in fills),
                'fill_ids':[record.handle.reservation_id for record in fills],
                'features':sorted(entry.features) if entry is not None else [],
                'cache_bytes':cache._bytes,'capacity_bytes':cache.capacity_bytes,
                'chunk_bytes':cache._chunk_bytes}

    def failure(self,arm,exc):
        atomic_json(self.state.directory/'concurrent-fill-error.json',
                    {'case_id':arm['case_id'],'error':f'{type(exc).__name__}: {exc}','overlap_covered':False})

    def admit(self,original,req,adder,*args,**kwargs):
        actor,arm=self.actor(req.rid,(req.spec_capture or {}).get('sample_id'))
        if actor is None:return original(req,adder,*args,**kwargs)
        self.requests[actor]=weakref.ref(req)
        entry=req.spec_feature_attempt
        self.executions[actor]={'server_epoch':self.bridge.epoch,'request_id':req.rid,
            'attempt_id':entry.attempt_id,'execution_epoch':req.spec_feature_execution_epoch}
        previous=getattr(_LOCAL,'admit',None);_LOCAL.admit=(self,actor,arm)
        try:
            result=original(req,adder,*args,**kwargs)
            plan=dict(getattr(req,'spec_feature_scalar_plan',{}) or {})
            self.emit('cf_admitted',actor,arm,member=any(item is req for item in adder.can_run_list),plan=plan,
                token_sha256=token_digest(req.origin_input_ids),sample_id=req.spec_capture['sample_id'])
            return result
        finally:_LOCAL.admit=previous

    def forward(self,original,batch,*args,**kwargs):
        selected=[]
        for req in batch.reqs:
            actor,arm=self.actor(req.rid,(req.spec_capture or {}).get('sample_id'))
            if actor is not None:selected.append((req,actor,arm))
        for req,actor,arm in selected:
            running=getattr(self.scheduler,'running_batch',None)
            candidates=list(batch.reqs)+list(getattr(running,'reqs',[]) or [])
            effective={id(item):item for item in candidates if not item.finished()}
            a_ref=self.requests.get('A');a=a_ref() if a_ref else None
            self.emit('cf_forward',actor,arm,batch_size=len(batch.reqs),effective_running=len(effective),
                a_finished=bool(a is not None and a.finished()),a_effective=bool(a is not None and id(a) in effective))
            if len(batch.reqs)!=1 or len(effective)>1:raise AssertionError('fixture changed actual max-running1 behavior')
            if actor=='B' and (a is None or not a.finished() or id(a) in effective):
                raise AssertionError('A has not released its effective engine running slot before B forward')
        result=original(batch,*args,**kwargs)
        for _,actor,arm in selected:self.emit('cf_forward_return',actor,arm)
        return result

    def lookup(self,original,namespace,tokens,limit):
        context=getattr(_LOCAL,'admit',None)
        if context is None or context[0] is not self:return original(namespace,tokens,limit)
        _,actor,arm=context
        key=self.bridge.cache.block_keys(namespace,tokens)[0]
        before=self.observation(key)
        lease=original(namespace,tokens,limit)
        if actor=='B':
            self.emit('cf_lookup_writing',actor,arm,key=key_wire(key),H=lease.coverage_end,limit=limit,
                gate_held=self.partial and not self.gate.is_set(),before=before)
        return lease

    def reserve(self,original,key,execution):
        actor,arm=self.actor(execution.request_id)
        if actor is None:return original(key,execution)
        before=self.observation(key);fill=original(key,execution)
        if actor=='A' and key.start==0 and fill is not None:self.target_fill=fill
        self.emit('cf_reserve',actor,arm,key=key_wire(key),before=before,
            fill_id=fill.reservation_id if fill is not None else None,
            reason=before['state'] if fill is None else 'RESERVED',owner=execution_wire(execution))
        return fill

    def publish(self,original,fill,features):
        actor,arm=self.actor(fill.execution.request_id)
        if actor is None:return original(fill,features)
        previous=getattr(_LOCAL,'fill',None)
        context={'observer':self,'actor':actor,'arm':arm,'fill':fill,'features':features,'copied':[],'inside':False}
        _LOCAL.fill=context
        try:
            result=original(fill,features)
            self.emit('cf_publish_ready',actor,arm,key=key_wire(fill.key),fill_id=fill.reservation_id,
                published=result,after=self.observation(fill.key),copied=list(context['copied']),
                worker=threading.get_ident(),thread=threading.current_thread().name)
            return result
        finally:_LOCAL.fill=previous

    def partial_copy(self,context,name):
        actor,arm,fill=context['actor'],context['arm'],context['fill']
        if actor!='A' or fill.key.start!=0 or name!='aux' or self.gate_used:return
        self.gate_used=True
        if context['copied']!=['aux']:raise AssertionError('partial cache gate did not follow only the real aux CPU copy')
        obs=self.observation(fill.key)
        if obs['state']!='WRITING' or not obs['writer_active'] or obs['features']:
            raise AssertionError('cache partially published before the second feature copy')
        self.partial=True;self.state.gates[arm['case_id']]=self.gate
        self.emit('cf_aux_copy_done',actor,arm,key=key_wire(fill.key),fill_id=fill.reservation_id,
            copied=['aux'],before=obs,worker=threading.get_ident(),thread=threading.current_thread().name)
        self.emit('cf_partial_gate',actor,arm,key=key_wire(fill.key),fill_id=fill.reservation_id)
        atomic_json(self.state.directory/f'fill-partial-{arm["case_id"]}.json',{'entered':True,'key':key_wire(fill.key)})
        if not self.gate.wait(float(arm['gate_timeout_s'])):
            exc=TimeoutError('B did not naturally snapshot before the real partial cache-fill gate expired')
            self.failure(arm,exc);self.emit('cf_gate_timeout',actor,arm);raise exc
        if not self.auto_release:
            raise AssertionError('partial gate released without a verified real B submit; overlap not covered')

    def copy(self,original,job):
        actor,arm=self.actor(job.plan.execution.request_id)
        if actor is not None:
            if actor=='B':self.b_copy_started=True
            self.emit('cf_copy_start',actor,arm,worker=threading.get_ident(),thread=threading.current_thread().name)
        return original(job)

    def submit(self,original,plan,span):
        actor,arm=self.actor(plan.execution.request_id)
        if actor is None:return original(plan,span)
        good=False
        try:
            self.executions[actor]=execution_wire(plan.execution)
            self.plans[actor]={'namespace':plan.namespace,'token_sha256':token_digest(plan.tokens),'N':plan.N,'L':plan.L,
                'schema_signature':plan.schema_signature,'key':key_wire(self.bridge.cache.block_keys(plan.namespace,plan.tokens)[0])}
            future=original(plan,span)
            info=self.plans[actor];cuda=all(value.is_cuda for value in span.features.values()) and set(span.features)=={'aux','last_hidden'}
            event=span.ready_event
            if not cuda or event is None or (span.start,span.end)!=(0,1280) or info['token_sha256']!=arm['token_sha256']:
                raise AssertionError('fixture lacks the real full-range CUDA snapshot of its own pair content')
            controller=self.bridge.controller;workers=controller._executor._max_workers
            if workers!=1:raise AssertionError('fixture requires unchanged single copy worker')
            stats=controller.stats()
            values={'key':info['key'],'token_sha256':info['token_sha256'],'namespace':plan.namespace,
                'schema_signature':plan.schema_signature,'N':plan.N,'L':plan.L,'span':[span.start,span.end],
                'features_cuda':cuda,'producer_event':True,'workers':workers,'future_pending':not future.done(),
                'controller':stats,'reservation':{'device_bytes':plan.reservation.device_bytes,
                    'pinned_bytes':plan.reservation.pinned_bytes,'host_bytes':plan.reservation.host_bytes}}
            if actor=='A':self.emit('cf_a_submitted',actor,arm,**values)
            else:
                if self.target_fill is None:raise AssertionError('B arrived without an actual A fill reservation')
                obs=self.observation(self.target_fill.key)
                if (not self.partial or self.gate.is_set() or self.b_copy_started or obs['state']!='WRITING'
                        or not obs['writer_active'] or obs['features'] or self.plans.get('A')!=info):
                    raise AssertionError('second snapshot did not naturally overlap A partial CPU cache fill')
                if stats['pending_copies']<2 or stats['output_reservations']<2 or stats['output_bytes']<=plan.reservation.host_bytes:
                    raise AssertionError('A/B real queued ownership and output reservations not observed')
                self.emit('cf_b_submitted',actor,arm,gate_held=True,b_copy_started=False,before=obs,**values)
                wait_for(event.query,min(10.,float(arm['gate_timeout_s'])/2),'real B CUDA producer completion while A fill held')
                self.emit('cf_b_producer_ready',actor,arm,producer_ready=True)
                good=True
                atomic_json(self.state.directory/f'fill-b-submitted-{arm["case_id"]}.json',{'queued':True,'producer_ready':True})
            return future
        except BaseException as exc:
            self.failure(arm,exc);raise
        finally:
            if actor=='B':
                self.auto_release=good
                try:
                    self.emit('cf_gate_release',actor,arm,reason='verified_b_submit' if good else 'b_submit_error')
                finally:
                    self.gate.set()  # Receipt I/O cannot strand the actual worker gate.


def guarded_cpu_copy(original,target,source,*args,**kwargs):
    context=getattr(_LOCAL,'fill',None)
    if context is None or context['inside'] or not threading.current_thread().name.startswith('spec-feature-capture'):
        return original(target,source,*args,**kwargs)
    if source.is_cuda or target.is_cuda or target.is_pinned():return original(target,source,*args,**kwargs)
    names=[name for name,value in context['features'].items()
           if source.data_ptr()==value.data_ptr() and tuple(source.shape)==tuple(value.shape) and source.dtype==value.dtype]
    if len(names)!=1:return original(target,source,*args,**kwargs)
    context['inside']=True
    try:result=original(target,source,*args,**kwargs)
    finally:context['inside']=False
    name=names[0];context['copied'].append(name)
    context['observer'].partial_copy(context,name)
    return result


def install(state):
    """Wrap existing methods; never replace computation, cache outcomes or executor."""
    import torch
    observer=_Observer(state);state.concurrent_fill_observer=observer;restorations=[]
    def patch(owner,name,method):
        original=getattr(owner,name);restorations.append((owner,name,original))
        setattr(owner,name,lambda *a,_call=original,_method=method,**kw:_method(_call,*a,**kw))
    patch(state.bridge,'admit',observer.admit)
    patch(state.scheduler,'run_batch',observer.forward)
    if state.bridge.controller is not None:
        patch(state.bridge.controller,'submit',observer.submit)
        patch(state.bridge.controller,'_copy_suffix',observer.copy)
        patch(state.bridge.cache,'lookup_and_pin',observer.lookup)
        patch(state.bridge.cache,'reserve_fill',observer.reserve)
        patch(state.bridge.cache,'publish_ready',observer.publish)
        patch(torch.Tensor,'copy_',guarded_cpu_copy)
    return restorations


def assert_receipts(rows,arm,tp_size=2):
    """Pure scalar oracle; simultaneous HTTP and global skip counters cannot pass."""
    selected=[row for row in rows if row.get('case_id')==arm['case_id']]
    def one(event,actor,rank=0):
        values=[row for row in selected if row['event']==event and row.get('actor')==actor and row['rank']==rank]
        if len(values)!=1:raise AssertionError(f'exactly one {event}/{actor}/rank{rank} required')
        return values[0]
    def require(value,message):
        if not value:raise AssertionError(message)
    plans={};identities={}
    for rank in range(tp_size):
        for actor in ('A','B'):
            admitted=one('cf_admitted',actor,rank);forward=one('cf_forward',actor,rank)
            returned=one('cf_forward_return',actor,rank)
            execution=admitted['execution'];plan=admitted['plan']
            if actor in identities:require(identities[actor]==execution,'TP execution identities differ')
            identities[actor]=execution
            require(admitted['member'] and plan['H']==plan['L']==0 and plan['N']==1280 and plan['span']==[0,1280], 'pair did not genuinely admit full prefill')
            require(execution['request_id']==arm['actors'][actor]['rid'] and admitted['sample_id']==arm['actors'][actor]['sample_id'], 'foreign actor receipt')
            require(admitted['token_sha256']==arm['token_sha256'], 'pair token content changed')
            require(forward['batch_size']==forward['effective_running']==1,'max-running1 not preserved')
            require(admitted['seq']<forward['seq']<returned['seq'],'actual admission/forward receipts out of order')
            if actor=='B':require(forward['a_finished'] and not forward['a_effective'],'A engine slot was not released before B forward')
            identity={key:plan[key] for key in ('N','L','H','span','execution_epoch','plan_digest')}
            if actor in plans:require(plans[actor]==identity,'TP scalar plans differ')
            plans[actor]=identity
    for row in selected:
        actor=row.get('actor')
        if actor in identities:require(row.get('execution')==identities[actor],'receipt has foreign execution/attempt')
    a=one('cf_a_submitted','A');b=one('cf_b_submitted','B');aux=one('cf_aux_copy_done','A');gate=one('cf_partial_gate','A')
    lookup=one('cf_lookup_writing','B');ready=one('cf_b_producer_ready','B');release=one('cf_gate_release','B')
    acopy=one('cf_copy_start','A');bcopy=one('cf_copy_start','B')
    for submitted in (a,b):
        require(submitted['features_cuda'] and submitted['producer_event'] and submitted['workers']==1 and submitted['future_pending'], 'real CUDA ownership/one worker proof missing')
        require(submitted['N']==1280 and submitted['L']==0 and submitted['span']==[0,1280] and submitted['token_sha256']==arm['token_sha256'],'wrong snapshot content/span')
        require(all(submitted['reservation'][name]>0 for name in ('device_bytes','pinned_bytes','host_bytes')), 'missing charged snapshot/output')
    require(a['key']==b['key']==aux['key']==lookup['key'] and a['schema_signature']==b['schema_signature'],'different content key/schema')
    require(a['execution']['attempt_id']!=b['execution']['attempt_id'] and a['execution']['request_id']!=b['execution']['request_id'],'pair attempt identities not independent')
    require(a['execution']['server_epoch']==b['execution']['server_epoch'],'pair server epochs differ')
    for obs in (aux['before'],lookup['before'],b['before']):
        require(obs['state']=='WRITING' and obs['writer_active'] and not obs['features'],'partial feature became visible')
        require(obs['fill_ids']==[aux['fill_id']],'WRITING reservation changed')
        require(obs['capacity_bytes']>=1280*36864 and obs['cache_bytes']<=obs['capacity_bytes'],'capacity pressure confounded dedup')
    require(aux['copied']==['aux'] and lookup['H']==0 and lookup['gate_held'] and b['gate_held'] and not b['b_copy_started'],'real partial/queued boundary missing')
    require(b['controller']['pending_copies']>=2 and b['controller']['output_reservations']>=2 and b['controller']['output_bytes']>b['reservation']['host_bytes'],'A/B resource overlap not proven')
    require(ready['producer_ready'] and release['reason']=='verified_b_submit','gate released without real B CUDA completion')
    require(aux['seq']<gate['seq']<lookup['seq']<b['seq']<ready['seq']<release['seq'],'WRITING→actual B submit partial order missing')
    require(acopy['worker']==aux['worker']==bcopy['worker'] and acopy['thread'].startswith('spec-feature-capture'),'copy worker changed')
    commits=[r for r in selected if r['rank']==0 and r['event']=='cf_publish_ready']
    reserves=[r for r in selected if r['rank']==0 and r['event']=='cf_reserve']
    expected_starts=list(range(0,1280,256))
    for actor in ('A','B'):
        owned=[r for r in reserves if r['actor']==actor]
        require(sorted(r['key']['start'] for r in owned)==expected_starts,'missing or duplicated actual fill attempts')
    require(len(commits)==5 and all(r['actor']=='A' and r['published'] for r in commits),'not exactly one A committer per complete chunk')
    a_reservations=[r for r in reserves if r['actor']=='A']
    key_id=lambda row:json.dumps(row['key'],sort_keys=True,separators=(',',':'))
    require(Counter(map(key_id,commits))==Counter(map(key_id,a_reservations)),
            'commits do not cover every distinct reserved content key exactly once')
    require(len({r['fill_id'] for r in a_reservations})==5,'A reused a fill reservation across keys')
    for committed in commits:
        key=committed['key'];owners=[r for r in reserves if r['key']==key]
        require(len(owners)==2,'content key differs or extra reservation observed')
        ar=next(r for r in owners if r['actor']=='A');br=next(r for r in owners if r['actor']=='B')
        require(ar['owner']==identities['A'] and br['owner']==identities['B'],'reserve owner does not bind actual execution')
        require(ar['fill_id']==committed['fill_id'] and ar['before']['state']=='MISSING' and ar['reason']=='RESERVED','A did not own original reservation')
        require(br['fill_id'] is None and br['reason']=='READY' and br['before']['state']=='READY' and not br['before']['fill_ids'],'B did not naturally READY-skip')
        require(committed['after']['state']=='READY' and committed['after']['features']==['aux','last_hidden'] and committed['copied']==['aux','last_hidden'],'incomplete all-feature READY')
        require(release['seq']<committed['seq']<bcopy['seq']<br['seq'],'B fill ran before A complete publication')
    require(not any(r['event'] in ('cf_gate_timeout',) for r in selected),'fixture timed out; overlap unexercised')
    return {'natural_overlap':True,'single_copy_worker':True,'single_committer_per_key':True,
            'ready_skips':5,'partial_ready_visible':False,'key':a['key'],'executions':{'A':a['execution'],'B':b['execution']}}


def _feature_report(features,reference):
    import torch
    result={}
    for name,value in features.items():
        raw=value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        different=int(torch.count_nonzero(value!=reference[name]).item())
        result[name]={'sha256':hashlib.sha256(raw).hexdigest(),'different_elements':different,'elements':value.numel()}
    return result


def run_case(client,directory,row):
    """Only called before the common driver golden/warmup in a fresh server."""
    from actual_faults.scenarios import command,rank_events,check_features
    directory=Path(directory);config=client.config;validate_config(config)
    if len(row['input_ids'])!=1280:raise AssertionError('pair requires the unchanged native extension-1280 row')
    stats=client.cleanup()
    if stats['cache']['ready_blocks'] or stats['cache']['writing_blocks']:raise AssertionError('pair was prewarmed; fresh key not proven')
    initial=command(directory,'snapshot',tp_size=client.tp_size,timeout=client.timeout)
    for snapshot in initial:
        names=' '.join(snapshot['engine']['cache_components']).upper()
        if 'FULL' not in names or 'MAMBA' not in names:raise AssertionError('actual A3B FULL+MAMBA backend absent')
    case_id='same-content-fill';actors={name:{'rid':'actual-'+uuid.uuid4().hex,'sample_id':client.run_id+':'+case_id+'-'+name} for name in ('A','B')}
    arm={'kind':KIND,'case_id':case_id,'actors':actors,'N':1280,'token_sha256':token_digest(row['input_ids']),
         'ranks':[0],'gate_timeout_s':float(config.get('fill_gate_timeout_s',45))}
    atomic_json(directory/'arm.json',arm);result={'kind':KIND,'case_id':case_id,'actors':actors,'passed':False}
    try:
        af,_=client.submit(case_id+'-A',row,rid=actors['A']['rid'])
        partial=directory/f'fill-partial-{case_id}.json';error=directory/'concurrent-fill-error.json'
        wait_for(lambda:partial.exists() or error.exists(),client.timeout,'actual partial cache fill before B')
        if error.exists():raise AssertionError(read_json(error))
        if af.done():raise AssertionError('A HTTP completed before its partial CPU cache-fill gate')
        bf,_=client.submit(case_id+'-B',row,rid=actors['B']['rid'])
        queued=directory/f'fill-b-submitted-{case_id}.json'
        wait_for(lambda:queued.exists() or error.exists(),client.timeout,'B natural CUDA snapshot/submit while A WRITING; serialized boundary is not coverage')
        if error.exists():raise AssertionError(read_json(error))
        a,ar=client.fetch(af,row);b,br=client.fetch(bf,row)
        result['receipt_oracle']=assert_receipts(rank_events(directory,case_id),arm,client.tp_size)
        for actor,ref in (('A',ar),('B',br)):
            wire=ref.metadata['capture_fence']
            if wire['sample_id']!=actors[actor]['sample_id'] or wire['attempt_id']!=result['receipt_oracle']['executions'][actor]['attempt_id']:
                raise AssertionError('delivered SampleRef does not bind the actual execution')
        if set(ar.feature_keys.values()) & set(br.feature_keys.values()):raise AssertionError('pair physical sample keys alias')
        atomic_json(directory/'arm.json',{})
        g,gr=client.fetch(client.submit(case_id+'-G',row,force_full=True)[0],row)
        if gr.metadata['capture_diagnostics']['L']!=0:raise AssertionError('golden did not run actual forced-full prefill')
        check_features(a,row,client.model,reference=g,**config['feature_tolerances'])
        check_features(b,row,client.model,reference=g,**config['feature_tolerances'])
        result['feature_comparisons']={'A_vs_G':_feature_report(a,g),'B_vs_G':_feature_report(b,g)}
        r,rr=client.fetch(client.submit(case_id+'-R',row)[0],row,reference=g)
        diag=rr.metadata['capture_diagnostics']
        if not 0<diag['L']<=diag['H']:raise AssertionError('READY cache not exercised by actual positive-L assembly')
        result['feature_comparisons']['R_vs_G']=_feature_report(r,g)
        result['diagnostics']={name:ref.metadata['capture_diagnostics'] for name,ref in (('A',ar),('B',br),('G',gr),('R',rr))}
        result['refs']={name:asdict(ref) for name,ref in (('A',ar),('B',br),('G',gr),('R',rr))}
        del a,b,g,r
        final=client.cleanup();result['clean_server']=final
        after=command(directory,'snapshot',tp_size=client.tp_size,timeout=client.timeout)
        for before,now in zip(initial,after):
            if now['engine']['locked_nodes']!=before['engine']['locked_nodes'] or now['engine']['req_pool_free']!=before['engine']['req_pool_free']:
                raise AssertionError('actual request engine locks/pool rows not drained')
        result['passed']=True
        atomic_json(directory/'concurrent-fill-result.json',result)
        return result
    finally:
        try:command(directory,'release_gate',tp_size=client.tp_size,timeout=min(5,client.timeout),case_id=case_id)
        except BaseException as exc:
            atomic_json(directory/'concurrent-fill-release-error.json',{'error':str(exc),'cleanup_proven':False})
            if result['passed']:raise
        finally:
            # Keep identities/partial receipts on failure for the existing outer ownership cleanup.
            if result['passed']:atomic_json(directory/'arm.json',{})
