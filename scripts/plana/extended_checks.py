"""Additional independent evidence checks; no scheduler/cache implementation."""
from __future__ import annotations

import json
import math
from pathlib import Path

from validation_helpers import tensor_errors,tensor_sha256


def compare_metric_trees(actual,expected,*,rtol,atol,path='metrics'):
    """Check all teacher metrics and loss numerator/denominator leaves."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):
            raise AssertionError(f'{path}: metric keys changed')
        return sum((compare_metric_trees(actual[key],expected[key],rtol=rtol,atol=atol,path=f'{path}/{key}') for key in sorted(expected)),[])
    if isinstance(expected,(list,tuple)):
        if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):
            raise AssertionError(f'{path}: metric length changed')
        return sum((compare_metric_trees(a,b,rtol=rtol,atol=atol,path=f'{path}/{i}') for i,(a,b) in enumerate(zip(actual,expected,strict=True))),[])
    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isfinite(expected):
            raise AssertionError(f'{path}: nonfinite or nonnumeric metric')
        absolute=abs(actual-expected)
        return [{'path':path,'actual':actual,'expected':expected,'max_abs':absolute,'passed':absolute<=atol+rtol*abs(expected)}]
    if actual!=expected:
        raise AssertionError(f'{path}: metadata metric changed')
    return []


def checkpoint_subset_audit(bundle,checkpoint_path):
    import torch
    from safetensors import safe_open
    directory=Path(checkpoint_path)
    index=directory/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text())['weight_map']
    elif (directory/'model.safetensors').exists():
        with safe_open(str(directory/'model.safetensors'),framework='pt',device='cpu') as source:
            mapping={key:'model.safetensors' for key in source.keys()}
    else:
        raise AssertionError('checkpoint byte audit requires actual safetensors checkpoint')
    state=bundle.draft_model.state_dict();keys=sorted(set(mapping)&set(state))
    if len(keys)<4:
        raise AssertionError('too few checkpoint tensors to audit pretrained initialization')
    chosen={keys[0],keys[len(keys)//2],keys[-1]}
    chosen.update(key for key in keys if key=='fc.weight' or key=='hidden_norm.weight')
    layer_keys=[key for key in keys if key.startswith('layers.')]
    if layer_keys:chosen.update((layer_keys[0],layer_keys[-1]))
    records=[]
    for key in sorted(chosen):
        with safe_open(str(directory/mapping[key]),framework='pt',device='cpu') as source:
            stored=source.get_tensor(key)
        actual=state[key].detach().cpu();expected=stored.to(actual.dtype)
        if actual.shape!=expected.shape or not torch.equal(actual,expected):
            raise AssertionError(f'actual pretrained initialization differs from checkpoint bytes: {key}')
        records.append({'name':key,'shape':list(actual.shape),'stored_dtype':str(stored.dtype),'loaded_dtype':str(actual.dtype),
                        'actual_sha256':tensor_sha256(actual),'checkpoint_cast_sha256':tensor_sha256(expected),'passed':True})
    return records


def extract_capture_result(response,sample_id):
    found=[]
    def visit(value):
        if isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,dict):
            capture=(value.get('meta_info') or {}).get('spec_capture')
            if capture is not None:
                for item in capture if isinstance(capture,list) else [capture]:
                    if isinstance(item,dict) and item.get('sample_id')==sample_id:found.append(item)
            elif 'data' in value:visit(value['data'])
    visit(response)
    if len(found)!=1:
        raise AssertionError(f'expected one recorded capture result for {sample_id}, got {len(found)}')
    return found[0]


def validate_diagnostics(result,N,*,require_cache):
    fields=('H','L','N','span','execution_epoch','rank','issue_sequence','attempt_id','reservation_token','server_epoch','capture_signature','capture_namespace','reused_tokens')
    missing=[field for field in fields if field not in result]
    if missing:
        if require_cache:raise AssertionError(f'candidate omitted required diagnostics: {missing}')
        return {'H':0,'L':0,'N':N,'span':[0,N],'source':'official-only full prefill'}
    if type(result['H']) is not int or type(result['L']) is not int or result['N']!=N or not 0<=result['L']<=result['H']<=N:
        raise AssertionError(f'invalid H/L/N diagnostics {result}')
    if type(result['issue_sequence']) is not int or result['issue_sequence']<1:
        raise AssertionError('invalid server-issued attempt sequence')
    if result['span']!=[result['L'],N] or result['reused_tokens']!=result['L'] or result['L']>=N:
        raise AssertionError('absolute suffix or engine N-1 constraint violated')
    return {field:result[field] for field in fields}


def feature_regions(actual,expected,*,L,layers,hidden_size,rtol,atol):
    N=actual['hidden_states'].shape[1]
    records=[]
    for name in ('hidden_states','target_last_hidden_states'):
        for region,start,end in (('prefix',0,L),('seam',max(0,L-2),min(N,L+2)),('suffix',L,N)):
            records.append({'feature':name,'region':region,'start':start,'end':end,**tensor_errors(actual[name][:,start:end],expected[name][:,start:end],rtol=rtol,atol=atol)})
        if name=='hidden_states':
            for index,layer in enumerate(layers):
                a=actual[name][...,index*hidden_size:(index+1)*hidden_size]
                b=expected[name][...,index*hidden_size:(index+1)*hidden_size]
                records.append({'feature':name,'region':f'layer-{layer}',**tensor_errors(a,b,rtol=rtol,atol=atol)})
    return records
