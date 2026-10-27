"""Supplemental real-engine scenarios; the original native 16 remain mandatory."""
from __future__ import annotations

from copy import deepcopy

CHUNK_TOKENS = 384
CAPACITY_CHUNKS = 2


def build_cases(native_rows):
    """Use whole native rows, preserving token IDs and supervised masks."""
    selections = ((9, 'warm-native256'), (0, 'base-native1024'),
                  (12, 'bounded-h-native512'), (0, 'repeat-native1024'),
                  (0, 'hs-only-native1024'), (14, 'pressure-native1024'),
                  (0, 'kv-only-native1024'))
    if len(native_rows) != 16:
        raise AssertionError('supplement requires the complete original native16 fixture')
    cases = []
    for source_index, name in selections:
        row = deepcopy(native_rows[source_index])
        row['source_case_id'], row['source_case_index'] = row['id'], source_index
        row['id'] = name
        if not any(a > .5 and b > .5 for a, b in zip(row['loss_mask'], row['loss_mask'][1:])):
            raise AssertionError(f'{name}: whole native row lacks supervised anchors')
        cases.append(row)
    if [len(row['input_ids']) for row in cases] != [256, 1024, 512, 1024, 1024, 1024, 1024]:
        raise AssertionError('native fixture lengths changed')
    base = cases[1]['input_ids']
    if cases[0]['input_ids'] != base[:256] or cases[2]['input_ids'] != base[:512]:
        raise AssertionError('native warm and bounded-H cases must share the base prefix')
    if cases[5]['input_ids'][:384] == base[:384]:
        raise AssertionError('pressure row must have a different first cache key')
    return cases


def cache_capacity_bytes(info):
    return CAPACITY_CHUNKS * CHUNK_TOKENS * 2 * info['hidden_size'] * (len(info['layers']) + 1)


def validate_outcome(cases, diagnostics, actions):
    """Do not invent a maximal legal L: assert actual bounded restore and causes."""
    expected_h = [0, 0, 384, 768, 768, 0, 0]
    if len(diagnostics) != len(cases):
        raise AssertionError('supplement diagnostics incomplete')
    for row, diagnostic, H in zip(cases, diagnostics, expected_h, strict=True):
        N, L = len(row['input_ids']), diagnostic['L']
        if diagnostic['H'] != H or diagnostic['N'] != N or not 0 <= L <= H or L >= N:
            raise AssertionError(f'{row["id"]}: unexpected real H/L/N: {diagnostic}')
        if diagnostic['span'] != [L, N] or diagnostic['reused_tokens'] != L:
            raise AssertionError('actual capture span is inconsistent with L')
    if not 0 < diagnostics[2]['L'] < diagnostics[2]['H']:
        raise AssertionError('bounded-H fixture did not prove an actual earlier legal checkpoint')
    if diagnostics[4]['L'] != 0:
        raise AssertionError('engine flush did not force HS-only restore to L0')
    if diagnostics[6]['L'] != 0:
        raise AssertionError('KV-only fixture reused a missing HS prefix')
    flush, ordinary = actions['engine_flush'], actions['ordinary_uncaptured_kv']
    if flush['status'] != 200 or not flush['hs_cache_unchanged']:
        raise AssertionError('HS-only fixture did not preserve ready HS across engine flush')
    if not ordinary['cached_tokens'] > 0 or ordinary['has_spec_capture'] or not ordinary['hs_cache_unchanged']:
        raise AssertionError('ordinary request did not prove uncaptured KV availability')
    return {'actual_H': [d['H'] for d in diagnostics], 'actual_L': [d['L'] for d in diagnostics],
            'bounded_checkpoint': diagnostics[2], 'hs_only': diagnostics[4], 'kv_only': diagnostics[6]}
