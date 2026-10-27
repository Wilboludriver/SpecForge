#!/usr/bin/env python3
"""Assert all saved teacher metrics and loss terms from a completed replay."""
import argparse
import json
from pathlib import Path

from extended_checks import compare_metric_trees
from validation_helpers import append_json,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.run_dir);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    tolerances=json.loads((root/'tolerances-predeclared.json').read_text())
    rows=[json.loads(line) for line in (root/'loss-gradient-results.jsonl').read_text().splitlines()]
    if len(rows)!=16:raise AssertionError(f'all16 gradient rows required, got {len(rows)}')
    checks=[]
    for row in rows:
        records=compare_metric_trees(row['actual_metrics'],row['reference_metrics'],rtol=tolerances['loss_rtol'],atol=tolerances['loss_atol'])
        if not any('/loss_terms/' in record['path'] for record in records):raise AssertionError('loss numerator/denominator evidence missing')
        if not any('teacher/' in record['path'] for record in records):raise AssertionError('teacher metric evidence missing')
        append_json(output/'metric-comparisons.jsonl',{'case':row['case'],'metrics':records,'passed':all(v['passed'] for v in records)})
        checks.extend(records)
    result={'cases':len(rows),'numeric_metric_leaves':len(checks),'max_abs':max(record['max_abs'] for record in checks),'passed':all(record['passed'] for record in checks)}
    write_json(output/'result.json',result);print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
