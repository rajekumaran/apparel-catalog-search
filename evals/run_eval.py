"""Preview: python evals/run_eval.py; paid run: add --live."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def assess(case, response):
    products=response.get('results',[])
    ids={p['product_id'] for p in products}
    candidates={p['product_id'] for p in response.get('verification',[])}
    failures=[]
    missing=set(case.get('expected_ids',[]))-ids
    if missing:failures.append('Missing expected products: '+', '.join(sorted(missing)))
    if case.get('expected_any_ids') and not ids.intersection(case['expected_any_ids']):
        failures.append('None of the expected alternative products returned')
    if ids.intersection(case.get('excluded_ids',[])):failures.append('Explicitly excluded product returned')
    if len(products)<case.get('min_results',0):failures.append('Too few results')
    if 'max_results' in case and len(products)>case['max_results']:failures.append('Too many results')
    if case.get('clarification_required') and not response.get('clarification'):failures.append('Missing clarification')
    if 'scope' in case and response.get('filters',{}).get('catalog_scope')!=case['scope']:failures.append('Incorrect catalog scope')
    for p in products:
        if case.get('category') and p.get('category')!=case['category']:failures.append(f"Wrong category: {p['product_id']}")
        if 'price_under' in case and (not isinstance(p.get('price'),(float,int)) or p['price']>=case['price_under']):failures.append(f"Price violation: {p['product_id']}")
        if any(term in p.get('title','').casefold() for term in case.get('forbidden_title_terms',[])):failures.append(f"Wrong garment type: {p['title']}")
        # Compare required canonical metadata for cases with explicit attribute labels.
        for name, allowed in case.get('required_attributes', {}).items():
            actual = p.get('attr_' + name)
            values = actual if isinstance(actual, list) else [actual]
            if not any(value in allowed for value in values):
                failures.append(f"Attribute violation {name}: {p['product_id']}")
        if response.get('verification_enabled', True) and not p.get('match_evidence',{}).get('quote'):failures.append(f"Missing match evidence: {p['product_id']}")
    gold=set(case.get('expected_ids',[]))
    return {'status':'diagnostic' if case.get('diagnostic_only') else ('fail' if failures else 'pass'),
            'failures':failures, 'result_count':len(products),
            'known_positive_recall':len(gold & ids)/len(gold) if gold else None,
            'missing_positive_stages':{pid:'verification_or_top_k' if pid in candidates else 'retrieval_or_scope' for pid in sorted(missing)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-verify',action='store_true',help='Skip the relevance verifier; evaluate fused retrieval only')
    parser.add_argument('--live',action='store_true',help='Call paid OpenAI APIs and Pinecone')
    parser.add_argument('--case',action='append',dest='selected',help='Case ID; repeat to select multiple')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--top-k',type=int,default=10)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.repeat<1 or not 1<=args.top_k<=100:parser.error('Invalid repeat or top-k')
    dataset=json.loads((ROOT/'evals/cases.json').read_text())
    all_cases=dataset['cases']
    if args.selected and set(args.selected)-{c['id'] for c in all_cases}:parser.error('Unknown case ID')
    cases=[c for c in all_cases if not args.selected or c['id'] in args.selected]
    if not args.live:
        print(f'Preview only: {len(cases)} cases × {args.repeat} run(s). No API calls.')
        for c in cases:print(c['id'],repr(c['query']), '(diagnostic)' if c.get('diagnostic_only') else '')
        print('Add --live to execute. Each supported query uses parsing, embedding, Pinecone retrieval, and verification.')
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    from search.pinecone_search import search
    output=args.output or ROOT/'evals/runs'/f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'dataset_version':dataset['version'],'top_k':args.top_k,'verification_enabled':not args.no_verify,'runs':[]}
    for repeat in range(args.repeat):
        previous={}
        for case in cases:
            start=time.perf_counter()
            try:
                response=search(case['query'],top_k=args.top_k,verify=not args.no_verify)
                outcome=assess(case,response)
                equivalent=previous.get(case.get('equivalent_to'))
                if equivalent:
                    left={p['product_id'] for p in equivalent.get('results',[])}
                    right={p['product_id'] for p in response.get('results',[])}
                    outcome['equivalence']={'same_result_ids':left==right,
                        'jaccard':len(left&right)/len(left|right) if left|right else 1.0,
                        'same_retrieval_filter':equivalent.get('retrieval_filter')==response.get('retrieval_filter')}
                    if left!=right or not outcome['equivalence']['same_retrieval_filter']:
                        outcome['failures'].append('Equivalent query produced different results or retrieval filters')
                        if outcome['status']!='diagnostic':outcome['status']='fail'
                previous[case['id']]=response
                record={'case_id':case['id'],'repeat':repeat+1,'query':case['query'],**outcome,'response':response}
            except Exception as error:
                record={'case_id':case['id'],'repeat':repeat+1,'status':'error','error':f'{type(error).__name__}: {error}'}
            record['seconds']=round(time.perf_counter()-start,3)
            report['runs'].append(record)
            report['summary']={status:sum(r['status']==status for r in report['runs']) for status in ['pass','fail','diagnostic','error']}
            temporary=output.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,default=str)+'\n');temporary.replace(output)
            print(case['id'],record['status'],f"{record['seconds']}s",flush=True)
    print('Report:',output)
    print(report['summary'])
    return int(bool(report['summary']['fail'] or report['summary']['error']))


if __name__=='__main__':raise SystemExit(main())
