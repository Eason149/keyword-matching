"""Offline lexical baselines and a common evaluator for team retrieval runs."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import zipfile
from engine import SparseIndex, score_ranking

ROOT = Path(__file__).resolve().parent
FIELDS = ('name', 'description', 'tags', 'column_names', 'readme')
METRICS = ('ndcg10', 'recall20', 'recall100', 'recall200', 'mrr200')

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]

def load_data(directory):
    corpus = read_rows(directory / 'benchmark_corpus.jsonl')
    queries = read_rows(directory / 'benchmark_queries.jsonl')
    qrels = json.loads((directory / 'benchmark_qrels.json').read_text())
    ids = [r['dataset_id'] for r in corpus]
    qids = [q['query_id'] for q in queries]
    if len(ids) != len(set(ids)) or len(qids) != len(set(qids)):
        raise ValueError('Duplicate dataset or query IDs')
    if set(qids) != set(qrels):
        raise ValueError('Queries and qrels must have identical query IDs')
    if any(q.get('split') not in ('dev', 'test') for q in queries):
        raise ValueError('Each query needs split dev or test')
    return corpus, queries, qrels

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def manifest(directory):
    return dict(python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ('numpy', 'nltk')},
                fields=FIELDS, k1=1.2, b=0.75, top_k=200,
                inputs={n: hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in
                        ('benchmark_corpus.jsonl', 'benchmark_queries.jsonl', 'benchmark_qrels.json')})

def evaluate(rankings, queries, qrels, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(query_id=q['query_id'], split=q['split'],
                 **score_ranking(rankings.get(q['query_id'], []), qrels[q['query_id']])) for q in queries]
    with (output/'per_query.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for split in ('dev', 'test', 'all'):
        selected = [r for r in rows if split == 'all' or r['split'] == split]
        summary.append(dict(split=split, n=len(selected), **{
            m: sum(r[m] for r in selected)/len(selected) if selected else None for m in METRICS}))
    save_json(output/'summary.json', summary)
    return summary

def parse_run(path, query_ids, corpus_ids):
    rows = read_rows(path); grouped = {}; nested = None
    for row in rows:
        qid = row['query_id']
        if qid not in query_ids: raise ValueError(f'Unknown query {qid}')
        current = 'results' in row
        if nested is not None and nested != current: raise ValueError('Mixed run formats')
        nested = current
        if current:
            if qid in grouped: raise ValueError(f'Duplicate query {qid}')
            grouped[qid] = row['results']
        else: grouped.setdefault(qid, []).append(row)
    result = {}
    for qid, hits in grouped.items():
        if not nested:
            ranks = [h['rank'] for h in hits]
            if any(type(r) is not int for r in ranks) or sorted(ranks) != list(range(1, len(hits)+1)):
                raise ValueError('Flat ranks must be consecutive integers from 1')
            hits = sorted(hits, key=lambda h:h['rank'])
        ids = [h['dataset_id'] for h in hits]
        if len(ids)>200 or len(ids)!=len(set(ids)): raise ValueError('At most 200 unique results per query')
        if set(ids)-corpus_ids: raise ValueError('Out-of-corpus dataset IDs')
        if any(type(h.get('score')) not in (int,float) or not math.isfinite(h['score']) for h in hits):
            raise ValueError('Every hit requires a finite numeric score')
        result[qid] = ids
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('reproduce', 'evaluate'):
        p=sub.add_parser(action); p.add_argument('--data-dir', type=Path, default=ROOT/'data')
        p.add_argument('--output-dir', type=Path, default=ROOT/'results/baseline')
        if action=='evaluate':
            p.add_argument('--run',type=Path,required=True);p.add_argument('--name',default='team_method')
    p=sub.add_parser('search');p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--query',required=True);p.add_argument('--method',choices=['keyword','bm25_document'],default='bm25_document')
    p.add_argument('--top-k',type=int,default=5)
    args=parser.parse_args()
    if args.action=='search':
        if args.top_k<1: parser.error('top-k must be positive')
        index=SparseIndex(read_rows(args.corpus),fields=FIELDS)
        for rank,(did,score) in enumerate(index.search(args.query,args.top_k,keyword=args.method=='keyword'),1):
            print(json.dumps(dict(query_id='adhoc',dataset_id=did,rank=rank,score=score)))
        return
    if args.data_dir.resolve() == (ROOT/'data').resolve() and not (args.data_dir/'benchmark_corpus.jsonl').exists():
        archive=ROOT/'benchmark_data.zip'
        if not archive.exists(): parser.error('Missing data directory and benchmark_data.zip')
        allowed={'data/benchmark_corpus.jsonl','data/benchmark_queries.jsonl','data/benchmark_qrels.json','results/benchmark/summary.json'}
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != allowed: raise ValueError('Unexpected benchmark archive contents')
            for name in sorted(allowed):
                target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    corpus,queries,qrels=load_data(args.data_dir)
    if args.action=='evaluate':
        if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
            parser.error('name must contain only letters, digits, underscores or hyphens')
        ranks=parse_run(args.run,{q['query_id'] for q in queries},{r['dataset_id'] for r in corpus})
        output=args.output_dir/args.name
        summary=evaluate(ranks,queries,qrels,output)
        info=manifest(args.data_dir);info['run_sha256']=hashlib.sha256(args.run.read_bytes()).hexdigest()
        info['missing_queries']=len(queries)-len(ranks);save_json(output/'manifest.json',info)
        print(json.dumps(summary,indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    index=SparseIndex(corpus,fields=FIELDS)
    frozen=json.loads((ROOT/'results/benchmark/summary.json').read_text()) if args.data_dir.resolve()==(ROOT/'data').resolve() else None
    for method in ('keyword','bm25_document'):
        output=args.output_dir/method;output.mkdir(parents=True,exist_ok=True);rankings={}
        with (output/'run.jsonl').open('w') as f:
            for q in queries:
                hits=index.search(q['text'],200,keyword=method=='keyword')
                rankings[q['query_id']]=[did for did,score in hits]
                f.write(json.dumps(dict(query_id=q['query_id'],results=[dict(dataset_id=d,score=s) for d,s in hits]))+'\n')
        summary=evaluate(rankings,queries,qrels,output)
        if frozen:
            expected=next(r for r in frozen if r['method']==method and r['split']=='test')
            actual=next(r for r in summary if r['split']=='test')
            if any(abs(actual[m]-expected[m])>1e-12 for m in METRICS):
                raise AssertionError(f'{method}: frozen metric mismatch')
        print(method,json.dumps(summary))
    info=manifest(args.data_dir);info['frozen_metric_check']='passed' if frozen else 'not applicable to custom data'
    save_json(args.output_dir/'manifest.json',info)
    print('Saved:',args.output_dir)

if __name__=='__main__': main()
