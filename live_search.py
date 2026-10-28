"""Live public Hugging Face discovery and offline replay of captured candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import platform
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from engine import SparseIndex, clean, tokenize

ROOT = Path(__file__).resolve().parent
FIELDS = ('name', 'description', 'tags', 'column_names', 'readme')
MAX_BYTES = 4_000_000
VERSION = 1


def fetch(url, timeout=20):
    """Bounded public GET; two retries for throttling or temporary server failures."""
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'DatasetLexicalBaseline/1.1', 'Accept': '*/*'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise ValueError('Response exceeds 4 MB limit')
                return body, dict(response.headers)
        except urllib.error.HTTPError as error:
            if attempt == 2 or (error.code != 429 and error.code < 500):
                raise
            time.sleep(attempt + 1)
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def discovery_terms(query):
    # No LLM rewriting. Phrase + first seven distinct task tokens, in query order.
    values = [query.strip()] + list(dict.fromkeys(tokenize(query, query_keywords=True)))[:7]
    return list(dict.fromkeys(x for x in values if x))


def capture(query, candidates, folder, timeout=20, getter=fetch):
    folder.mkdir(parents=True, exist_ok=False)
    raw = folder / 'raw'; raw.mkdir()
    requests = []; errors = []
    terms = discovery_terms(query)
    lists = []
    for i, term in enumerate(terms):
        url = 'https://huggingface.co/api/datasets?' + urllib.parse.urlencode({
            'search': term, 'limit': candidates, 'full': 'true', 'sort': 'downloads', 'direction': -1})
        entry = {'url': url, 'term': term}
        try:
            body, headers = getter(url, timeout)
            items = json.loads(body)
            if not isinstance(items, list): raise ValueError('Dataset list response is not a list')
            (raw / f'list-{i}.json').write_bytes(body)
            entry.update(file=f'raw/list-{i}.json', sha256=hashlib.sha256(body).hexdigest(), headers=headers)
            lists.append(items)
        except (OSError, ValueError) as error:
            entry['error'] = str(error); errors.append({'url': url, 'error': str(error)})
            lists.append([])
        requests.append(entry)
    # Round-robin interleave query sources so a broad first term cannot consume all slots.
    selected = {}; max_len = max(map(len, lists), default=0)
    for position in range(max_len):
        for items in lists:
            if position >= len(items): continue
            item = items[position]; did = item.get('id', '')
            if not isinstance(did, str) or not re.fullmatch(r'[\w.-]+(?:/[\w.-]+)?', did): continue
            if did not in selected: selected[did] = item
            if len(selected) >= candidates: break
        if len(selected) >= candidates: break

    def card(pair):
        number, (did, item) = pair
        sha = item.get('sha', '')
        revision = sha if isinstance(sha, str) and re.fullmatch('[a-fA-F0-9]{40}', sha) else 'main'
        url = 'https://huggingface.co/datasets/' + urllib.parse.quote(did, safe='/') + '/resolve/' + revision + '/README.md'
        evidence = dict(dataset_id=did, url=url, revision=revision)
        text = ''
        try:
            body, headers = getter(url, timeout)
            (raw / f'card-{number}.md').write_bytes(body)
            decoded = body.decode('utf-8', errors='replace'); text = decoded[:100_000]
            evidence.update(file=f'raw/card-{number}.md', sha256=hashlib.sha256(body).hexdigest(),
                            headers=headers, locally_capped=len(decoded)>100_000)
        except (OSError, ValueError) as error:
            evidence['error'] = str(error)
        row = dict(dataset_id=did, name=did, description=item.get('description') or '',
                   tags=item.get('tags') or [], column_names=[], readme=text,
                   url='https://huggingface.co/datasets/' + did, revision=revision)
        return row, evidence

    with ThreadPoolExecutor(max_workers=4) as pool:
        cards = list(pool.map(card, enumerate(selected.items())))
    rows = [row for row, evidence in cards]
    corpus = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows).encode('utf-8')
    (folder/'corpus.jsonl').write_bytes(corpus)
    info = dict(schema_version=VERSION, captured_at=datetime.now(timezone.utc).isoformat(), query=query,
                candidate_limit=candidates, candidate_count=len(rows), discovery_terms=terms,
                python=platform.python_version(), packages={p:importlib.metadata.version(p) for p in ('numpy','nltk')},
                engine_sha256=hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest(),
                scope='Hugging Face dataset API name-search candidates only; not a full Hub search',
                discovery_policy='phrase plus first seven task tokens using QUERY_STOP; downloads order; round-robin deduplication',
                fields=FIELDS, k1=1.2, b=0.75, corpus_sha256=hashlib.sha256(corpus).hexdigest(),
                list_requests=requests, cards=[evidence for row, evidence in cards],
                errors=errors, successful_list_requests=sum('error' not in entry for entry in requests),
                card_failures=sum('error' in evidence for row, evidence in cards))
    (folder/'manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
    return rows, info


def load_snapshot(folder):
    info = json.loads((folder/'manifest.json').read_text())
    if info.get('schema_version') != VERSION: raise ValueError('Unsupported snapshot schema')
    body = (folder/'corpus.jsonl').read_bytes()
    if hashlib.sha256(body).hexdigest() != info['corpus_sha256']:
        raise ValueError('Snapshot corpus checksum mismatch')
    rows = [json.loads(line) for line in body.decode('utf-8').split('\n') if line.strip()]
    if len(rows) != info['candidate_count'] or len({r['dataset_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid snapshot candidate count or duplicate IDs')
    return rows, info


def rank(rows, query, method, top_k):
    index = SparseIndex(rows, fields=FIELDS)
    records = {r['dataset_id']: r for r in rows}
    return [dict(query_id='adhoc', dataset_id=did, rank=i, score=score,
                 url=records[did]['url'], description=' '.join(clean(records[did].get('readme') or records[did].get('description',''), mode='section').split())[:280])
            for i, (did, score) in enumerate(index.search(query, top_k, keyword=method=='keyword'), 1)]


def execute(args):
    if args.action == 'online':
        folder = args.snapshot_dir or ROOT/'results/live'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
        print('Fetching public Hugging Face metadata and dataset cards...', file=sys.stderr)
        rows, info = capture(args.query, args.candidates, folder, args.timeout)
        query = args.query
    else:
        folder = args.snapshot; rows, info = load_snapshot(folder); query = args.query or info['query']
    results = rank(rows, query, args.method, args.top_k)
    # A unique run filename keeps both baseline runs and repeated runs intact.
    run_path = folder/(args.method+'-'+uuid.uuid4().hex[:8]+'.json')
    output = dict(query=query, method=args.method, candidate_count=len(rows), results=results,
                  corpus_sha256=info['corpus_sha256'], engine_sha256=hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest(), snapshot=str(folder.resolve()), k1=1.2, b=.75)
    run_path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(f"Candidates: {len(rows)} | Method: {args.method} | Card failures: {info['card_failures']}", file=sys.stderr)
    print('Scope: reranking captured HF name-search candidates, not all Hub datasets.', file=sys.stderr)
    if info['errors']:
        print(f"Warning: {len(info['errors'])} discovery requests failed; see manifest.json.", file=sys.stderr)
    print('Snapshot:', folder.resolve(), file=sys.stderr)
    print('Run:', run_path.resolve(), file=sys.stderr)
    if not info['successful_list_requests']:
        print('ERROR: All discovery requests failed. See manifest.json for details.', file=sys.stderr)
        return 2
    if not results:
        print('No positive lexical matches. Try shorter English topic words or increase --candidates.', file=sys.stderr)
    for result in results:
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"\n{result['rank']}. {result['dataset_id']} | score={result['score']:.6f}\n   {result['url']}\n   {result['description']}")
    return 0


def add_commands(sub):
    for action in ('online', 'replay'):
        p = sub.add_parser(action, help='Live HF candidate search' if action=='online' else 'Offline search over a captured live candidate pool')
        p.add_argument('--query', required=action=='online')
        p.add_argument('--method', choices=['keyword','bm25_document'], default='bm25_document')
        p.add_argument('--top-k', type=int, default=5)
        p.add_argument('--json', action='store_true', help='Print flat ranked JSONL to stdout')
        if action=='online':
            p.add_argument('--candidates', type=int, default=50, help='Maximum unique candidate datasets (1-500)')
            p.add_argument('--snapshot-dir', type=Path)
            p.add_argument('--timeout', type=int, default=20, help='Timeout per HTTP attempt in seconds (1-60)')
        else: p.add_argument('--snapshot', type=Path, required=True)


def validate_args(parser, args):
    if args.top_k < 1: parser.error('--top-k must be positive')
    if args.query is not None and not args.query.strip(): parser.error('--query must not be blank')
    if args.action=='online':
        if not 1 <= args.candidates <= 500: parser.error('--candidates must be between 1 and 500')
        if not 1 <= args.timeout <= 60: parser.error('--timeout must be between 1 and 60')
        if args.snapshot_dir and args.snapshot_dir.exists(): parser.error('--snapshot-dir must be a new directory')
