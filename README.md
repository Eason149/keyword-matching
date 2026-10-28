# Keyword Matching and BM25 Baselines

Yi's lexical baselines for comparing the team's dataset-discovery method. This repository implements two established reference systems, not the team method.

## Fixed and live search modes

Fixed mode remains the reproducible benchmark. Live mode discovers public Hugging Face name-search candidates and reranks their metadata and cards. After activating the environment:

```bash
python baseline.py online --query "medical question answering" --top-k 5
python baseline.py online --query "Find datasets to improve the model at counting and probability problems." --candidates 50 --snapshot-dir results/live/counting_demo
python baseline.py replay --snapshot results/live/counting_demo --method keyword
python baseline.py replay --snapshot results/live/counting_demo --method bm25_document
```

Use a new snapshot directory for each capture. Results include scores, descriptions and clickable URLs. `--json` produces JSONL. Replay is offline and gives both methods the same candidate pool. Live mode does not search all Hub dataset-card text and does not compute accuracy; see [LIVE_SEARCH.md](LIVE_SEARCH.md) for the scope, snapshot files and error handling. Run `python test_live_search.py` for offline integration checks. No additional dependencies are required.

## Quick start

```bash
git clone https://github.com/Eason149/keyword-matching.git
cd keyword-matching
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
python baseline.py reproduce
```

On Windows activate with `.venv\Scripts\activate`. The first run unpacks the included frozen benchmark locally. After dependency installation no network, model download, API key or Google Drive access is needed. Only NumPy and NLTK are required; no NLTK corpus download is used.

## Expected results

3,294 dataset documents; 392 labeled queries; 74 dev and 318 retrospective test queries. Test results:

| Method | nDCG@10 | Recall@100 | Recall@200 |
|---|---:|---:|---:|
| Keyword overlap | 0.086945 | 0.424173 | 0.513155 |
| BM25 k1=1.2 b=0.75 | 0.103801 | 0.420998 | 0.515847 |

The command asserts agreement with frozen values within 1e-12 and writes `results/baseline/{keyword,bm25_document}/{run.jsonl,per_query.csv,summary.json}`. `manifest.json` fingerprints the inputs and records parameters and package versions. Timing is machine-dependent.

## Search and compare your method

```bash
python baseline.py search --corpus data/benchmark_corpus.jsonl --query "math reasoning" --method bm25_document --top-k 5
python baseline.py evaluate --run team_run.jsonl --name team_method
```

Replace `bm25_document` with `keyword` for the other baseline. The evaluator accepts one JSON object per query (line order inside results is authoritative):

```json
{"query_id":"COPY_AN_EXACT_ID_FROM_data/benchmark_queries.jsonl","results":[{"dataset_id":"COPY_AN_ID_FROM_data/benchmark_corpus.jsonl","score":1.0}]}
```

Flat JSONL with `query_id`, `dataset_id`, `rank` and `score` also works. Flat ranks must be consecutive from 1. Use exact IDs and at most 200 unique datasets per query. Missing queries count as empty results; unknown queries and out-of-corpus or duplicate dataset IDs are rejected. Run `python test_baseline.py` for evaluator contract checks. An empty file is valid and receives zero scores.

Compare `results/baseline/team_method/summary.json` with both baselines on the **same split**. Keep query text, corpus, qrels and candidate budget fixed. Do not tune on the test results. With custom data use `--data-dir PATH`; provide the three benchmark JSON/JSONL files with the same schemas, unique IDs and dev/test splits. Custom runs skip the original-study metric assertion.

## What is implemented

Keyword overlap counts distinct shared query tokens. BM25 uses positive IDF, k1=1.2 and b=0.75. Both concatenate identical name, description, tags, column_names and readme fields; normalize Unicode/lowercase; apply the same tokenization and stopwords; return positive matches; break ties by dataset ID. No embeddings, LLM, query expansion or training are part of these primary baselines. See BASELINE_PROTOCOL.md for exact formulas and the human/training comparison plan.

## Scope and reproducibility

Official positives absent from the corpus remain in metric denominators. Unjudged documents receive zero gain. The split has prior exposure and is not a pristine unseen test set. No result here establishes team-method superiority or downstream training gains. The included English report retains the broader local study's supplementary ablations and HF catalogue results; this minimal repository reproduces **the two primary benchmark baselines**, not every extended experiment. Supplementary scripts and the HF catalogue remain in the local study.

See NOTICE.md and DATAFINDER_LICENSE for source provenance. The report is Lexical_Retrieval_Study.pdf.
