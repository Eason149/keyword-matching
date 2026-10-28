# Fixed and live dataset search

The fixed benchmark remains unchanged. Live mode discovers public Hugging Face candidates, retrieves their README cards, then reranks the captured pool with the same keyword/BM25 implementation. No new dependency or API key is required.

## Live search

From the project directory with the existing virtual environment activated:

```bash
python baseline.py online --query "medical question answering" --top-k 5
python baseline.py online --query "Find datasets to improve the model at counting and probability problems." --candidates 50 --top-k 5
```

The default is BM25 over at most 50 unique candidates. To use keyword overlap, add `--method keyword`. Use `--candidates 100` for a larger pool (maximum 500). This is a cap, not a guarantee; the provider can return fewer matches. The default timeout is 20 seconds per HTTP attempt, configurable with `--timeout` (1–60). Network search takes longer than fixed local search.

Each result prints the dataset repository ID, score, direct Hugging Face URL and a card excerpt. Excerpts are presentation aids, not evidence of verified training suitability. Add `--json` for flat ranked JSONL on stdout; progress and snapshot paths go to stderr.

## Save and replay without network

```bash
python baseline.py online --query "counting probability problems" --candidates 50 --snapshot-dir results/live/counting_demo
python baseline.py replay --snapshot results/live/counting_demo --method bm25_document
python baseline.py replay --snapshot results/live/counting_demo --method keyword
```

`--snapshot-dir` must be new; an existing directory is rejected to protect saved captures. Without it, the program creates a timestamped directory under `results/live/`. Replaying uses no network, verifies the saved corpus checksum, and defaults to the original query. An optional `--query` reranks the existing pool for a different query; it does not discover new candidates.

A snapshot stores `corpus.jsonl`, raw API lists and README bytes, response hashes and headers, query and capture time, candidate-selection settings, available repository revisions, errors and environment metadata. Each invocation saves a separate method run JSON. Share the same snapshot with the team to compare methods on identical candidates. Large snapshots are ignored by Git.

## Candidate scope and limitations

Discovery uses the public `/api/datasets` **name-search** parameter. It requests the original phrase plus up to seven distinct query tokens after removing the fixed QUERY_STOP boilerplate list. It requests each list in descending download order, interleaves results round-robin, deduplicates exact repository IDs, and stops at the candidate cap. These are heuristic candidate-generation choices; they do not translate, infer synonyms or search all README text on the Hub. Long natural-language requests are accepted, but short topic terms can give a better pool. The original query is retained for baseline scoring.

README retrieval uses the commit SHA returned in the metadata whenever available; otherwise it uses `main` and records that fallback. Up to four cards are fetched concurrently. Successful raw responses are cached for replay; READMEs above 100,000 characters are capped for indexing. Each HTTP body is limited to 4 MB. Public access only: inaccessible or gated cards are logged, with available metadata still indexed. HTTP 429 and temporary server/network failures receive at most two retries. If every list request fails the command exits with status 2 and retains diagnostics; a successful empty search reports no matches. Partial failures are reported rather than hidden.

BM25 statistics are calculated within the captured pool. Changing the pool can change both rankings and scores; scores across snapshots are not directly comparable. If candidate discovery misses a dataset, local reranking cannot recover it. Empty or zero-score results are not padded. Live output has no independent relevance labels and does not yield automatic accuracy estimates.

For the official fixed baseline experiment continue using:

```bash
python baseline.py reproduce
python baseline.py search --corpus data/benchmark_corpus.jsonl --query "image classification" --method keyword --top-k 5
```

Formal team comparisons should retain the fixed benchmark or share one captured live pool. Running two independent live searches and treating them as the same retrieval population is not a controlled comparison. No actual training data are downloaded and no downstream model is trained.

## Checks

```bash
python test_live_search.py
python test_baseline.py
```

The offline checks cover snapshot/replay parity, checksums, candidate deduplication and interleaving, missing cards, partial and total discovery failure, empty lexical matches, and the team-run contract. A real public API smoke test is recorded in the local `results/live/` directory; live rankings may change over time.

Sources: [Hugging Face Hub API](https://huggingface.co/docs/hub/api), [Dataset listing API client documentation](https://huggingface.co/docs/huggingface_hub/main/package_reference/hf_api).
