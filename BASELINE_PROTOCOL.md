<!-- PAGE -->

## 9 Baseline Specification for Team Comparison

Yi provides the lexical baselines. The team method must be submitted as a separate run; no current result demonstrates its superiority. The primary comparison is against both keyword overlap and standard full-document BM25. Report supplementary variants separately, including the development-selected BM25 setting, so conclusions do not depend on choosing a weak reference after seeing results.

### Inputs and Index Construction

Use one JSONL record per dataset, with a unique dataset_id. The indexed fields are name, description, tags, column_names and readme. Missing fields become empty strings and list values are joined by spaces. Both primary baselines concatenate the same fields with equal weight. In DataFinder these normalized fields contain the supplied dataset descriptions and paper information; the HF experiment contains dataset cards and metadata. Neither baseline reads training samples.

Basic cleaning removes HTML comments, script/style blocks, URLs and HTML tags; Markdown links retain their visible text. The tokenizer applies Unicode NFKC and lowercase, extracts ASCII letters and digits, and represents Chinese runs as characters and adjacent bigrams. The fixed stopword list is: a an the and or of for to in on with is are be this that dataset datasets data. Both primary methods use this identical processing. There is no stemming, translation, synonym expansion or query rewriting in the primary runs.

The implementation builds an inverted index mapping tokens to dataset IDs and term counts. Document length is the number of retained tokens, including repetitions. N is the number of documents, df(t) the number containing t, and avgdl the mean document length. Query terms are deduplicated for both methods.

### Exact Scoring and Ranking

Keyword score is the number of distinct query tokens found in a document. It is an OR match, not a phrase or all-terms requirement. BM25 sums the following contribution once per distinct matching query token t:

IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)).

Contribution(t,d) = IDF(t) times [tf(t,d) times (k1 + 1)] divided by [tf(t,d) + k1 times (1 - b + b times length(d) / avgdl)].

The fixed primary parameters are k1 = 1.2 and b = 0.75. Only positive scores are returned, in descending order; ties use ascending dataset_id. Return at most 200 candidates without padding empty matches. Raw scores across different methods are not calibrated or directly comparable.

The implementation is engine.py with NumPy arrays and NLTK Porter stemming for the optional stemming variant. It uses no external search server, embedding model, LLM, NLTK corpus download or model training. The supplementary parameter search uses only development queries and selects k1 = 1.2, b = 0; it remains distinct from the fixed primary BM25.

<!-- PAGE -->

## 10 Reproduction and Comparison Interface

### Offline Baseline Run

Use Python 3.12. From the repository root, create an environment with python3.12 -m venv .venv, activate it, then run python -m pip install -r requirements-baseline.txt. Run python baseline.py reproduce. Dependencies require an initial installation; the experiment itself uses only the included local benchmark files and does not contact Google Drive or Hugging Face.

The command rebuilds both primary indices, runs all 392 official queries, and exports ranked runs, per-query metrics and dev/test/all summaries into results/baseline/. It checks the 318-query test metrics against the frozen study values with an absolute tolerance of 1e-12. Latency is not expected to reproduce exactly across machines. The command records input SHA256 values, parameters and package versions in manifest.json.

For an individual search, run python baseline.py search --corpus data/benchmark_corpus.jsonl --query "math reasoning" --method bm25_document --top-k 5. Use --method keyword for the identical keyword baseline. The JSONL output includes query_id, dataset_id, rank and score. For project requests, the original search.py also supports --query-id p13 on the saved HF catalogue.

### Team Run Contract

Export one JSONL object per query with query_id and results. Each results list contains dataset_id and score objects in ranked order, with at most 200 unique dataset IDs. A flat JSONL form with query_id, dataset_id, rank and score is also accepted. Rank order is authoritative; scores are retained for traceability. The evaluator rejects unknown queries, duplicate datasets and out-of-corpus IDs. Absent queries receive an empty ranking and are included in evaluation.

Run python baseline.py evaluate --run team_run.jsonl --name team_method. The evaluator writes metrics to results/baseline/team_method/ using the same qrels and metric code. Use --output-dir to keep independent runs. Compare the test summary with keyword and bm25_document under results/baseline/. Do not mix 313-query keyphrase results with 318-query sentence results.

The default benchmark directory is data. A custom directory must contain benchmark_corpus.jsonl, benchmark_queries.jsonl and benchmark_qrels.json with the same schemas; pass --data-dir to reproduce or evaluate. Custom data do not use the frozen-result assertion. The manifest fingerprints the exact inputs, so all compared runs can be checked against the same corpus and judgments.

Reproduce and evaluate report macro nDCG@10, Recall@20, Recall@100, Recall@200 and MRR@200. Missing corpus positives remain in recall denominators and unjudged items have zero gain. The full study and independent metric checks remain available through experiments.py and validate.py; requirements.txt includes their additional dependencies.

<!-- PAGE -->

## 11 Fair Evaluation of the Team Method

### Retrieval Comparison

Freeze the corpus, query text, query IDs, qrels and candidate budget before comparing methods. Run keyword overlap, fixed BM25, development-selected BM25 and the team method on the same test population. Select parameters using development queries only. Report which additional metadata, semantic representations or query processing the team method uses. An end-to-end system may use such features, but its advantage should be described as a system-level result; a same-input ablation is needed to isolate a scoring contribution.

Use Recall@100 and Recall@200 to assess candidate coverage and nDCG@10 to assess early ranking. Report paired query-level differences, uncertainty intervals and multiplicity-adjusted tests for predefined comparisons. The retrospective split here has prior exposure; a fresh evaluation set is needed for a prospective generalization claim. The current report contains baseline results only, with no fabricated team-method numbers.

### Human Review

The proposed project evaluation pools the top five results of both primary baselines and the team method, deduplicates each query-dataset pair, hides method identity and randomizes presentation. Use relevance 0 for unrelated or unusable, 1 for partial fit, and 2 for suitable task fit. Separately record language, schema, split and license compatibility as supported, contradicted or unknown. Keep evidence passages and reviewer IDs. A second reviewer should independently assess a subset when available; single-reviewer judgments must be identified as such.

Compute nDCG@5 and the fraction of top-five results judged suitable on the same queries. The existing 645-pair annotation pool covers six lexical variants only; it must be expanded with the team output before being used for that comparison. Those cells are currently unjudged. This is a planned protocol, not a completed human study.

### Downstream Training

A proposed controlled study selects a small number of trainable tasks and fixes the base model, held-out test data, sample or token budget, preprocessing, training steps and random seeds across retrieval methods. Apply the same dataset-selection rule to every method, such as the first ranked eligible dataset, with eligibility and a maximum inspection depth defined in advance. Record failures to find eligible data instead of silently selecting another convenient task.

If manual selection from top-k is part of the workflow, give every method the same review budget and criteria, and hide method identity. Otherwise the comparison confounds retrieval with extra human curation. Keep a no-additional-training reference, inspect overlap with test data, and report task-specific outcomes with repeated training seeds where feasible. No downstream training has been performed in this study; a retrieval gain is not yet evidence of training utility.
