# Search evaluations

Run from `catalog_search`.

```bash
# List cases: no API calls
.venv/bin/python evals/run_eval.py

# Test one known failure (paid API calls)
.venv/bin/python evals/run_eval.py --live --case micro_shorts_drawstring

# Compare whitespace variants
.venv/bin/python evals/run_eval.py --live --case unlined_shirt --case unlined_shirt_spaces

# Full suite; repeat to measure model variability
.venv/bin/python evals/run_eval.py --live --repeat 2
```

Live runs use your `.env` OpenAI/Pinecone credentials. They do not re-extract,
re-embed products, or upload records. They incur query parsing/embedding,
verification, and Pinecone read usage. Token usage is retained from parser and
verifier responses; embedding usage and dollar totals are not currently tracked.

Reports in `evals/runs/` are saved after every case, including errors, so a later
failure does not lose prior results. Exit code 1 means at least one scored case
failed or encountered an error; diagnostic-only cases do not determine success.

Cases cover unlined shirts, whitespace variants, micro shorts with drawstring
waists, drapey shoulder dresses, unsupported shoes, pants-versus-skirts/shorts,
price caps, and diagnostic wide-leg/below-knee requests. Positive product IDs
are seed labels from our examples, not exhaustive judgments. Review and expand
labels before using the suite as an accuracy benchmark. Known-positive recall
is not full catalog recall; precision is not computed without comprehensive
relevance labels. Empty results for comfy pants do not establish retrieval quality.

Reports contain actual filters, model names, token usage, per-product verdicts,
source quotes, result IDs, latency, known-positive recall, and failure stages.
Whitespace comparisons evaluate both result-ID sets and actual retrieval filters;
rank differences and repeated-run variance require review of saved responses.
Candidate misses can come from retrieval, scope blocking, verification, or top-k
truncation. The report distinguishes observed stages without guessing a root cause.

For an equivalent-pair comparison, select both cases in the same run. Source
quotes are not sufficient to prove semantic entailment: review borderline results
manually. No live calls happen unless `--live` is passed.

Blue-jeans regression: `.venv/bin/python evals/run_eval.py --live --case blue_jeans_under_100`
(add `--no-verify` to inspect retrieval alone). Checks category, exclusive price cap,
blue/jeans metadata, and exclusion of the known Delaynie shorts false positive.
