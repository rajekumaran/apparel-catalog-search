# Catalog search

## Code organization

Run commands from `catalog_search/`. Python components are grouped by responsibility:

```text
extraction/   # Category attribute extraction
storage/      # Load saved extractions into DuckDB
search/       # Query parsing, embeddings, Pinecone sync, BM25 and rank fusion
validation/   # Attribute evidence checks and query validation schemas
interfaces/   # Website API and notebook result display
web/          # Website HTML and static assets
tests/       # Offline unit tests
evals/       # Live search evaluation runner, cases and reports
scripts/      # Batch extraction and publishing shell commands
data/         # Taxonomies, saved extractions, database and manifests
```

Common commands (module paths replace the old root-level Python filenames):

```bash
# Extract attributes; existing batch scripts still work
.venv/bin/python -m extraction.extract_catalog tops
bash scripts/extract_additional.sh 20 --dry-run

# Load saved attributes, then publish embeddings and metadata
.venv/bin/python -m storage.load_attribute_tables
bash scripts/sync_catalog.sh

# Start the website
bash run.sh

# Offline tests and evaluation preview
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evals/run_eval.py
```

Notebook imports now use `from search import pinecone_search` and
`from interfaces.search_display import ...`. Restart the notebook kernel and run
its cells again after updating to this layout.


Fresh workspace for apparel attribute extraction and search experiments.

## Data

`data/apparel_catalog.duckdb` is a copy of the original
`week1_project/apparel_catalog.duckdb`. Keep this source catalog unchanged;
save enrichment results separately.

## Python environment

Python 3.12, with dependency definitions and the lockfile copied from
`apparel_catalog_app`. This retains the existing dependency baseline;
Jupyter kernel (`ipykernel`) and pandas are included as development dependencies.
Jev has not been added yet.

From this folder, create an independent environment:

```bash
uv sync --locked
```

Use `.venv/bin/python` as the interpreter for this project.
API credentials have not been copied. If needed, copy `.env.example` to `.env`
and set your credentials there.

No application code, embeddings, or derived attributes were copied.

## Explore tops

Open `01_explore_tops.ipynb` and select this folder's `.venv/bin/python`
as the notebook kernel. Run cells in order. Queries open DuckDB read-only.

## Pinecone semantic search

Permanent extraction results remain in JSON and DuckDB. All three extraction
scripts skip product IDs already saved, including previous taxonomy versions.
Source/taxonomy changes do not automatically trigger re-extraction.

`search/pinecone_search.py` builds documents for the 60 enriched products, omits unknown
attributes, caches embeddings in DuckDB's `product_embeddings` table, and records
acknowledged uploads in `pinecone_sync`. Metadata changes reuse cached vectors.
Failed uploads can be retried by rerunning sync. `--reupload` restores remote
records even if local sync status says they were uploaded; it reuses local vectors.
The target host and namespace are part of sync identity. Do not reuse this
namespace for a different embedding model.

Configure `.env` (never commit credentials):

```text
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_HOST=... # Existing dense-vector index: dimension 1536, cosine
PINECONE_NAMESPACE=catalog-text-embedding-3-small-v1
```

This implementation uses Pinecone's vector `upsert/query` interface, not a
Documents API index. It does not create or delete indexes.

```bash
uv sync --locked
.venv/bin/python -m search.pinecone_search preview  # Local only; no API charges
.venv/bin/python -m search.pinecone_search sync
.venv/bin/python -m search.pinecone_search search 'blue high-rise jeans under $80'
.venv/bin/python -m search.pinecone_search sync --reupload
```

Natural-language search uses a Pydantic-validated `ShopperIntent`, converts explicit
requirements to Pinecone metadata filters, embeds the query, and returns scores
and display metadata directly from Pinecone. No DuckDB reads occur during search.
USD is the explicit store default. Subjective preferences stay in semantic text.
Unsupported constraints are reported instead of intentionally relaxed. Scores
indicate similarity, not factual confidence; no-match results remain empty.
Pinecone writes may take time to become visible. Deleted products require a
separate removal workflow; sync currently only inserts/updates saved products.

Tests (no network or API costs):

```bash
.venv/bin/python -m unittest tests.test_pinecone_search
```

Implementation references: [Pinecone vectors and metadata filters](https://sdk.pinecone.io/python/how-to/vectors/upsert-and-query.html),
[OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings).

## Website / Replit

The new Thread storefront uses the Pinecone search backend directly. It includes
free-text search, example searches, image cards, interpreted-filter chips, product
attributes, retailer links, and loading/empty/error states. It searches the 60
indexed products. There is no new extraction or indexing during a website search.

Run locally from this folder:

```bash
uv run --locked uvicorn interfaces.web_api:app --host 0.0.0.0 --port 3000
```

Open http://localhost:3000. On Replit, import the contents of
`../catalog_search_replit.zip` as the workspace root and add these Secrets:
`OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME=catalog-search`, and
`PINECONE_NAMESPACE=catalog-text-embedding-3-small-v1`.
Optionally set `PINECONE_INDEX_HOST` to the index host. Press Run using `.replit`.
The supplied lockfile is installed by `uv run --locked`. The website does not
need DuckDB data files; runtime search reads Pinecone. Credentials remain on the
server, never in browser code. The public search endpoint incurs API charges;
add hosting-level access controls/rate limits before opening it to broad traffic.
Replit hosting has not been verified from this local workspace.

## Per-attribute evidence

New extractions use `attribute_evidence.py` and the `evidence-v1` format across
all three extraction scripts. Each known value requires an exact source quote:

```json
{
  "attributes": {"neckline": "v_neck"},
  "evidence": {
    "neckline": [{"value": "v_neck", "source_field": "Description", "quote": "V-neck"}]
  }
}
```

The extractor saves the five allowed source fields in `source_snapshot`. Python
checks quoted substrings, allowed values, single/multi cardinality, and exclusive
sentinels before saving. Invalid responses are retried up to three attempts.
Omitted attributes still default to unknown. Matching quotes establish provenance,
not semantic truth; reviewers must still assess whether a quote supports a value.

`storage/load_attribute_tables.py` imports evidence into `attribute_evidence` with columns
`extraction_id`, `attribute_name`, `attribute_value`, `source_field`, and `quote`.
Evidence remains available in `extractions.raw_result` and source JSON as well.
Existing saved products remain untouched and have no retroactively invented
quotes. No additional LLM calls are made merely by running the loader.

## Extract additional batches

From `catalog_search`, preview 20 new products per category (60 total), without
API calls or writes:

```bash
bash scripts/extract_additional.sh 20 --dry-run
```

Extract the batch with per-value evidence and import into DuckDB:

```bash
bash scripts/extract_additional.sh 20
```

Extract and publish embeddings/metadata to Pinecone afterward:

```bash
bash scripts/extract_additional.sh 20 --sync
```

Already extracted IDs in JSON or DuckDB are excluded. Selection is stable CSV
order, with distinct product IDs. If fewer new products remain, all remaining
products are selected. Each category's exact source rows are saved under the
printed `data/batches/<batch>/` directory. Keep those plans for retries:

```bash
bash scripts/extract_additional.sh --resume data/batches/YOUR-BATCH --sync
```

Resume uses the saved plans, skips completed products, and retries unfinished
ones. A fresh invocation without `--resume` intentionally starts a new batch.
Failures stop the script; successful per-product saves are retained. Avoid
running multiple extraction/sync processes concurrently against this folder.
New LLM calls incur usage charges; --sync also creates uncached embeddings.
Existing extractions are never automatically re-extracted.

To publish a previously extracted batch separately:

```bash
bash scripts/sync_catalog.sh
```

This loads saved results, syncs Pinecone, and refreshes browse ID/retailer
manifests for acknowledged current uploads. No extraction calls occur during
this publishing step. Restart/reload the local site; Replit also needs the
updated manifest files. Website copy and search limits still reflect the
original 60-product prototype and should be expanded before larger deployments.

## Semantic retrieval with relevance verification

`search()` normalizes whitespace and uses GPT-5.6 Luna for query parsing.
Free-text attribute interpretations remain visible in `filters` but are not
Pinecone equality requirements. Category, currency, price, and explicit site
selections still constrain retrieval. `retrieval_filter` shows the actual filter.

The backend retrieves up to 100 candidates (at least 30 when available), then
makes an additional GPT-5.6 Luna call to judge relevance using original title,
description, material, recorded fit, and color. Exact source quotes are checked
locally. Rejected/insufficient-evidence candidates are excluded; accepted results
preserve vector rank and include `match_reason` and `match_evidence`.
`verification` contains all candidate verdicts and `verifier_info` reports model
and token usage. Empty matches stay empty; a failed review does not return
unverified candidates. This model-based review still requires evaluation and
is not a guarantee of factual accuracy. Candidate limits can affect recall.

The source fields are copied to Pinecone by `sync()` without changing embedding
text. Existing cached vectors can therefore be reused. Searches continue to
avoid DuckDB reads. The extra review call increases latency and API costs.

## BM25 + semantic hybrid retrieval

`hybrid_retrieval.py` scores keyword matches using BM25 over source text and known
attributes fetched from Pinecone for `data/indexed_product_ids.json`. No DuckDB
read or additional embedding call is used for BM25. The current implementation
fetches the small corpus per search; it is a prototype, not a scalable hosted
BM25 index. Refresh the manifest after publishing new products.

Both retrieval branches apply the same metadata filters. The top keyword and
vector rankings are combined with reciprocal rank fusion (equal weights,
constant 60), capped at 100 candidates, then sent to the existing verifier.
The verifier preserves fused ordering for accepted products.

`response['retrieval_info']` reports branch counts. Each result includes
`retrieval_sources`, `semantic_rank`/`semantic_score` when vector-retrieved, and
`bm25_rank`/`bm25_score` when keyword-retrieved. `score` is now the fused ranking
score, NOT cosine similarity or confidence. BM25 and cosine scores are not added
directly because their scales differ. Keyword scoring uses case-folded Unicode
word tokens, without stemming or synonym expansion. The semantic branch handles
related wording. BM25 may increase candidate count, verification latency, and
verification token usage; it has no separate model charge. Pinecone fetches still
consume service usage.

## Search latency controls

Search now verifies the top 20 fused candidates by default, with two concurrent
10-product review batches. Fused order is preserved. This reduces work but can
miss relevant products below rank 20; requesting top_k above 20 does not increase
the review budget automatically. To compare broader retrieval during evaluation:

```python
response = search(query, top_k=10, verification_limit=100)
print(response['timings'])
print(response['retrieval_info'])
```

The fetched BM25 corpus is cached in process for five minutes. Cache identity
includes Pinecone host, index name, namespace, and product-ID list. Changes to
IDs invalidate the cache; metadata-only updates become visible after expiry.
Use `pinecone_search.clear_search_cache()` to refresh immediately in a running
notebook/server process. Sync clears the cache in its own process only. This
cache saves Pinecone fetching; BM25 scoring itself is still recalculated.

Timings cover parsing, query embedding, index connection, vector search, corpus
fetch/cache lookup, BM25/fusion, verification, and total search seconds. They do
not include notebook image downloads. Concurrent verification does not itself
reduce token charges; reviewing fewer candidates usually does. Search-response
caching and concurrent parsing/embedding are not implemented.

The website **Verify search results** checkbox defaults to on. Toggle it to rerun
the current query with or without relevance verification. It does not affect browsing.
The search API accepts a top-level `verify` boolean (default `true`).

### Selective free-text attribute filters

Search now enforces parsed explicit `fit`, `color_primary`, `color_secondary`,
and `lined` requirements as metadata filters, in both semantic and BM25 retrieval.
Other parsed attributes (including ambiguous garment subtypes) remain semantic.
Category, currency, price and selected site dropdowns remain mandatory.
The complete query is still embedded. Verification remains optional.

Inspect `response["attribute_filtering"]` for the exact/semantic attribute split,
and `response["retrieval_filter"]` for actual database conditions.
The allowlist is `EXACT_QUERY_ATTRIBUTES` in `search/pinecone_search.py`.
This is a conservative field policy, not a confidence detector: it still relies
on correct parsing, and missing product attributes fail exact requirements.
Re-run search cells to obtain fresh results; previously saved notebook outputs
reflect the old policy. Extend the policy only after reviewing evaluation cases.
