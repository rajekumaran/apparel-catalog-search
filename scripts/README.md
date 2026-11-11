# Catalog batch scripts

Run these commands from `week1_project/catalog_search`:

```bash
cd /Users/sundar/Raje/agentic_ai/week1_project/catalog_search
```

## Which script should I run?

| What you want | Command |
|---|---|
| Preview the next 60 products without API calls | `bash scripts/extract_additional.sh 20 --dry-run` |
| Extract 60 new products and save attributes/evidence locally | `bash scripts/extract_additional.sh 20` |
| Extract 60 new products and publish them to Pinecone | `bash scripts/extract_additional.sh 20 --sync` |
| Publish results already extracted locally | `bash scripts/sync_catalog.sh` |
| Resume an interrupted batch | `bash scripts/extract_additional.sh --resume data/batches/YOUR-BATCH --sync` |

## 1. Preview first

```bash
bash scripts/extract_additional.sh 20 --dry-run
```

`20` means **20 products per category**: 20 tops, 20 bottoms, and 20 dresses.
The script prints their IDs and titles. No API calls, extraction, or batch-file
writes occur. To preview a smaller batch, replace `20` with `1` (3 total).

## 2. Extract and save locally

```bash
bash scripts/extract_additional.sh 20
```

This selects products not already extracted in JSON or DuckDB, extracts
attributes with per-value source evidence, saves JSON under
`data/extracted_tops/`, `data/extracted_bottoms/`, and
`data/extracted_dresses/`, and imports the results into DuckDB.

It does **not** publish to Pinecone. Extraction uses your OpenAI account and
incurs usage charges. Existing saved products are reused, not re-extracted.
If a category has fewer new products remaining, it processes those remaining.

## 3. Extract and publish in one run

```bash
bash scripts/extract_additional.sh 20 --sync
```

After extraction and database loading, this creates any missing embeddings,
uploads product records to Pinecone, and refreshes the website's browse-ID and
retailer files. Cached embeddings are reused. New embeddings incur API charges.

## 4. Publish an already extracted batch

```bash
bash scripts/sync_catalog.sh
```

Use this if you extracted without `--sync`, or need to retry publishing.
It imports saved JSON, syncs Pinecone, and refreshes browse manifests.
It does **not** call the attribute-extraction LLM.

`refresh_manifest.py` is an internal helper called by this script; normally you
should not run it separately. It includes products whose current upload has
been acknowledged for the configured Pinecone target.

## 5. Resume after a failure

The extraction script prints a directory such as:

```text
data/batches/20261101-143000-12345
```

Use the **actual directory printed by your run**:

```bash
bash scripts/extract_additional.sh --resume data/batches/20261101-143000-12345 --sync
```

Saved category plans preserve the selected source rows. Completed products are
skipped and unfinished ones are retried. Keep the batch directory until the run
is complete. Without `--resume`, a new invocation intentionally selects a fresh
batch of unprocessed products.

## Before running

- Create the project environment once with `uv sync --locked`.
- Configure OpenAI credentials in `catalog_search/.env` for extraction.
- Configure OpenAI and Pinecone credentials, index name/host, and namespace for publishing.
- Run only one extraction/publishing process at a time against this folder.
- Failures stop the script, but successfully saved products remain on disk.

## After publishing

Refresh the website. Restart its server if it is not using automatic reload.
For Replit, also upload the updated `data/indexed_product_ids.json` and
`data/product_retailers.json` files.

The current website still has copy and search-result limits based on the original
60-product prototype. Publishing more products updates browse manifests, but
those interface limits need a separate update before larger-catalog deployment.
