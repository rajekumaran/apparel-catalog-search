#!/usr/bin/env bash
# Publish saved extractions to Pinecone; extraction LLM is never called here.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
"$PYTHON" -m storage.load_attribute_tables
"$PYTHON" -m search.pinecone_search sync
"$PYTHON" scripts/refresh_manifest.py
