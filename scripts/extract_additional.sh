#!/usr/bin/env bash
# New batches: ./scripts/extract_additional.sh [per-category count] [--sync] [--dry-run]
# Resume:     ./scripts/extract_additional.sh --resume data/batches/<batch> [--sync]
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Missing environment. Run uv sync --locked from catalog_search first.' >&2
  exit 1
fi
COUNT=20
SYNC=false
DRY=false
BATCH_DIR=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync) SYNC=true; shift ;;
    --dry-run) DRY=true; shift ;;
    --resume)
      [[ $# -ge 2 && -d "$2" ]] || { echo 'Provide an existing batch directory after --resume.' >&2; exit 1; }
      BATCH_DIR="$2"; shift 2 ;;
    --help|-h) sed -n '2,3p' "$0"; exit 0 ;;
    *)
      [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid argument: $1" >&2; exit 1; }
      COUNT="$1"; shift ;;
  esac
done
if [[ -z "$BATCH_DIR" ]]; then
  BATCH_DIR="data/batches/$(date +%Y%m%d-%H%M%S)-$$"
fi
if ! "$DRY"; then
  mkdir -p "$BATCH_DIR"
  echo "Batch: $BATCH_DIR"
  echo "Resume after failure: bash scripts/extract_additional.sh --resume '$BATCH_DIR'"
fi
for CATEGORY in tops bottoms dresses; do
  OPTIONS=("$CATEGORY" --new-count "$COUNT" --plan "$BATCH_DIR/$CATEGORY.json")
  if "$DRY"; then OPTIONS+=(--dry-run); fi
  "$PYTHON" -m extraction.extract_catalog "${OPTIONS[@]}"
done
if "$DRY"; then exit 0; fi
"$PYTHON" -m storage.load_attribute_tables
if "$SYNC"; then
  bash scripts/sync_catalog.sh
else
  echo 'Saved attributes and evidence in JSON and DuckDB. To publish: bash scripts/sync_catalog.sh'
fi
