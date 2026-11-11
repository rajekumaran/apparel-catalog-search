"""Refresh browse IDs only for current records acknowledged by Pinecone sync."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from search.pinecone_search import ROOT, DB, records, digest, MODEL, DIMENSIONS, namespace
import duckdb
import os
load_dotenv(ROOT / '.env')
# A previous sync resolves the host in its own process; resolve it here if needed.
from search.pinecone_search import connect_index
connect_index()
target = os.environ['PINECONE_INDEX_HOST'] + '/' + namespace()
with duckdb.connect(str(DB), read_only=True) as con:
    ids = []
    for pid, text, metadata in records(con):
        payload = digest([digest([MODEL, DIMENSIONS, text]), metadata])
        if con.execute('SELECT 1 FROM pinecone_sync WHERE target=? AND product_id=? AND payload_hash=?', [target, pid, payload]).fetchone():
            ids.append(pid)
    sources = {pid: json.loads(source) for pid, source in con.execute('SELECT product_id, source_data FROM products').fetchall()}
for filename, value in [('indexed_product_ids.json', ids), ('product_retailers.json', {pid:sources[pid].get('Retailer','') for pid in ids})]:
    path = ROOT / 'data' / filename
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)
print(f'Updated website browse manifests: {len(ids)} products.')
