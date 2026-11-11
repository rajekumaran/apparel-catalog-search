"""Import saved extractions into data/catalog_attributes.duckdb; no API calls.

Unknown attributes are omitted from product_attributes. A missing attribute on a
completed extraction means unknown. Multi values occupy separate rows.
Legacy results lack source hashes/timestamps; these remain NULL rather than guessed.
"""
import csv
import hashlib
import json
from pathlib import Path
import duckdb
from validation.attribute_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/catalog_attributes.duckdb'


def load():
    with (ROOT.parent / 'womens_apparel_1000.csv').open(encoding='utf-8-sig', newline='') as f:
        products = {r['Product ID']: r for r in csv.DictReader(f)}
    con = duckdb.connect(str(DB))
    con.execute('''CREATE TABLE IF NOT EXISTS products (
        product_id VARCHAR PRIMARY KEY, title VARCHAR, description VARCHAR,
        category VARCHAR, price DECIMAL(18,2), currency VARCHAR, source_data JSON)''')
    con.execute('''CREATE TABLE IF NOT EXISTS extractions (
        extraction_id VARCHAR PRIMARY KEY, product_id VARCHAR REFERENCES products(product_id),
        taxonomy_category VARCHAR, taxonomy_version VARCHAR, model VARCHAR,
        extracted_at TIMESTAMPTZ, source_hash VARCHAR, taxonomy_hash VARCHAR,
        result_file VARCHAR, status VARCHAR, raw_result JSON)''')
    con.execute('''CREATE TABLE IF NOT EXISTS product_attributes (
        extraction_id VARCHAR REFERENCES extractions(extraction_id),
        attribute_name VARCHAR, attribute_value VARCHAR,
        PRIMARY KEY (extraction_id, attribute_name, attribute_value))''')
    con.execute("""CREATE TABLE IF NOT EXISTS attribute_evidence (
        extraction_id VARCHAR REFERENCES extractions(extraction_id),
        attribute_name VARCHAR, attribute_value VARCHAR,
        source_field VARCHAR, quote VARCHAR,
        PRIMARY KEY(extraction_id, attribute_name, attribute_value, source_field, quote))""")
    con.execute('BEGIN')
    try:
        # Original catalog is copied here; the source CSV/database stays unchanged.
        for pid, r in products.items():
            con.execute('INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING',
                        [pid, r['Title'], r['Description'], r['Category'], r['Price'] or None, r['Currency'], json.dumps(r)])
        for category in ['tops', 'bottoms', 'dresses']:
            taxonomy = json.loads((ROOT / f'data/womens_{category}_taxonomy.json').read_text())
            for path in sorted((ROOT / f'data/extracted_{category}').glob('*.json')):
                result = json.loads(path.read_text())
                assert result['product_id'] in products
                assert products[result['product_id']]['Category'].casefold() == category
                assert result['taxonomy_version'] == taxonomy['version']
                assert set(result['attributes']) == set(taxonomy['attributes'])
                values = []
                for name, spec in taxonomy['attributes'].items():
                    value = result['attributes'][name]
                    assert isinstance(value, list) if spec['selection'] == 'multi' else isinstance(value, str)
                    for v in value if isinstance(value, list) else [value]:
                        assert v in spec['values'], (path, name, v)
                        if v != 'unknown':
                            values.append((name, v))
                evidence = result.get('evidence', {})
                if result.get('extraction_format') == 'evidence-v1':
                    validate_evidence(result['attributes'], evidence, result['source_snapshot'], taxonomy)
                # Deterministic result identity makes repeated imports idempotent.
                eid = hashlib.sha256(json.dumps([category, result], sort_keys=True).encode()).hexdigest()
                if con.execute('SELECT 1 FROM extractions WHERE extraction_id=?', [eid]).fetchone():
                    continue
                con.execute('INSERT INTO extractions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                            [eid, result['product_id'], taxonomy['category'], result['taxonomy_version'],
                             result.get('model'), result.get('extracted_at'), result.get('source_hash'),
                             result.get('taxonomy_hash'), str(path.relative_to(ROOT)), 'completed', json.dumps(result)])
                con.executemany('INSERT INTO product_attributes VALUES (?, ?, ?)',
                                [(eid, name, v) for name, v in sorted(set(values))]) if values else None
                evidence_rows = [(eid, name, item['value'], item['source_field'], item['quote'])
                                 for name, entries in evidence.items() for item in entries]
                if evidence_rows:
                    con.executemany('INSERT INTO attribute_evidence VALUES (?, ?, ?, ?, ?)', evidence_rows)
        con.execute('COMMIT')
    except Exception:
        con.execute('ROLLBACK')
        raise
    print(f'Database: {DB}')
    for table in ['products', 'extractions', 'product_attributes']:
        print(table, con.execute(f'SELECT count(*) FROM {table}').fetchone()[0])
    print(con.execute('SELECT taxonomy_category, count(*) FROM extractions GROUP BY 1 ORDER BY 1').fetchall())
    con.close()


if __name__ == '__main__':
    load()
