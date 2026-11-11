"""Render search results with embedded thumbnails inside the results table."""
import base64
from functools import lru_cache
from html import escape
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
from IPython.display import HTML, display


@lru_cache(maxsize=128)
def thumbnail(url):
    if not isinstance(url, str) or urlparse(url).scheme not in {'https', 'http'}:
        return 'No image'
    try:
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=15) as response:
            mime = response.headers.get_content_type()
            if mime not in {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}:
                raise ValueError('Unsupported image type')
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError('Image exceeds 10 MB')
        encoded = base64.b64encode(data).decode('ascii')
        return f'<img src="data:{mime};base64,{encoded}" alt="Product" style="width:100px;height:120px;object-fit:contain">'
    except Exception as error:
        return f'<a href="{escape(url, quote=True)}">Image unavailable ({escape(type(error).__name__)})</a>'


def results_html(products):
    table = pd.DataFrame(products)
    if table.empty:
        return '<p>No matches among the indexed products.</p>'
    columns = [c for c in ['product_id', 'title', 'score', 'price', 'currency', 'category', 'product_url'] if c in table]
    columns += sorted(c for c in table if c.startswith('attr_'))
    # Escape every catalog value before inserting our controlled image HTML.
    safe = table[columns].map(lambda value: escape(str(value)) if value is not None else '')
    safe.insert(0, 'image', [thumbnail(p.get('image_url')) for p in products])
    return safe.to_html(index=False, escape=False)


def display_search_results(products):
    display(HTML(results_html(products)))
