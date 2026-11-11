"""Replit-ready storefront. Run: uv run uvicorn interfaces.web_api:app --host 0.0.0.0 --port 3000."""
from contextlib import asynccontextmanager
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from search.pinecone_search import ROOT, search, connect_index, namespace


@asynccontextmanager
async def lifespan(app):
    load_dotenv(ROOT / '.env')  # Replit Secrets/environment take precedence.
    yield


app = FastAPI(title='Thread catalog search', lifespan=lifespan)


class SiteFilters(BaseModel):
    model_config = ConfigDict(extra='forbid')
    brand: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)
    sleeve: list[str] = Field(default_factory=list)
    neck: list[str] = Field(default_factory=list)
    fit: list[str] = Field(default_factory=list)


def apply_site_filters(products, filters):
    retailers = json.loads((ROOT / 'data/product_retailers.json').read_text())
    mapping = {'color':'attr_color_primary', 'sleeve':'attr_sleeve_length',
               'neck':'attr_neckline', 'fit':'attr_fit'}
    matches=[]
    for product in products:
        product = dict(product, retailer=retailers.get(product['product_id'], ''))
        if filters.brand and product['retailer'] not in filters.brand:
            continue
        if any(values and product.get(mapping[key]) not in values
               for key, values in filters.model_dump().items() if key in mapping):
            continue
        matches.append(product)
    return matches


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=60, ge=1, le=60)
    filters: SiteFilters = Field(default_factory=SiteFilters)
    verify: bool = True


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/api/search')
async def search_catalog(request: SearchRequest):
    try:
        mapping = {'brand':'retailer', 'color':'attr_color_primary', 'sleeve':'attr_sleeve_length', 'neck':'attr_neckline', 'fit':'attr_fit'}
        clauses = [{mapping[key]:{'$in':values}} for key,values in request.filters.model_dump().items() if values]
        explicit = {'$and':clauses} if clauses else None
        result = await run_in_threadpool(search, request.query, 60, explicit, verify=request.verify)
        result['results'] = apply_site_filters(result['results'], request.filters)[:request.top_k]
        return result
    except ValueError as error:
        # Unsupported shopping requirements can be shown without exposing API internals.
        if str(error).startswith('Unsupported constraints:'):
            raise HTTPException(422, str(error)) from error
        raise HTTPException(422, 'Could not interpret this request. Try a simpler search.') from error
    except Exception as error:
        import logging
        logging.getLogger(__name__).exception('Catalog search failed')
        raise HTTPException(503, 'Search is unavailable. Check server configuration or try again shortly.') from error


@app.get('/')
def storefront():
    return FileResponse(ROOT / 'web/index.html')


def browse_products():
    ids = json.loads((ROOT / 'data/indexed_product_ids.json').read_text())
    index = connect_index()
    response = index.index.documents.fetch(namespace=namespace(), ids=ids)
    products = []
    for pid in ids:
        doc = response.documents.get(pid)
        if doc is None:
            continue
        product = doc.to_dict()
        for key in ['_id', '_score', 'embedding']:
            product.pop(key, None)
        products.append({'product_id': pid, **product})
    return {'results': apply_site_filters(products, SiteFilters())}


@app.get('/api/products')
async def all_products():
    try:
        return await run_in_threadpool(browse_products)
    except Exception as error:
        raise HTTPException(503, 'Unable to load the collection. Please retry.') from error


@app.get('/api/facets')
async def facets():
    products = (await run_in_threadpool(browse_products))['results']
    mapping={'brand':'retailer','color':'attr_color_primary','sleeve':'attr_sleeve_length','neck':'attr_neckline','fit':'attr_fit'}
    return {key:sorted({p[field] for p in products if isinstance(p.get(field),str) and p[field]!='unknown'}) for key,field in mapping.items()}
