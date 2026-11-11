"""Local BM25 scoring over Pinecone source records and reciprocal rank fusion.

No DuckDB access or new embedding calls. Suitable for the current small catalog;
fetching the full catalog per request should be replaced by an index at scale.
"""
import math
import re
from collections import Counter


def tokens(text):
    return re.findall(r'[^\W_]+', text.casefold(), flags=re.UNICODE)


def matches_filter(product, clause):
    for field, condition in clause.items():
        if field == '$and':
            if not all(matches_filter(product, child) for child in condition): return False
        elif field == '$or':
            if not any(matches_filter(product, child) for child in condition): return False
        else:
            actual = product.get(field)
            for op, expected in condition.items():
                if op == '$eq':
                    if actual != expected: return False
                elif op == '$in':
                    values = actual if isinstance(actual,list) else [actual]
                    if not any(value in expected for value in values): return False
                elif op in ('$lt','$lte','$gt','$gte'):
                    if not isinstance(actual,(int,float)): return False
                    if not {'$lt':actual < expected,'$lte':actual <= expected,
                            '$gt':actual > expected,'$gte':actual >= expected}[op]: return False
                else: raise ValueError(f'Unsupported metadata operator: {op}')
    return True


def document_text(product):
    parts = [str(product.get(k,'')) for k in ['title','description','material','source_fit','source_color']]
    for key,value in sorted(product.items()):
        if key.startswith('attr_'):
            parts.append(' '.join(value) if isinstance(value,list) else str(value))
    return '\n'.join(parts)


def bm25_rank(query, products, where, limit=100, k1=1.5, b=0.75):
    """BM25 with positive Robertson IDF; corpus statistics use the full catalog."""
    if not products: return []
    docs=[Counter(tokens(document_text(p))) for p in products]
    lengths=[sum(doc.values()) for doc in docs]
    average=sum(lengths)/len(lengths) or 1
    terms=set(tokens(query))
    df={term:sum(term in doc for doc in docs) for term in terms}
    ranked=[]
    for product,doc,length in zip(products,docs,lengths):
        if not matches_filter(product,where): continue
        score=0.0
        for term in terms:
            frequency=doc[term]
            if frequency:
                idf=math.log(1+(len(docs)-df[term]+0.5)/(df[term]+0.5))
                score+=idf*frequency*(k1+1)/(frequency+k1*(1-b+b*length/average))
        if score>0:
            ranked.append({**product,'bm25_score':score})
    return sorted(ranked,key=lambda p:(-p['bm25_score'],p['product_id']))[:limit]


def fuse_rankings(dense, keyword, limit=100, rank_constant=60):
    merged={}
    for channel,ranking in [('semantic',dense),('bm25',keyword)]:
        seen=set()
        for rank,product in enumerate(ranking,1):
            pid=product['product_id']
            if pid in seen:continue
            seen.add(pid)
            if pid not in merged:
                merged[pid]={**product,'score':0.0,'retrieval_sources':[]}
            item=merged[pid]
            item['score']+=1/(rank_constant+rank)
            item['retrieval_sources'].append(channel)
            item[channel+'_rank']=rank
            if channel=='semantic':item['semantic_score']=product['score']
            else:item['bm25_score']=product['bm25_score']
    return sorted(merged.values(),key=lambda p:(-p['score'],p['product_id']))[:limit]


def fetch_catalog(index, ids, namespace):
    products=[]
    for start in range(0,len(ids),100):
        response=index.index.documents.fetch(namespace=namespace,ids=ids[start:start+100])
        for pid,doc in response.documents.items():
            data=doc.to_dict()
            for key in ['_id','_score','embedding']:data.pop(key,None)
            products.append({**data,'product_id':pid})
    return products
