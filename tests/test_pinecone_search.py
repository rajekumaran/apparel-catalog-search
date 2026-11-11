import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import duckdb
from pydantic import ValidationError
import search.pinecone_search as service


class SearchTests(unittest.TestCase):
    def test_filters_and_cross_category(self):
        # Build a validated intent directly; this test does not call the LLM.
        # Check that price and rise become Pinecone filter conditions.
        # Also check that category is optional and 'unknown' is not a valid fit request.
        intent = service.ShopperIntent(category='Bottoms', price_under=80,
            attributes=[{'name':'rise','values':['high']}])
        clauses=service.pinecone_filter(intent)['$and']
        self.assertIn({'price':{'$lt':80}},clauses)
        self.assertIn({'attr_rise':{'$in':['high']}},clauses)
        self.assertIsNone(service.ShopperIntent().category)
        with self.assertRaises(ValidationError):
            service.ShopperIntent(category='Tops',attributes=[{'name':'fit','values':['unknown']}])

    def test_draped_dress_uses_semantics_and_keeps_price(self):
        # Simulate parser output containing 'draped', which is not an allowed silhouette.
        # Keep that meaning as a semantic fallback instead of an invalid exact filter.
        # The valid color attribute and the strict price limit must remain.
        intent = service.ShopperIntent.model_validate({
            'category':'Dresses', 'price_under':100,
            'attributes':[{'name':'silhouette','values':['draped']},
                          {'name':'color_primary','values':['black']}],
            'semantic_query':'draped black dress under $100'})
        self.assertEqual([a.name for a in intent.attributes], ['color_primary'])
        self.assertEqual(len(intent.semantic_fallbacks),1)
        clauses=service.pinecone_filter(intent)['$and']
        self.assertIn({'price':{'$lt':100}},clauses)
        self.assertFalse(any('attr_silhouette' in clause for clause in clauses))

    def test_out_of_scope_never_retrieves_or_embeds(self):
        # Pretend the LLM has classified the request as unsupported.
        # Replace API clients with mocks: no real LLM or Pinecone calls are made.
        # Expect clarification and no products, and ensure embedding/retrieval never starts.
        # The fake model name checks metadata reporting, not the configured production model.
        from unittest.mock import MagicMock
        parser = MagicMock()
        parser.invoke.return_value = {'parsed': service.ShopperIntent(catalog_scope='unsupported'), 'parsing_error': None, 'raw': SimpleNamespace(response_metadata={'model_name':'gpt-5.4-test'}, usage_metadata={'input_tokens':10})}
        with patch.object(service, 'ChatOpenAI') as chat, patch.object(service, 'embed') as embed, patch.object(service, 'connect_index') as connect:
            chat.return_value.with_structured_output.return_value = parser
            result = service.search('wedding shoes')
            self.assertEqual(result['results'], [])
            self.assertEqual(result['parser_info']['reported_model'], 'gpt-5.4-test')
            self.assertIn('only tops', result['clarification'])
            embed.assert_not_called()
            connect.assert_not_called()

    def test_verifier_excludes_unknown_and_fabricated_evidence(self):
        # Give the verifier two products and a mocked 'match' decision for each.
        # Only the first supporting quote actually occurs in its product description.
        # Accept that product; downgrade the fabricated quote to insufficient evidence.
        # This checks quote existence, not whether every valid quote proves relevance.
        from unittest.mock import MagicMock
        candidates=[{'product_id':'1','title':'Cotton tee','description':'Unlined tee'},
                    {'product_id':'2','title':'Other','description':'Lined shirt'}]
        verdicts=service.RelevanceReview(verdicts=[
            {'product_id':'1','decision':'match','reason':'Unlined tee','source_field':'description','quote':'Unlined tee'},
            {'product_id':'2','decision':'match','reason':'Incorrect claim','source_field':'description','quote':'Unlined'}])
        raw=SimpleNamespace(response_metadata={'model_name':'test'},usage_metadata={})
        parser=MagicMock();parser.invoke.return_value={'parsed':verdicts,'parsing_error':None,'raw':raw}
        with patch.object(service,'ChatOpenAI') as chat:
            chat.return_value.with_structured_output.return_value=parser
            accepted,audit,info=service.verify_candidates('unlined shirt',candidates)
        self.assertEqual([p['product_id'] for p in accepted],['1'])
        self.assertEqual(audit[1]['decision'],'insufficient_evidence')
        self.assertEqual(audit[1]['title'],'Other')

    def test_verifier_receives_price_without_requiring_a_price_quote(self):
        # A catalog price is structured evidence; the garment quote stays textual.
        from unittest.mock import MagicMock
        products = [{'product_id':'dress', 'title':'Blue mini dress',
                     'description':'Cotton dress', 'price':79.0, 'currency':'USD'}]
        review = service.RelevanceReview(verdicts=[{
            'product_id':'dress', 'decision':'match',
            'reason':'Dress costs USD 79, below USD 100.',
            'source_field':'title', 'quote':'Blue mini dress'}])
        parser = MagicMock()
        parser.invoke.return_value = {'parsed':review, 'parsing_error':None,
            'raw':SimpleNamespace(response_metadata={'model_name':'test'}, usage_metadata={})}
        with patch.object(service, 'ChatOpenAI') as chat:
            chat.return_value.with_structured_output.return_value = parser
            accepted, _, _ = service.verify_candidates('dresses under $100', products)
        messages = parser.invoke.call_args.args[0]
        payload = json.loads(messages[1][1])['products'][0]
        self.assertEqual(payload['price'], 79.0)
        self.assertEqual(payload['currency'], 'USD')
        self.assertNotIn('79', payload['title'])
        self.assertEqual(accepted[0]['product_id'], 'dress')
        self.assertIn('strictly less than', messages[0][1])

    def test_semantic_filter_does_not_enforce_free_text_attributes(self):
        # Build an intent that includes a specific shirt type and a price cap.
        # With include_attributes=False, omit the shirt-type metadata restriction.
        # Keep the price cap so semantic matching cannot bypass the budget.
        intent=service.ShopperIntent(category='Tops',price_under=80,attributes=[{'name':'top_type','values':['button_down_shirt']}])
        clauses=service.pinecone_filter(intent,include_attributes=False)['$and']
        self.assertFalse(any('attr_top_type' in c for c in clauses))
        self.assertIn({'price':{'$lt':80}},clauses)

    def test_selective_attributes_enforce_requirements_without_narrowing_shirt_type(self):
        # Explicit fit/color/lined requirements restrict both retrieval branches.
        # A potentially ambiguous garment subtype stays semantic.
        intent = service.ShopperIntent(category='Tops', price_under=80, attributes=[
            {'name':'top_type','values':['button_down_shirt']},
            {'name':'fit','values':['relaxed']},
            {'name':'color_primary','values':['black']},
            {'name':'lined','values':['no']}])
        where, policy = service.selective_attribute_filter(intent)
        from search.hybrid_retrieval import matches_filter
        product = {'embedding_model':service.MODEL, 'currency':'USD',
                   'category':'Tops', 'price':50, 'attr_fit':'relaxed',
                   'attr_color_primary':'black', 'attr_lined':'no'}
        self.assertTrue(matches_filter(product, where))
        self.assertFalse(matches_filter(dict(product, attr_fit='fitted'), where))
        self.assertFalse(matches_filter(dict(product, attr_lined='yes'), where))
        self.assertFalse(matches_filter(dict(product, price=90), where))
        missing = dict(product)
        missing.pop('attr_fit')
        self.assertFalse(matches_filter(missing, where))
        self.assertEqual([a['name'] for a in policy['semantic_attributes']], ['top_type'])
        self.assertEqual(len(policy['exact_attributes']), 3)

    def test_category_ambiguity_does_not_block_supported_garment(self):
        # A known category can resolve an otherwise unclear scope as supported.
        # Explicit unsupported/mixed decisions must not be overwritten.
        # Without a category, an unclear scope must stay unclear.
        intent=service.ShopperIntent(category='Tops',catalog_scope='unclear',
            attributes=[{'name':'lined','values':['no']}])
        self.assertEqual(intent.catalog_scope,'supported')
        self.assertIsNotNone(intent.scope_resolution)
        for scope in ['unsupported','mixed']:
            self.assertEqual(service.ShopperIntent(category='Dresses',catalog_scope=scope).catalog_scope,scope)
        self.assertEqual(service.ShopperIntent(catalog_scope='unclear').catalog_scope,'unclear')

    def test_review_missing_duplicate_and_unknown_ids_fail_closed(self):
        # Simulate a malformed review: ID 0 is repeated, ID 2 is missing, and one ID is invented.
        # Only ID 1 has a single valid review, so only that product may be accepted.
        # Keep an audit entry for each real candidate and report the review problems.
        from unittest.mock import MagicMock
        candidates=[{'product_id':str(i),'title':str(i),'description':'unlined'} for i in range(3)]
        raw=SimpleNamespace(response_metadata={'model_name':'test'},usage_metadata={'input_tokens':10})
        entries=[{'product_id':pid,'decision':'match','reason':'test','source_field':'description','quote':'unlined'} for pid in ['0','0','1','invented']]
        parser=MagicMock();parser.invoke.return_value={'parsed':service.RelevanceReview(verdicts=entries),'parsing_error':None,'raw':raw}
        with patch.object(service,'ChatOpenAI') as chat:
            chat.return_value.with_structured_output.return_value=parser
            accepted,audit,info=service.verify_candidates('unlined',candidates)
        self.assertEqual([p['product_id'] for p in accepted],['1'])
        self.assertEqual(len(audit),3)
        self.assertEqual(audit[0]['decision'],'insufficient_evidence')
        self.assertEqual(audit[2]['decision'],'insufficient_evidence')
        self.assertEqual(len(info['review_warnings']),3)

    def test_review_batches_and_preserves_order(self):
        # Use 23 fake candidates to exercise splitting into three batches of at most 10.
        # The local batch stub accepts every product and reports 10 input tokens per batch.
        # Check that output order is preserved and token usage totals 30.
        candidates=[{'product_id':str(i)} for i in range(23)]
        def batch(query,products):
            # Stand in for one verifier API call using controlled local behavior.
            self.assertLessEqual(len(products),10)
            return products,[],{'reported_model':'test','token_usage':{'input_tokens':10}}
        with patch.object(service,'_verify_candidate_batch',side_effect=batch):
            accepted,audit,info=service.verify_candidates('query',candidates)
        self.assertEqual(accepted,candidates)
        self.assertEqual(len(info['batches']),3)
        self.assertEqual(info['token_usage']['input_tokens'],30)

    def test_corpus_cache_hit_expiry_and_id_invalidation(self):
        # Start with an empty cache and mock catalog fetching and elapsed time.
        # The first call fetches; the next reuses the cache; time 400 triggers expiry.
        # Changing the product IDs must also fetch again, even with a recent cache.
        # Clear the cache afterwards so this test does not affect other tests.
        service.clear_search_cache()
        with patch.object(service,'fetch_catalog',return_value=[{'product_id':'1'}]) as fetch:
            with patch.object(service,'monotonic',side_effect=[0,1,400,400]):
                self.assertFalse(service.cached_catalog(None,['1'],'test')[1])
                self.assertTrue(service.cached_catalog(None,['1'],'test')[1])
                self.assertFalse(service.cached_catalog(None,['1'],'test')[1])
            self.assertFalse(service.cached_catalog(None,['2'],'test')[1])
            self.assertEqual(fetch.call_count,3)
        service.clear_search_cache()

    def test_two_review_batches_can_run_concurrently(self):
        # Split 20 candidates into two batches and make both wait at a threading barrier.
        # Both batches must reach the barrier together; sequential execution would time out.
        # Check that concurrent work still returns products in their original order.
        from threading import Barrier
        barrier=Barrier(2)
        candidates=[{'product_id':str(i)} for i in range(20)]
        def batch(query,products):
            # Stand in for one verifier API call using controlled local behavior.
            barrier.wait(timeout=3)
            return products,[],{'reported_model':'test','token_usage':{}}
        with patch.object(service,'_verify_candidate_batch',side_effect=batch):
            accepted,_,_=service.verify_candidates('query',candidates)
        self.assertEqual(accepted,candidates)

    def test_search_can_bypass_verification(self):
        # Mock parsing, embedding, Pinecone, and corpus fetching to isolate search control flow.
        # Supply one retrieved product and call search with verify=False.
        # Ensure the verifier is never called, the product is returned, and diagnostics show verification was skipped.
        from unittest.mock import MagicMock
        intent=service.ShopperIntent(category='Tops',catalog_scope='supported')
        raw=SimpleNamespace(response_metadata={'model_name':'test'},usage_metadata={})
        parser=MagicMock();parser.invoke.return_value={'parsed':intent,'parsing_error':None,'raw':raw}
        index=MagicMock();index.query.return_value=SimpleNamespace(matches=[
            SimpleNamespace(id='1',score=0.9,metadata={'title':'Example','category':'Tops'})])
        with patch.object(service,'ChatOpenAI') as chat, patch.object(service,'OpenAI'), patch.object(service,'embed',return_value=[0.1]), patch.object(service,'connect_index',return_value=index), patch.object(service,'cached_catalog',return_value=([],True)), patch.object(service,'verify_candidates') as verifier:
            chat.return_value.with_structured_output.return_value=parser
            result=service.search('shirt',top_k=1,verify=False)
        verifier.assert_not_called()
        self.assertEqual(result['results'][0]['product_id'],'1')
        self.assertFalse(result['verification_enabled'])
        self.assertIsNone(result['verifier_info'])
        self.assertEqual(result['verification'],[])
        self.assertEqual(result['retrieval_info']['verified_candidates'],0)

    def test_sync_recovery_and_cache(self):
        # Create a temporary DuckDB database with one product and its saved extraction.
        # Mock embedding and Pinecone upload, so no external API calls are made.
        # Force the first upload to fail, then retry and confirm the cached vector is reused.
        # An unchanged sync is skipped; an explicit reupload also reuses the embedding.
        # Finally check sync tracking, extraction linkage, and exclusion of unknown attributes.
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'test.duckdb'
            # Tiny fixture for the actual record-building query.
            with duckdb.connect(str(db)) as con:
                con.execute('CREATE TABLE products(product_id VARCHAR,title VARCHAR,description VARCHAR,category VARCHAR,price DOUBLE,currency VARCHAR,source_data JSON)')
                con.execute("INSERT INTO products VALUES ('1','Blue jeans','Denim','Bottoms',50,'USD','{}')")
                con.execute('CREATE TABLE extractions(product_id VARCHAR,extracted_at TIMESTAMP,extraction_id VARCHAR,status VARCHAR,raw_result JSON)')
                con.execute("INSERT INTO extractions VALUES ('1',NULL,'a','completed',?)",[json.dumps({'attributes':{'rise':'high','fit':'unknown'}})])
            index=SimpleNamespace(upsert=lambda **kw: SimpleNamespace(upserted_count=1))
            with patch.object(service,'DB',db), patch.object(service,'connect_index',return_value=index), patch.object(service,'OpenAI'), patch.object(service,'embed',return_value=[0.1]*1536) as embed, patch.dict('os.environ',{'PINECONE_INDEX_HOST':'test-host'}):
                with patch.object(index,'upsert',side_effect=RuntimeError('offline')):
                    with self.assertRaises(RuntimeError): service.sync()
                service.sync()  # Upload retry reuses the saved vector.
                service.sync()  # Unchanged product is skipped.
                service.sync(reupload=True)  # Restore remote record without re-embedding.
                self.assertEqual(embed.call_count,1)
                with duckdb.connect(str(db)) as con:
                    self.assertEqual(con.execute('SELECT count(*) FROM pinecone_sync').fetchone()[0],1)
                    _,text,meta=next(service.records(con))
                    self.assertEqual(meta['extraction_id'], 'a')
                    self.assertNotIn('attr_fit',meta)
                    self.assertNotIn('unknown',text)

if __name__=='__main__': unittest.main()
