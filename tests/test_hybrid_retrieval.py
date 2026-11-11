import unittest
from search.hybrid_retrieval import bm25_rank, fuse_rankings, matches_filter


class HybridTests(unittest.TestCase):
    def test_keywords_find_distinctive_name_and_respect_price(self):
        # Search three sample products by the distinctive keyword Quiny with a price cap of 80.
        # Only product 1 matches both requirements; product 3 is too expensive.
        # A keyword absent from every document should produce no BM25 matches.
        products=[{'product_id':'1','title':'Quiny unlined shirt','price':50},
                  {'product_id':'2','title':'Unlined blouse','price':20},
                  {'product_id':'3','title':'Quiny shirt','price':100}]
        ranked=bm25_rank('Quiny',products,{'price':{'$lt':80}})
        self.assertEqual([p['product_id'] for p in ranked],['1'])
        self.assertEqual(bm25_rank('nonexistent',products,{}),[])

    def test_fusion_keeps_both_channels_without_adding_incompatible_scores(self):
        # Provide overlapping semantic and BM25 rankings with different score scales.
        # Reciprocal rank fusion should put product 2 first because it appears in both lists.
        # Keep all three unique products, record both sources, and preserve the original semantic score.
        dense=[{'product_id':'1','score':0.9},{'product_id':'2','score':0.8}]
        keyword=[{'product_id':'2','bm25_score':8},{'product_id':'3','bm25_score':7}]
        result=fuse_rankings(dense,keyword)
        self.assertEqual(result[0]['product_id'],'2')
        self.assertEqual(result[0]['retrieval_sources'],['semantic','bm25'])
        self.assertEqual(len(result),3)
        self.assertEqual(result[0]['semantic_score'],0.8)

    def test_compound_and_list_filters(self):
        # Check that a product satisfies every condition in an AND filter.
        # The fit must belong to the allowed-value list; a product missing that field must fail.
        self.assertTrue(matches_filter({'category':'Tops','attr_fit':'relaxed'},
            {'$and':[{'category':{'$eq':'Tops'}},{'attr_fit':{'$in':['relaxed']}}]}))
        self.assertFalse(matches_filter({}, {'attr_fit':{'$in':['relaxed']}}))

if __name__=='__main__':unittest.main()
