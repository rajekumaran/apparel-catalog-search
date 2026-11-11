import sys
from pathlib import Path
import unittest
from evals.run_eval import assess


class EvalTests(unittest.TestCase):
    def test_missing_positive_stage_and_exclusion(self):
        # Define one required product and one explicitly forbidden product.
        # Simulate a response that returns the forbidden product and omits the required one.
        # The evaluator must report failure and zero recall for the known positive.
        # Because the wanted ID appears in the review audit, mark its loss as verification or top-k.
        case={'expected_ids':['wanted'],'excluded_ids':['wrong']}
        r={'results':[{'product_id':'wrong','match_evidence':{'quote':'x'}}],
           'verification':[{'product_id':'wanted'}]}
        result=assess(case,r)
        self.assertEqual(result['status'],'fail')
        self.assertEqual(result['known_positive_recall'],0)
        self.assertEqual(result['missing_positive_stages']['wanted'],'verification_or_top_k')

    def test_required_attributes_reject_shorts_and_missing_metadata(self):
        # Matching color and price cannot make shorts satisfy a jeans requirement.
        case = {'required_attributes': {'bottom_type': ['jeans'], 'color_primary': ['blue']}}
        for bottom_type, expected in [('shorts', 'fail'), (None, 'fail'), ('jeans', 'pass')]:
            with self.subTest(bottom_type=bottom_type):
                response = {'verification_enabled': False, 'results': [{
                    'product_id': 'sample', 'attr_bottom_type': bottom_type,
                    'attr_color_primary': 'blue'}]}
                self.assertEqual(assess(case, response)['status'], expected)

    def test_unsupported_requires_clarification_and_no_matches(self):
        # Define an unsupported-query case requiring clarification and no results.
        # A response with unsupported scope, clarification, and an empty result list should pass.
        case={'max_results':0,'clarification_required':True,'scope':'unsupported'}
        r={'filters':{'catalog_scope':'unsupported'},'clarification':'Not sold','results':[]}
        self.assertEqual(assess(case,r)['status'],'pass')

if __name__=='__main__':unittest.main()
