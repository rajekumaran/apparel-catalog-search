import unittest
from validation.attribute_evidence import extraction_schema, unpack_evidence, sanitize_evidence, parse_evidence_response


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        # Run before each test to create a small taxonomy and a known source description.
        # Build the extraction schema from that taxonomy so tests use its allowed values.
        self.taxonomy={'attributes':{
            'neckline':{'selection':'single','values':['v_neck','unknown']},
            'details':{'selection':'multi','values':['pockets','bow','none','unknown']}}}
        self.schema=extraction_schema(self.taxonomy)
        self.source={'Description':'V neck with pockets and a bow'}

    def test_values_and_missing_defaults(self):
        # Supply one valid neckline claim with an exact supporting quote.
        # Check that it is saved with evidence and omitted multi-value details default to unknown.
        result=self.schema.model_validate({'neckline':[{'value':'v_neck','source_field':'Description','quote':'V neck'}]})
        attrs,evidence=unpack_evidence(result,self.source,self.taxonomy)
        self.assertEqual(attrs,{'neckline':'v_neck','details':['unknown']})
        self.assertEqual(evidence['neckline'][0]['quote'],'V neck')

    def test_fabricated_quote_rejected(self):
        # Supply a quote that does not occur in the source description.
        # The strict unpacking function must raise ValueError rather than accept invented evidence.
        result=self.schema.model_validate({'neckline':[{'value':'v_neck','source_field':'Description','quote':'invented'}]})
        with self.assertRaises(ValueError):unpack_evidence(result,self.source,self.taxonomy)

    def test_invalid_quote_dropped_but_valid_attributes_preserved(self):
        # Point the neckline claim at a missing Title field, but provide valid pockets evidence.
        # The sanitizing path should drop only the bad claim, preserve pockets, and record a warning.
        result=self.schema.model_validate({
            'neckline':[{'value':'v_neck','source_field':'Title','quote':'V neck'}],
            'details':[{'value':'pockets','source_field':'Description','quote':'pockets'}]})
        attrs,evidence,warnings=sanitize_evidence(result,self.source,self.taxonomy)
        self.assertEqual(attrs['neckline'],'unknown')
        self.assertNotIn('neckline',evidence)
        self.assertEqual(attrs['details'],['pockets'])
        self.assertEqual(warnings[0]['attribute'],'neckline')

    def test_invalid_enum_rejected_without_losing_good_claim(self):
        # Simulate an LLM response that failed schema parsing because relaxed is not a neckline.
        # Recover the raw claims, reject the invalid value, and retain the valid pockets claim.
        # Check that the warning records which value was rejected.
        from types import SimpleNamespace
        raw={'neckline':[{'value':'relaxed','source_field':'Description','quote':'V neck'}],
             'details':[{'value':'pockets','source_field':'Description','quote':'pockets'}]}
        response={'parsed':None,'raw':SimpleNamespace(tool_calls=[{'args':raw}]),'parsing_error':ValueError('invalid enum')}
        attrs,evidence,warnings=parse_evidence_response(response,self.schema,self.source,self.taxonomy)
        self.assertEqual(attrs['neckline'],'unknown')
        self.assertEqual(attrs['details'],['pockets'])
        self.assertEqual(warnings[0]['rejected_evidence']['value'],'relaxed')

    def test_each_multi_value_has_quote(self):
        # Supply two details, each with its own exact source quote.
        # Check that both values survive and both evidence entries are saved.
        entries=[{'value':v,'source_field':'Description','quote':q} for v,q in [('pockets','pockets'),('bow','bow')]]
        attrs,evidence=unpack_evidence(self.schema.model_validate({'details':entries}),self.source,self.taxonomy)
        self.assertEqual(attrs['details'],['pockets','bow'])
        self.assertEqual(len(evidence['details']),2)

if __name__=='__main__':unittest.main()
