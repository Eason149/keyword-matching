"""Contract tests for team run evaluation; run with python test_baseline.py."""
import json,tempfile,unittest
from pathlib import Path
from baseline import parse_run,evaluate

class ContractTests(unittest.TestCase):
    def test_formats_and_invalid_runs(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'run.jsonl'
            def parse(rows):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                return parse_run(path,{'q'},{'a','b'})
            self.assertEqual(parse([{'query_id':'q','results':[{'dataset_id':'b','score':1},{'dataset_id':'a','score':2}]}]),{'q':['b','a']})
            self.assertEqual(parse([{'query_id':'q','dataset_id':'a','rank':2,'score':1},{'query_id':'q','dataset_id':'b','rank':1,'score':2}]),{'q':['b','a']})
            for rows in [
                [{'query_id':'unknown','results':[]}],
                [{'query_id':'q','results':[{'dataset_id':'unknown','score':1}]}],
                [{'query_id':'q','results':[{'dataset_id':'a','score':1}]*2}],
                [{'query_id':'q','dataset_id':'a','rank':2,'score':1}],
                [{'query_id':'q','results':[{'dataset_id':'a','score':float('nan')}]}],
                [{'query_id':'q','results':[]},{'query_id':'q','results':[]}],
            ]:
                with self.assertRaises(ValueError):parse(rows)
            self.assertEqual(parse([]),{})
    def test_missing_query_and_unreachable_positive(self):
        with tempfile.TemporaryDirectory() as t:
            rows=evaluate({'q':['a']},[{'query_id':'q','split':'test'},{'query_id':'r','split':'test'}],{'q':{'a':1,'missing':1},'r':{'b':1}},Path(t))
            test=next(r for r in rows if r['split']=='test')
            self.assertEqual(test['n'],2);self.assertEqual(test['recall200'],.25)

if __name__=='__main__':unittest.main(verbosity=2)
