"""Offline integration checks for live discovery, partial failures and replay."""
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from live_search import capture, load_snapshot, rank, discovery_terms

class LiveTests(unittest.TestCase):
    def test_capture_and_replay(self):
        calls=[]
        def getter(url, timeout):
            calls.append(url)
            if '/api/' in url:
                return json.dumps([{'id':'org/medical','sha':'a'*40,'description':'medical question answering','tags':['medical']},
                                   {'id':'org/other','sha':'b'*40,'description':'medical images','tags':[]}]).encode(),{}
            if 'org/other' in url: raise HTTPError(url,404,'missing',{},None)
            return b'# Medical\nQuestion answering documentation',{}
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)/'snapshot'
            rows,info=capture('medical question',2,folder,getter=getter)
            self.assertEqual(len(rows),2);self.assertEqual(info['card_failures'],1)
            self.assertTrue(any('/resolve/'+('a'*40)+'/' in url for url in calls))
            restored,metadata=load_snapshot(folder)
            self.assertEqual(rank(rows,'medical question','bm25_document',5),rank(restored,'medical question','bm25_document',5))
            self.assertEqual(rank(rows,'medical question','keyword',5)[0]['score'],2)
            self.assertEqual(len(rank(rows,'absent','keyword',5)),0)
            (folder/'corpus.jsonl').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):load_snapshot(folder)
    def test_partial_and_total_discovery_failure(self):
        def getter(url,timeout):
            if 'search=medical&' in url:return b'[{"id":"org/x","description":"medical"}]',{}
            raise HTTPError(url,503,'unavailable',{},None)
        with tempfile.TemporaryDirectory() as t:
            rows,info=capture('medical answering',3,Path(t)/'partial',getter=getter)
            self.assertEqual(len(rows),1);self.assertEqual(info['successful_list_requests'],1)
            rows,info=capture('unknown',3,Path(t)/'total',getter=getter)
            self.assertEqual(rows,[]);self.assertEqual(info['successful_list_requests'],0)
    def test_round_robin_and_query_policy(self):
        def getter(url,timeout):
            if '/api/' not in url:return b'card',{}
            items=[{'id':'org/a'},{'id':'org/b'}] if 'search=medical&' in url else [{'id':'org/c'},{'id':'org/d'}]
            return json.dumps(items).encode(),{}
        with tempfile.TemporaryDirectory() as t:
            rows,info=capture('medical question',2,Path(t)/'pool',getter=getter)
            self.assertEqual([r['dataset_id'] for r in rows],['org/c','org/a'])
        self.assertEqual(discovery_terms('medical medical question'),['medical medical question','medical','question'])

if __name__=='__main__':unittest.main(verbosity=2)
