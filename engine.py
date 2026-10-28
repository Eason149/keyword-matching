"""Sparse retrieval with explicit preprocessing and field ablations."""
import math,re,unicodedata
from collections import Counter,defaultdict
from functools import lru_cache
import numpy as np
from nltk.stem import PorterStemmer

BASE_STOP=set('a an the and or of for to in on with is are be this that dataset datasets data'.split())
# Fixed list, chosen without benchmark-label inspection. Applied to query keyword extraction only.
QUERY_STOP=BASE_STOP|set('i we our my want wants would like need looking find looking propose proposed present paper approach method model models system systems build using use used can could should improve improving better study investigate new novel based via through'.split())
STEMMER=PorterStemmer()
stem_word=lru_cache(maxsize=150000)(STEMMER.stem)
TOKENIZER_VERSION='nfkc-ascii-alnum-v1'

def clean(text,mode='basic'):
    text=str(text or '')
    text=re.sub(r'<!--.*?-->',' ',text,flags=re.S)
    text=re.sub(r'<(?:script|style)\b[^>]*>.*?</(?:script|style)>',' ',text,flags=re.S|re.I)
    if mode=='section':
        text=text.replace('\\r\\n','\n').replace('\\n','\n')
        text=re.sub(r'\A\s*---\s*\n.*?\n---\s*\n',' ',text,flags=re.S)
        text=re.sub(r'```.*?```',' ',text,flags=re.S)
        lines=[];skip=False
        for line in text.splitlines():
            if re.match(r'^\s*#{1,6}\s',line):
                skip=bool(re.search(r'\b(citation|references|table of contents|acknowledg|bibtex)\b',line,re.I))
            if skip or re.search(r'\[.*?\]\(#[^)]*\)',line):continue
            if 'More Information Needed' in line:continue
            lines.append(line)
        text='\n'.join(lines)
    text=re.sub(r'!?\[([^\]]*)\]\([^)]*\)',r'\1',text)
    text=re.sub(r'https?://\S+',' ',text)
    return re.sub(r'<[^>]+>',' ',text)

def tokenize(text,stem=False,query_keywords=False):
    text=unicodedata.normalize('NFKC',str(text)).lower()
    words=[]
    for token in re.findall(r'[a-z0-9]+|[\u3400-\u9fff]+',text):
        if re.fullmatch(r'[\u3400-\u9fff]+',token):
            words.extend(token);words.extend(token[i:i+2] for i in range(len(token)-1))
        else:words.append(token)
    stop=QUERY_STOP if query_keywords else BASE_STOP
    words=[w for w in words if w not in stop]
    return [stem_word(w) for w in words] if stem else words

def field(row,key,mode):
    value=row.get(key,'')
    text=' '.join(map(str,value)) if isinstance(value,list) else str(value or '')
    return clean(text,mode) if key in ['readme','description','description_only'] else text

class SparseIndex:
    def __init__(self,records,fields=('name','description','tags','column_names','readme'),stem=False,cleaning='basic',k1=1.2,b=.75,weights=None):
        if not math.isfinite(k1) or k1<=0 or not 0<=b<=1:raise ValueError('Invalid BM25 parameters')
        self.records=records;self.fields=fields;self.stem=stem;self.k1=k1;self.b=b;self.n=len(records)
        self.ids=[r['dataset_id'] for r in records];self.weights=weights
        # Cache tokenized fields: independent fields only for the weighted-score experiment.
        groups={f:[f] for f in fields} if weights else {'all':list(fields)}
        self.groups={}
        for group,keys in groups.items():
            counts=[Counter(tokenize(' '.join(field(r,k,cleaning) for k in keys),stem)) for r in records]
            lengths=np.array([sum(c.values()) for c in counts],dtype=float)
            avg=lengths.mean() if len(lengths) else 0
            postings=defaultdict(list)
            for i,counter in enumerate(counts):
                for term,tf in counter.items():postings[term].append((i,tf))
            prepared={}
            for term,values in postings.items():
                indices=np.array([x[0] for x in values],dtype=np.int32)
                tfs=np.array([x[1] for x in values],dtype=float)
                idf=math.log1p((self.n-len(values)+.5)/(len(values)+.5))
                normalized=tfs*(k1+1)/(tfs+k1*(1-b+b*lengths[indices]/(avg or 1)))
                prepared[term]=(indices,idf*normalized)
            self.groups[group]=prepared
        self.vocabulary=set().union(*(set(x) for x in self.groups.values()))

    def search(self,text,k=200,keyword=False,query_keywords=False,query_year=None):
        terms=set(tokenize(text,self.stem,query_keywords));scores=np.zeros(self.n)
        for group,postings in self.groups.items():
            weight=self.weights.get(group,1) if self.weights else 1
            for term in terms:
                if term in postings:
                    indices,values=postings[term]
                    scores[indices]+=weight if keyword else weight*values
        if query_year is not None:
            for i,r in enumerate(self.records):
                if r.get('year') and r['year']>int(query_year):scores[i]=0
        hits=np.flatnonzero(scores>0)
        order=sorted(hits,key=lambda i:(-scores[i],self.ids[i]))[:k]
        return [(self.ids[i],float(scores[i])) for i in order]

def score_ranking(ranking,labels):
    positive={d for d,g in labels.items() if g>0};den=len(positive)
    dcg=lambda grades:sum((2**g-1)/math.log2(i+2) for i,g in enumerate(grades))
    ideal=dcg(sorted(labels.values(),reverse=True)[:10])
    metrics={'ndcg10':dcg([labels.get(x,0) for x in ranking[:10]])/ideal if ideal else 0}
    for k in [20,100,200]:metrics['recall'+str(k)]=len(set(ranking[:k])&positive)/den if den else 0
    metrics['mrr200']=next((1/i for i,d in enumerate(ranking[:200],1) if d in positive),0)
    return metrics
