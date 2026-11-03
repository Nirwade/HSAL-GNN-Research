import pandas as pd, numpy as np, re
from pathlib import Path
HERE=Path(__file__).resolve().parent
d=pd.read_csv(HERE.parent/'inputs'/'Goodreads_young_adult_HSAL_100k.csv')
s=d.drop_duplicates('item_id').set_index('item_id').item_series_id
def parse(x):
    m=re.match(r'Series_(\d+)_P([\d.]+)',x); return (m.group(1),float(m.group(2))) if m else (None,None)
sid={i:parse(x)[0] for i,x in s.items()}; pos={i:parse(x)[1] for i,x in s.items()}
E=pd.read_csv(HERE/'edges_all.csv')
V={'all':E,'strict':E[(E.low_coherence_flag!=True)|(E.series_match=='match')],'confirmed':E[(E.match_type=='exact')|(E.series_match=='match')]}
def eset(x): 
    a=set(zip(x.item_id,x.matched_item_id)); return a|{(b,c) for c,b in a}
ES={k:eset(v) for k,v in V.items()}
nb={k:{} for k in ES}
for k,e in ES.items():
    for a,b in e: nb[k].setdefault(a,set()).add(b)
d=d.sort_values(['user_id','time'],kind='stable')
rows=[]
for u,seq in d.groupby('user_id').item_id:
    seq=list(seq); n=len(seq)
    if n<3: continue
    for j in range(1,n-1):
        split='test' if j==n-2 else ('val' if j==n-3 else 'train')
        hist=seq[max(0,j-49):j+1]  # items in window up to last_item (j inclusive)
        t=seq[j+1]; st=sid.get(t)
        same=[h for h in hist if st is not None and sid.get(h)==st]
        c='B' if same and any(pos[t]>pos[h] for h in same) else ('C' if same else 'N')
        r={'split':split,'cls':c}
        for k in ES: r['hit_'+k]=any(t in nb[k].get(h,()) for h in hist)
        rows.append(r)
R=pd.DataFrame(rows)
def lab(r,k): 
    if r.cls in('B','C'): return r.cls
    return 'D' if r['hit_'+k] else 'A'
for k in ES:
    R['final_'+k]=R.apply(lambda r:lab(r,k),axis=1)
    print('\n',k); print(pd.crosstab(R.split,R['final_'+k],margins=True))
print('\nLLM hit inside B/C (test):',R[(R.split=='test')&R.cls.isin(['B','C'])].groupby('cls').hit_all.agg(['sum','count']))
R.to_csv(HERE/'cd_result.csv',index=False)
