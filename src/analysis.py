"""User-group accuracy, diversity, and popularity bias on the TEST split.

Run after `python -m src.fusion`:
    python -m src.analysis

Writes results/analysis_groups_seed{seed}.csv and results/analysis_beyond_accuracy_seed{seed}.csv
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_data

BINS=[0,50,100,200,500,np.inf]
LABELS=['<50','50-99','100-199','200-499','500+']


def genre_matrix(data,n_items):
    movies=pd.read_csv(Path(data)/'movies.csv')
    names=sorted({g for gs in movies.genres for g in gs.split('|')})
    col={g:k for k,g in enumerate(names)}
    m=np.zeros((n_items,len(names)),dtype=np.float32)
    for r in movies.itertuples():
        for g in r.genres.split('|'): m[int(r.item_idx),col[g]]=1
    return m


def intra_list_diversity(top,genres):
    """Mean pairwise (1 - Jaccard) genre distance inside each list."""
    g=genres[top]                                    # [users, k, genres]
    inter=np.einsum('ukg,ulg->ukl',g,g)
    size=g.sum(-1)
    union=size[:,:,None]+size[:,None,:]-inter
    dist=1-inter/np.maximum(union,1)
    k=top.shape[1]
    return float((dist.sum((1,2))/(k*(k-1))).mean())


def main(args):
    out=Path(args.out)
    tr,_,_,_,nu,ni=read_data(args.data)
    pred=np.load(out/f'fusion_seed{args.seed}_test_top20.npz')
    truth=pred['truth'];hist_len=pred['history_len'];k=args.k
    methods=[f[len('top20_'):] for f in pred.files if f.startswith('top20_')]
    popularity=np.bincount(tr.item_idx.to_numpy(),minlength=ni)
    pop_pct=pd.Series(popularity).rank(pct=True).to_numpy()
    head=popularity>=np.quantile(popularity,.8)
    genres=genre_matrix(args.data,ni)
    group=pd.cut(hist_len,BINS,labels=LABELS,right=False)

    group_rows=[];beyond=[]
    for name in methods:
        top=pred[f'top20_{name}'][:,:k]
        matches=top==truth[:,None]
        hit=matches.any(1)
        ndcg=np.where(hit,1/np.log2(np.argmax(matches,1)+2),0.)
        df=pd.DataFrame({'group':group,'hit':hit,'ndcg':ndcg})
        for g,sub in df.groupby('group',observed=True):
            group_rows.append({'method':name,'history_len':g,'users':len(sub),
                               f'Recall@{k}':sub.hit.mean(),f'NDCG@{k}':sub.ndcg.mean()})
        beyond.append({'method':name,f'Recall@{k}':hit.mean(),
                       f'coverage@{k}':len(np.unique(top))/ni,
                       f'ILD@{k}':intra_list_diversity(top,genres),
                       'mean_pop_percentile':pop_pct[top].mean(),
                       'long_tail_share':1-head[top].mean()})
    if 'gate_weights' in pred.files:
        w=pd.DataFrame(pred['gate_weights'],columns=['w_deepfm','w_sasrec','w_ngcf']);w['group']=group
        print('Mean gate weight by history length:\n'+w.groupby('group',observed=True).mean().to_string(float_format='%.3f')+'\n')
    groups=pd.DataFrame(group_rows);beyond=pd.DataFrame(beyond)
    groups.to_csv(out/f'analysis_groups_seed{args.seed}.csv',index=False)
    beyond.to_csv(out/f'analysis_beyond_accuracy_seed{args.seed}.csv',index=False)
    print(f'Recall@{k} by history length:')
    print(groups.pivot(index='method',columns='history_len',values=f'Recall@{k}')[LABELS].to_string(float_format='%.4f'))
    print('\nBeyond accuracy:\n'+beyond.to_string(index=False,float_format='%.4f'))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/processed')
    ap.add_argument('--out',default='results')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--k',type=int,default=10)
    main(ap.parse_args())
