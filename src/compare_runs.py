"""Before/after comparison for the timestamp tie-break fix (Phase 1).

OLD = item-id tie-break split (git HEAD data, results/archive_id_tiebreak/)
NEW = seeded random tie-break split (data/processed, results/)

    python -m src.compare_runs

Writes (never overwrites; adds a numeric suffix if a file exists):
  results/analysis/timestamp_fix_comparison.csv / .md
  results/analysis/gate_weights_by_cohort_before_after.csv

Note: the fix changes WHICH item is held out for users whose last ratings share a
timestamp, so OLD and NEW are evaluated on partly different targets. The differences
are not paired, and we do not compute paired statistics across them.
"""
import argparse
import io
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .common import rank_metrics

METRICS=['Recall@10','Recall@20','NDCG@10','NDCG@20']
METHODS=['popularity','deepfm','sasrec','ngcf','equal_weights','best_fixed_weights','trifuserec_gate']
HIST_BINS=[0,50,100,200,500,np.inf];HIST_LABELS=['<50','50-99','100-199','200-499','500+']


def unique_path(path):
    path=Path(path)
    if not path.exists(): return path
    k=2
    while path.with_name(f'{path.stem}_{k}{path.suffix}').exists(): k+=1
    return path.with_name(f'{path.stem}_{k}{path.suffix}')


def load_split(source):
    """source='git:<rev>' reads data/processed from git, otherwise a directory."""
    def read(name):
        if source.startswith('git:'):
            txt=subprocess.run(['git','show',f'{source[4:]}:data/processed/{name}.csv'],
                               check=True,capture_output=True,text=True).stdout
            return pd.read_csv(io.StringIO(txt))
        return pd.read_csv(Path(source)/f'{name}.csv')
    tr,va,te=read('train'),read('valid').sort_values('user_idx'),read('test').sort_values('user_idx')
    order=['user_idx','seq'] if 'seq' in tr.columns else ['user_idx','timestamp','item_idx']
    tr=tr.sort_values(order)
    return tr,va,te


def popularity_metrics(tr,va,te,n_items):
    nu=int(va.user_idx.max())+1
    hist=[[] for _ in range(nu)]
    for u,i in zip(tr.user_idx.to_numpy(),tr.item_idx.to_numpy()): hist[u].append(int(i))
    pop=np.bincount(tr.item_idx,minlength=n_items).astype(float)
    scores=np.broadcast_to(pop,(nu,n_items)).copy()
    valid=va.item_idx.to_numpy();test=te.item_idx.to_numpy()
    mv,_=rank_metrics(scores,valid,hist)
    mt,_=rank_metrics(scores,test,[h+[int(v)] for h,v in zip(hist,valid)])
    return {'validation':mv,'test':mt}


def user_cohorts(tr,va):
    """History length (= user degree in the train graph) and activity rate cohorts."""
    g=tr.groupby('user_idx')
    n=g.size().to_numpy()+1   # test-time history includes the validation item
    span_days=((g.timestamp.max()-g.timestamp.min())/86400.).to_numpy()
    rate=n/(span_days+1.)     # interactions per active day
    hist=pd.cut(n,HIST_BINS,labels=HIST_LABELS,right=False)
    act=pd.qcut(rate,4,labels=['Q1 (least active)','Q2','Q3','Q4 (most active)'])
    return pd.DataFrame({'history_len':hist,'activity':act})


def method_table(fusion_csv,pop):
    df=pd.read_csv(fusion_csv)
    rows=[]
    for split in ('validation','test'):
        for m in METHODS:
            if m=='popularity': vals=pop[split]
            else:
                hit=df[(df.method==m)&(df.split==split)]
                if hit.empty: continue
                vals=hit.iloc[0][METRICS].to_dict()
            rows.append({'method':m,'split':split,**{k:float(vals[k]) for k in METRICS}})
    return pd.DataFrame(rows)


def gate_cohorts(npz,cohorts,label):
    w=np.load(npz)['gate_weights']
    df=pd.DataFrame(w,columns=['w_deepfm','w_sasrec','w_ngcf'])
    out=[]
    for c in cohorts.columns:
        g=df.assign(cohort=cohorts[c].to_numpy()).groupby('cohort',observed=True).mean()
        g.insert(0,'cohort_type',c);g.insert(0,'run',label)
        out.append(g.reset_index())
    return pd.concat(out)


def main(args):
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    n_items=int(pd.read_csv(Path(args.new_data)/'item_map.csv').shape[0])
    old_split=load_split(args.old_data);new_split=load_split(args.new_data)
    pop_old=popularity_metrics(*old_split,n_items);pop_new=popularity_metrics(*new_split,n_items)
    old=method_table(Path(args.old_results)/'fusion_seed42.csv',pop_old)
    new=method_table(Path(args.new_results)/'fusion_seed42.csv',pop_new)
    m=old.merge(new,on=['method','split'],suffixes=('_old','_new'),how='outer')
    for k in METRICS:
        m[f'{k}_abs']=m[f'{k}_new']-m[f'{k}_old']
        m[f'{k}_rel%']=100*m[f'{k}_abs']/m[f'{k}_old']
    m['method']=pd.Categorical(m.method,METHODS,ordered=True)
    m=m.sort_values(['split','method'])
    csv=unique_path(out/'timestamp_fix_comparison.csv');m.to_csv(csv,index=False)

    changed={s:float((o.item_idx.to_numpy()!=n.item_idx.to_numpy()).mean())
             for s,o,n in (('validation',old_split[1],new_split[1]),('test',old_split[2],new_split[2]))}

    lines=['# Timestamp tie-break fix: before vs after (seed 42)','',
           'OLD = ties ordered by item id. NEW = ties ordered randomly (seed 42).',
           f"Held-out target changed for {100*changed['validation']:.1f}% of users (validation) "
           f"and {100*changed['test']:.1f}% (test), so OLD and NEW are **not paired** comparisons.",
           'Popularity was recomputed on both splits (the old run only saved validation).','']
    for split in ('test','validation'):
        lines+=[f'## {split.capitalize()}','','| Method | '+' | '.join(f'{k} old → new (Δ, rel.)' for k in METRICS)+' |',
                '|---|'+'---|'*len(METRICS)]
        for _,r in m[m.split==split].iterrows():
            cells=[f"{r[f'{k}_old']:.4f} → {r[f'{k}_new']:.4f} ({r[f'{k}_abs']:+.4f}, {r[f'{k}_rel%']:+.1f}%)" for k in METRICS]
            lines.append(f'| {r.method} | '+' | '.join(cells)+' |')
        lines.append('')

    # SASRec advantage and gate-vs-fixed, on test.
    def val(df,meth,k='Recall@10',split='test'): return float(df[(df.method==meth)&(df.split==split)][k].iloc[0])
    lines+=['## Key comparisons (test)','','| Quantity | Metric | OLD | NEW |','|---|---|---|---|']
    for k in METRICS:
        for name,f in [('SASRec − best other single model',lambda d:val(d,'sasrec',k)-max(val(d,x,k) for x in ('deepfm','ngcf','popularity'))),
                       ('SASRec / best other single model',lambda d:val(d,'sasrec',k)/max(val(d,x,k) for x in ('deepfm','ngcf','popularity'))),
                       ('Gate − best fixed weights',lambda d:val(d,'trifuserec_gate',k)-val(d,'best_fixed_weights',k)),
                       ('Best fixed − SASRec',lambda d:val(d,'best_fixed_weights',k)-val(d,'sasrec',k))]:
            lines.append(f'| {name} | {k} | {f(old):.4f} | {f(new):.4f} |')
    lines.append('')

    old_npz=Path(args.old_gate_npz);new_npz=Path(args.new_results)/'fusion_seed42_test_top20.npz'
    if old_npz.exists() and new_npz.exists():
        g=pd.concat([gate_cohorts(old_npz,user_cohorts(*old_split[:2]),'old'),
                     gate_cohorts(new_npz,user_cohorts(*new_split[:2]),'new')])
        gcsv=unique_path(out/'gate_weights_by_cohort_before_after.csv');g.to_csv(gcsv,index=False)
        lines+=['## Mean per-user gate weights by cohort (test)','',
                '| Cohort type | Cohort | OLD D / S / N | NEW D / S / N |','|---|---|---|---|']
        piv={r:g[g.run==r].set_index(['cohort_type','cohort']) for r in ('old','new')}
        order={'history_len':HIST_LABELS,'activity':['Q1 (least active)','Q2','Q3','Q4 (most active)']}
        for idx in [(t,c) for t in ('history_len','activity') for c in order[t] if (t,c) in piv['new'].index]:
            fmt=lambda r:' / '.join(f'{piv[r].loc[idx,c]:.2f}' for c in ('w_deepfm','w_sasrec','w_ngcf')) if idx in piv[r].index else 'n/a'
            lines.append(f'| {idx[0]} | {idx[1]} | {fmt("old")} | {fmt("new")} |')
        lines.append('')
    md=unique_path(out/'timestamp_fix_comparison.md');md.write_text('\n'.join(lines))
    print('\n'.join(lines));print(f'\nwrote {csv} and {md}')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--old-data',default='git:HEAD')
    ap.add_argument('--new-data',default='data/processed')
    ap.add_argument('--old-results',default='results/archive_id_tiebreak')
    ap.add_argument('--old-gate-npz',default='results/analysis/inputs_id_tiebreak/fusion_seed42_test_top20.npz')
    ap.add_argument('--new-results',default='results')
    ap.add_argument('--out-dir',default='results/analysis')
    main(ap.parse_args())
