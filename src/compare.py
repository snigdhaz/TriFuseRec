"""Combine baseline metrics and quantify whether different models help different users."""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


def main(out, seed):
    out=Path(out)
    # Only per-model result files (popularity.csv, <model>_seed<k>.csv), not curves or reports.
    csvs=sorted(p for p in out.glob('*.csv') if p.name=='popularity.csv' or re.fullmatch(r'(deepfm|sasrec|ngcf)_seed\d+\.csv',p.name))
    if csvs:
        summary=pd.concat([pd.read_csv(p) for p in csvs],ignore_index=True)
        summary.to_csv(out/'baseline_summary.csv',index=False)
        print(summary.to_string(index=False))
    hit_masks={}
    for model in ('deepfm','sasrec','ngcf'):
        p=out/f'{model}_seed{seed}_top20.npz'
        if p.exists():
            pred=np.load(p)
            hit_masks[model]=(pred['top20'][:,:10]==pred['validation_truth'][:,None]).any(axis=1)
    if len(hit_masks)>1:
        names=list(hit_masks)
        masks=np.stack([hit_masks[n] for n in names],axis=1)
        rows=[]
        for bits in range(2**len(names)):
            choice=np.array([(bits>>i)&1 for i in range(len(names))],dtype=bool)
            rows.append({'correct_models':'+'.join([n for i,n in enumerate(names) if choice[i]]) or 'none',
                         'users':int((masks==choice).all(axis=1).sum())})
        overlap=pd.DataFrame(rows)
        overlap.to_csv(out/f'overlap_seed{seed}.csv',index=False)
        print('\nTop-10 hit overlap:\n'+overlap.to_string(index=False))
    else:
        print('Train at least two neural models to create an overlap report.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args();main(args.out,args.seed)
