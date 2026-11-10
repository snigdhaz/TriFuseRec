"""TriFuseRec: personalized gating over DeepFM / SASRec / NGCF scores.

Run after `python -m src.train --model X` (or `--eval-only`) for all three models:
    python -m src.fusion

Protocol (no test leakage):
  * Base models are trained on train only.
  * Gate and fixed-weight baselines are fit on VALIDATION targets
    (gate: 80% of users to fit, 20% for early stopping).
  * Everything is reported on TEST (history = train + validation item).

Each model's scores are z-normalized per user over unseen items, so the gate
weights are comparable across models with very different score scales.

Recency feature (--recency):
  history      days between the user's last two HISTORY events (default since audit v2).
  target_time  days from the last history event to the held-out TARGET's timestamp.
               This was the original behaviour. It uses metadata of the held-out
               event (e.g. "rated in the same second as the previous movie"), so it is
               kept only to reproduce archived results. See docs/AUDIT.md.
"""
import argparse
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from .common import seed_everything, read_data, rank_metrics

MODELS=('deepfm','sasrec','ngcf')
MASKED=-1e4  # stand-in for -inf that stays finite when multiplied by a weight
FEATURES=['log_history_len','mean_log_popularity','niche_fraction','recency_days',
          *[f'conf_{m}' for m in MODELS]]


def seen_matrix(histories,n_items):
    seen=np.zeros((len(histories),n_items),dtype=bool)
    for u,h in enumerate(histories): seen[u,h]=True
    return seen


def normalized_scores(out,seed,split,seen):
    """[users, models, items] float32, z-scored per user over unseen items."""
    z=np.stack([np.load(Path(out)/'scores'/f'{m}_seed{seed}_{split}.npy') for m in MODELS],axis=1)
    mask=np.broadcast_to(seen[:,None,:],z.shape)
    z=np.where(mask,np.nan,z)
    z=(z-np.nanmean(z,axis=2,keepdims=True))/(np.nanstd(z,axis=2,keepdims=True)+1e-6)
    return np.where(mask,MASKED,z).astype(np.float32)


def user_features(z,histories,last_ts,ref_ts,popularity):
    """`ref_ts - last_ts` is the recency gap; callers decide whether ref_ts is leak-free."""
    """Per-user gate inputs: activity, mainstreamness, recency, and per-model confidence."""
    logpop=np.log1p(popularity)
    niche=popularity<np.quantile(popularity,.8)  # long-tail items: bottom 80% by count
    rows=[]
    for h in histories:
        rows.append([np.log1p(len(h)),logpop[h].mean(),niche[h].mean()])
    f=np.array(rows,dtype=np.float32)
    recency=((ref_ts-last_ts)/86400.).astype(np.float32)[:,None]
    # Confidence: how far a model's top item stands above its own 10th item.
    top=-np.sort(-z,axis=2)[:,:,:10]
    conf=(top[:,:,0]-top[:,:,9]).astype(np.float32)
    return np.concatenate([f,np.log1p(recency),conf],axis=1)


class Gate(nn.Module):
    def __init__(self,n_features,n_models=len(MODELS),hidden=16):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_features,hidden),nn.ReLU(),nn.Linear(hidden,n_models))
        self.log_scale=nn.Parameter(torch.zeros(()))  # softmax temperature for the CE loss
    def forward(self,x):
        return torch.softmax(self.net(x),dim=-1)
    def fused(self,x,z):
        return (self(x)[:,:,None]*z).sum(1)


def fuse(weights,z):
    """weights [users, models] or [models]; z [users, models, items]."""
    w=np.broadcast_to(np.asarray(weights,dtype=np.float32),z.shape[:2])
    return np.einsum('um,umi->ui',w,z)


def fit_gate(x,z,truth,seed,epochs=300,batch=256,lr=5e-3,patience=25):
    rng=np.random.default_rng(seed)
    users=rng.permutation(len(truth));cut=int(.8*len(users))
    fit_u,stop_u=users[:cut],users[cut:]
    X=torch.from_numpy(x);Z=torch.from_numpy(z);T=torch.from_numpy(truth)
    gate=Gate(x.shape[1]);opt=torch.optim.Adam(gate.parameters(),lr=lr,weight_decay=1e-4)
    def loss_on(idx):
        logits=gate.fused(X[idx],Z[idx])*gate.log_scale.exp()
        return nn.functional.cross_entropy(logits,T[idx])
    best=(np.inf,None);bad=0
    for epoch in range(epochs):
        gate.train()
        for s in range(0,len(fit_u),batch):
            idx=torch.from_numpy(fit_u[s:s+batch])
            opt.zero_grad();loss=loss_on(idx);loss.backward();opt.step()
        rng.shuffle(fit_u)
        gate.eval()
        with torch.no_grad(): val=float(loss_on(torch.from_numpy(stop_u)))
        if val<best[0]-1e-4: best=(val,{k:v.clone() for k,v in gate.state_dict().items()});bad=0
        else:
            bad+=1
            if bad>=patience: break
    gate.load_state_dict(best[1]);gate.eval()
    print(f'gate stopped after {epoch+1} epochs, early-stop CE={best[0]:.4f}')
    return gate


def best_fixed_weights(z,truth,seen,step=.1):
    """Grid search over the weight simplex on validation NDCG@10."""
    grid=np.round(np.arange(0,1+1e-9,step),6);best=(-1,None)
    for a,b in product(grid,grid):
        if a+b>1+1e-9: continue
        w=np.array([a,b,1-a-b])
        m,_=rank_metrics(fuse(w,z),truth,seen)
        if m['NDCG@10']>best[0]: best=(m['NDCG@10'],w)
    return best[1]


def main(args):
    seed_everything(args.seed)
    out=Path(args.out)
    tr,histories,valid,test,nu,ni=read_data(args.data)
    popularity=np.bincount(tr.item_idx.to_numpy(),minlength=ni).astype(np.float64)
    last_train_ts=tr.groupby('user_idx').timestamp.max().reindex(range(nu)).to_numpy(np.float64)
    vdf=pd.read_csv(Path(args.data)/'valid.csv').sort_values('user_idx')
    tdf=pd.read_csv(Path(args.data)/'test.csv').sort_values('user_idx')
    hist={'validation':histories,'test':[h+[int(v)] for h,v in zip(histories,valid)]}
    truth={'validation':valid,'test':test}
    if (Path(args.data)/'fusion.csv').exists():
        raise SystemExit('loo3 split detected: use `python -m src.cand_gate`, which fits on the fusion item.')
    if args.recency=='target_time':
        # Legacy (leaky): gap from the last history event to the held-out target's timestamp.
        last_ts={'validation':last_train_ts,'test':vdf.timestamp.to_numpy(np.float64)}
        ref_ts={'validation':vdf.timestamp.to_numpy(np.float64),'test':tdf.timestamp.to_numpy(np.float64)}
    else:
        # Leak-free: gap between the user's last two history events.
        prev_train_ts=tr.sort_values(['user_idx','timestamp']).groupby('user_idx').timestamp.agg(lambda x:x.iloc[-2]).reindex(range(nu)).to_numpy(np.float64)
        last_ts={'validation':prev_train_ts,'test':last_train_ts}
        ref_ts={'validation':last_train_ts,'test':vdf.timestamp.to_numpy(np.float64)}

    data={}
    for split in ('validation','test'):
        seen=seen_matrix(hist[split],ni)
        z=normalized_scores(out,args.seed,split,seen)
        x=user_features(z,hist[split],last_ts[split],ref_ts[split],popularity)
        data[split]=(seen,z,x)
    mu=data['validation'][2].mean(0);sd=data['validation'][2].std(0)+1e-6
    for split in data:
        seen,z,x=data[split];data[split]=(seen,z,((x-mu)/sd).astype(np.float32))

    seen_v,z_v,x_v=data['validation']
    fixed=best_fixed_weights(z_v,valid,hist['validation'])
    print('best fixed weights',dict(zip(MODELS,fixed.round(2))))
    gate=fit_gate(x_v,z_v,valid,args.seed)

    rows=[];tops={};weights_out={}
    for split in ('validation','test'):
        seen,z,x=data[split]
        with torch.no_grad(): gw=gate(torch.from_numpy(x)).numpy()
        methods={**{m:np.eye(len(MODELS))[k] for k,m in enumerate(MODELS)},
                 'equal_weights':np.full(len(MODELS),1/len(MODELS)),
                 'best_fixed_weights':fixed,'trifuserec_gate':gw}
        hits={}
        for name,w in methods.items():
            m,top=rank_metrics(fuse(w,z),truth[split],hist[split])
            rows.append({'method':name,'split':split,**m,
                         **{f'w_{mn}':float(np.mean(np.broadcast_to(w,(nu,len(MODELS)))[:,k])) for k,mn in enumerate(MODELS)}})
            hits[name]=(top[:,:10]==truth[split][:,None]).any(1)
            if split=='test': tops[name]=top
        oracle=np.stack([hits[m] for m in MODELS]).any(0).mean()
        rows.append({'method':'oracle_any_model (upper bound)','split':split,'Recall@10':float(oracle)})
        if split=='test': weights_out=gw
    res=pd.DataFrame(rows);res['recency']=args.recency
    res.to_csv(out/f'fusion_seed{args.seed}.csv',index=False)
    print(res[res.split=='test'].drop(columns='split').to_string(index=False,float_format='%.4f'))
    np.savez_compressed(out/f'fusion_seed{args.seed}_test_top20.npz',truth=test,gate_weights=weights_out,
                        history_len=np.array([len(h) for h in hist['test']]),**{f'top20_{k}':v for k,v in tops.items()})
    torch.save({'state_dict':gate.state_dict(),'feature_mean':mu,'feature_std':sd,'features':FEATURES,
                'fixed_weights':fixed,'recency':args.recency},out/f'gate_seed{args.seed}.pt')
    # Linear read-out of what drives each weight (correlation with raw feature).
    _,_,x_t=data['test']
    corr={f:{m:float(np.corrcoef(x_t[:,j],weights_out[:,k])[0,1]) for k,m in enumerate(MODELS)} for j,f in enumerate(FEATURES)}
    print('\ncorrelation(feature, gate weight) on test:\n'+pd.DataFrame(corr).T.to_string(float_format='%.2f'))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/processed')
    ap.add_argument('--out',default='results')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--recency',choices=['history','target_time'],default='history',
                    help="'target_time' reproduces pre-audit results (uses held-out timestamps)")
    main(ap.parse_args())
