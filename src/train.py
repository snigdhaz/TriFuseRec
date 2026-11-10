"""Train (or re-evaluate) one baseline and save full-catalog scores.

python -m src.train --model sasrec --epochs 3
python -m src.train --model deepfm --epochs 3
python -m src.train --model ngcf --epochs 3
python -m src.train --model sasrec --eval-only     # reuse results/sasrec_seed42.pt

Writes to --out:
  {model}_seed{seed}.csv          validation + test metrics
  {model}_seed{seed}_top20.npz    top-20 lists (validation and test)
  {model}_seed{seed}.pt           weights
  scores/{model}_seed{seed}_{split}.npy   full [users, items] score matrix (used by fusion)
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from .common import seed_everything, read_data, rank_metrics, seen_matrix, sample_unseen, heldout_splits
from .models import DeepFM,SASRec,NGCF


def features(path,n_users,n_items,device):
    users=pd.read_csv(path/'users.csv')
    movies=pd.read_csv(path/'movies.csv')
    ages=sorted(users.age.unique());genders=sorted(users.gender.unique());jobs=sorted(users.occupation.unique())
    a={v:k for k,v in enumerate(ages)};g={v:k for k,v in enumerate(genders)};o={v:k for k,v in enumerate(jobs)}
    user_f=np.zeros((n_users,3),dtype=np.int64)
    for r in users.itertuples():user_f[int(r.user_idx)]=[a[r.age],g[r.gender],o[r.occupation]]
    genre_names=sorted({genre for genres in movies.genres for genre in genres.split('|')})
    genre_map={v:k for k,v in enumerate(genre_names)}
    genre_f=np.zeros((n_items,len(genre_names)),dtype=np.float32)
    for r in movies.itertuples():
        for genre in r.genres.split('|'):genre_f[int(r.item_idx),genre_map[genre]]=1
    return (torch.from_numpy(user_f).to(device),torch.from_numpy(genre_f).to(device),
            [len(ages),len(genders),len(jobs),len(genre_names)])


def sequence_rows(histories,max_len):
    inputs=[];targets=[]
    for h in histories:
        for t in range(1,len(h)):
            inputs.append(([0]*max(0,max_len-t)+[x+1 for x in h[max(0,t-max_len):t]])[-max_len:])
            targets.append(h[t])
    return torch.tensor(inputs,dtype=torch.long),torch.tensor(targets,dtype=torch.long)


def fit(args,model,tr,histories,val_hist,valid,nu,ni,device,uf=None,gf=None):
    """Train on `histories` (train split); early-stop on validation NDCG@10 using `val_hist`."""
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(args.seed)
    seen=seen_matrix(tr,ni)
    positives=tr[['user_idx','item_idx']].to_numpy(dtype=np.int64)
    if args.model=='sasrec':
        sequences,labels=sequence_rows(histories,args.max_len)
        loader=DataLoader(TensorDataset(sequences,labels),batch_size=args.batch,shuffle=True)
    best={'NDCG@10':-1.,'epoch':0,'state':None};bad=0
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        if args.model=='ngcf':
            # NGCF propagates across the entire graph for each optimizer step.
            rng.shuffle(positives)
            for chunk in np.array_split(positives,max(1,int(np.ceil(len(positives)/args.graph_batch)))):
                j=sample_unseen(chunk[:,0],seen,rng)
                ut,it,jt=[torch.as_tensor(a,device=device) for a in (chunk[:,0],chunk[:,1],j)]
                opt.zero_grad();loss=model.bpr(ut,it,jt);loss.backward();opt.step();losses.append(float(loss.item()))
        else:
            if args.model=='deepfm':
                # Fresh negatives every epoch.
                neg_u=np.repeat(positives[:,0],args.neg);neg_i=sample_unseen(neg_u,seen,rng)
                us=np.concatenate([positives[:,0],neg_u]);its=np.concatenate([positives[:,1],neg_i])
                ys=np.concatenate([np.ones(len(positives)),np.zeros(len(neg_u))]).astype(np.float32)
                loader=DataLoader(TensorDataset(torch.from_numpy(us),torch.from_numpy(its),torch.from_numpy(ys)),batch_size=args.batch,shuffle=True)
            for batch in loader:
                opt.zero_grad()
                if args.model=='deepfm':
                    u,i,y=[x.to(device) for x in batch]
                    f=uf[u];logit=model(u,i,f[:,0],f[:,1],f[:,2],gf[i])
                    loss=nn.functional.binary_cross_entropy_with_logits(logit,y)
                else:
                    seq,y=[x.to(device) for x in batch]
                    loss=nn.functional.cross_entropy(model(seq),y)
                loss.backward();opt.step();losses.append(float(loss.item()))
        msg=f'{args.model} epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f}'
        if epoch%args.eval_every==0 or epoch==args.epochs:
            metrics,_=rank_metrics(score_all(args,model,val_hist,nu,ni,device,uf,gf),valid,val_hist)
            msg+=f" val Recall@10={metrics['Recall@10']:.4f} NDCG@10={metrics['NDCG@10']:.4f}"
            if metrics['NDCG@10']>best['NDCG@10']:
                best={'NDCG@10':metrics['NDCG@10'],'epoch':epoch,
                      'state':{k:v.detach().clone() for k,v in model.state_dict().items()}};bad=0
            else:
                bad+=1
            if bad>=args.patience:
                print(msg+' (early stop)',flush=True);break
        print(msg,flush=True)
    model.load_state_dict(best['state'])
    print(f"best epoch {best['epoch']} (val NDCG@10={best['NDCG@10']:.4f})")
    return best['epoch']


@torch.no_grad()
def score_all(args,model,histories,nu,ni,device,uf=None,gf=None):
    """Full [users, items] score matrix given each user's (chronological) history."""
    model.eval()
    scores=np.empty((nu,ni),dtype=np.float32)
    if args.model=='ngcf':
        eu,ei=model.encode()
        for start in range(0,nu,args.eval_batch):
            scores[start:start+args.eval_batch]=(eu[start:start+args.eval_batch]@ei.T).cpu().numpy()
        return scores
    items=torch.arange(ni,device=device)
    for start in range(0,nu,args.eval_batch):
        end=min(start+args.eval_batch,nu)
        if args.model=='sasrec':
            seq=np.zeros((end-start,args.max_len),dtype=np.int64)
            for row,u in enumerate(range(start,end)):
                hist=[i+1 for i in histories[u][-args.max_len:]]
                seq[row,-len(hist):]=hist
            scores[start:end]=model(torch.from_numpy(seq).to(device)).cpu().numpy()
        else:
            for s in range(start,end,16):
                e=min(s+16,end)
                uu=torch.arange(s,e,device=device).repeat_interleave(ni);ii=items.repeat(e-s);f=uf[uu]
                scores[s:e]=model(uu,ii,f[:,0],f[:,1],f[:,2],gf[ii]).view(e-s,ni).cpu().numpy()
    return scores


def train(args):
    seed_everything(args.seed)
    path=Path(args.data);output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    (output/'scores').mkdir(exist_ok=True)
    if args.cpu: device=torch.device('cpu')
    elif torch.cuda.is_available(): device=torch.device('cuda')
    elif torch.backends.mps.is_available(): device=torch.device('mps')
    else: device=torch.device('cpu')
    print('device',device)
    tr,histories,valid,test,nu,ni=read_data(path)
    uf=gf=None
    if args.model=='deepfm':
        uf,gf,cats=features(path,nu,ni,device)
        model=DeepFM(nu,ni,*cats,d=args.dim).to(device)
    elif args.model=='sasrec':
        model=SASRec(ni,d=args.dim,max_len=args.max_len).to(device)
    else:
        model=NGCF(nu,ni,tr[['user_idx','item_idx']].to_numpy(dtype=np.int64),d=args.dim).to(device)
    stem=f'{args.model}_seed{args.seed}'
    splits=heldout_splits(path,histories)  # loo2: validation,test   loo3: fusion,validation,test
    best_epoch=None
    if args.eval_only:
        model.load_state_dict(torch.load(output/f'{stem}.pt',map_location=device))
    else:
        best_epoch=fit(args,model,tr,histories,*splits['validation'],nu,ni,device,uf,gf)
        torch.save(model.state_dict(),output/f'{stem}.pt')

    # Every split ranks all items minus that split's history (train + earlier held-out
    # items). Model weights are fixed (trained on train only).
    train_edges=tr[['user_idx','item_idx']].to_numpy(dtype=np.int64)
    rows=[];tops={};cache=None
    for split,(hist,truth) in splits.items():
        if args.model=='sasrec':
            scores=score_all(args,model,hist,nu,ni,device,uf,gf)
        elif args.model=='ngcf' and args.ngcf_infer_graph=='history':
            # Opt-in: propagate over train + held-out history edges, so NGCF sees the
            # same history as SASRec. Default 'train' reproduces earlier results.
            extra=[(u,i) for u,h in enumerate(hist) for i in h[len(histories[u]):]]
            model.set_graph(np.concatenate([train_edges,np.array(extra,dtype=np.int64).reshape(-1,2)]))
            scores=score_all(args,model,hist,nu,ni,device,uf,gf)
        else:
            # DeepFM (and NGCF on the train graph) ignore the history: score once.
            if cache is None: cache=score_all(args,model,hist,nu,ni,device,uf,gf)
            scores=cache
        np.save(output/'scores'/f'{stem}_{split}.npy',scores)
        metrics,top=rank_metrics(scores,truth,hist)
        rows.append({'model':args.model,'seed':args.seed,'split':split,**metrics,'dim':args.dim,'best_epoch':best_epoch,
                     'ngcf_infer_graph':args.ngcf_infer_graph if args.model=='ngcf' else None});tops[split]=top
        print(split.upper()+':',json.dumps(metrics))
    pd.DataFrame(rows).to_csv(output/f'{stem}.csv',index=False)
    np.savez_compressed(output/f'{stem}_top20.npz',top20=tops['validation'],validation_truth=splits['validation'][1],
                        test_top20=tops['test'],test_truth=splits['test'][1])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=['deepfm','sasrec','ngcf'])
    p.add_argument('--data',default='data/processed')
    p.add_argument('--out',default='results')
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dim',type=int,default=32)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--graph-batch',type=int,default=8192)
    p.add_argument('--eval-batch',type=int,default=128)
    p.add_argument('--max-len',type=int,default=50)
    p.add_argument('--neg',type=int,default=2)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--eval-every',type=int,default=1,help='validate every N epochs')
    p.add_argument('--patience',type=int,default=3,help='stop after N validations without NDCG@10 gain')
    p.add_argument('--ngcf-infer-graph',choices=['train','history'],default='train',
                   help="NGCF scoring graph: 'train' (v1/v2 default) or train + held-out history edges")
    p.add_argument('--eval-only',action='store_true',help='load {model}_seed{seed}.pt instead of training')
    train(p.parse_args())
