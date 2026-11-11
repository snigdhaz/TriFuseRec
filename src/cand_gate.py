"""Per-user vs candidate-aware gating over frozen expert scores (Phases 4-5).

    python -m src.cand_gate --scores results/scores --out results/cand_gate/seed42

Fusion:   s(u,i) = sum_k w_k(.) * s_k(u,i)
  mode=user       w(u)   = softmax(MLP(q_user, conf_user))           same weights for every item
  mode=candidate  w(u,i) = softmax(MLP(q_user, q_item, q_expert(u,i)))

Data arrangement (no label is used both to fit and to select a gate):
  loo3 split (fusion.csv exists):  fit = fusion item (all users)
                                   select/early-stop = validation item, test = test item
  loo2 split:                      fit = validation item of a seeded 70% of users
                                   select/early-stop = validation item of the other 30%
                                   test = test item of all users
  Caveat (loo2): the frozen base models were themselves early-stopped on the
  validation item of all users, so selection numbers are mildly optimistic. Test
  labels are never used for fitting or selection under either protocol.

Inputs (all computed from histories, train popularity, item metadata and expert
scores — never from held-out labels or held-out timestamps):
  user_history  log history length, mean log popularity of history, long-tail share
  recency       log gap between the user's last two HISTORY events, log activity rate
  item_pop      log train popularity of the candidate
  item_genre    18 genre indicators of the candidate (Amazon Beauty: 6 sub-category indicators)
  expert        per expert: clipped z-score, within-user log-rank, per-user confidence
                (top-1 minus top-10 z); plus cross-expert disagreement (rank std)

Normalization (before fusion), computed per user over UNSEEN items only:
  z     (s - mean) / std; users whose std < 1e-6 get z = 0 (no information)
  rank  -log(1 + rank), rank 0 = the expert's top unseen item

Loss: sampled softmax cross-entropy. Candidates per fit user = the target, the
union of each expert's top-H unseen items (hard negatives; target removed), and R
uniform unseen items re-sampled every epoch. Hard negatives matter because top-10
ranking is decided among items the experts already rank highly.
"""
import argparse
import gc
import hashlib
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .common import seed_everything, read_data, heldout_splits, rank_metrics

EXPERTS=('deepfm','sasrec','ngcf')
GROUPS=('user_history','recency','item_pop','item_genre','expert')
USER_GROUPS=('user_history','recency')
MASKED=-1e4
METHODS=('best_single','equal_weights','fixed_weights','per_user_gate','candidate_gate')


# ---------------------------------------------------------------- data

def split_identity(data):
    h=hashlib.sha1()
    for f in ('train.csv','fusion.csv','valid.csv','test.csv'):
        p=Path(data)/f
        if p.exists(): h.update(p.read_bytes())
    return h.hexdigest()[:12]


def history_times(data,train):
    """Per held-out split: (first, previous, last) timestamp of each user's HISTORY."""
    data=Path(data)
    order=['user_idx','seq'] if 'seq' in train.columns else ['user_idx','timestamp','item_idx']
    g=train.sort_values(order).groupby('user_idx').timestamp
    first=g.min().to_numpy(np.float64)
    last=g.agg(lambda x:x.iloc[-1]).to_numpy(np.float64);prev=g.agg(lambda x:x.iloc[-2]).to_numpy(np.float64)
    out={}
    for name,fname in (('fusion','fusion.csv'),('validation','valid.csv'),('test','test.csv')):
        if not (data/fname).exists(): continue
        out[name]=(first,prev,last)
        ts=pd.read_csv(data/fname).sort_values('user_idx').timestamp.to_numpy(np.float64)
        prev,last=last,ts       # this held-out event becomes history for later splits
    return out


def item_categories(data,n_items):
    """[items, C] category indicators for the `item_genre` feature group.
    ML-1M: 18 movie genres (movies.csv). Amazon Beauty: one-hot of the 6 product
    sub-categories (items.csv `subcategory`; items without metadata get all zeros)."""
    data=Path(data)
    if (data/'movies.csv').exists():
        df=pd.read_csv(data/'movies.csv').dropna(subset=['item_idx'])
        rows=[(int(r.item_idx),r.genres.split('|')) for r in df.itertuples()]
    else:
        df=pd.read_csv(data/'items.csv')
        rows=[(int(r.item_idx),[r.subcategory]) for r in df.itertuples() if isinstance(r.subcategory,str) and r.subcategory]
    names=sorted({g for _,gs in rows for g in gs});k={g:j for j,g in enumerate(names)}
    out=np.zeros((n_items,len(names)),dtype=np.float32)
    for i,gs in rows:
        for g in gs: out[i,k[g]]=1
    return out


def user_features(hist,times,logpop,niche):
    """[users, 5]: user_history (3) + recency (2). Uses history only — never the target."""
    first,prev,last=times
    f=np.array([[np.log1p(len(h)),logpop[h].mean(),niche[h].mean()] for h in hist],dtype=np.float32)
    gap=np.log1p(np.maximum(last-prev,0)/86400.)
    rate=np.log([len(h)/((l-s)/86400.+1.) for h,s,l in zip(hist,first,last)])
    return np.concatenate([f,np.stack([gap,rate],1).astype(np.float32)],1)


def normalize(scores,seen,eps=1e-6):
    """Per-user z-score over unseen items; zero-variance users map to 0. Masked -> MASKED."""
    s=np.where(seen,np.nan,scores.astype(np.float64))
    mu=np.nanmean(s,axis=1,keepdims=True);sd=np.nanstd(s,axis=1,keepdims=True)
    z=np.where(sd<eps,0.,(s-mu)/np.maximum(sd,eps))
    return np.where(seen,MASKED,z).astype(np.float32)


def log_ranks(z):
    """log1p(rank) of each item within the user's list (seen items, at MASKED, rank last)."""
    order=np.argsort(-z,axis=1,kind='stable')
    ranks=np.empty_like(order)
    np.put_along_axis(ranks,order,np.broadcast_to(np.arange(z.shape[1]),z.shape),axis=1)
    return np.log1p(ranks).astype(np.float32),order


class View:
    """Expert evidence for one held-out split: z [U,E,I], log-rank [U,E,I], seen, features."""
    def __init__(self,name,hist,truth,times,score_files,logpop,niche,hard_k,chunk=2048):
        self.name=name;self.hist=hist;self.truth=truth
        nu=len(hist);self.seen=np.zeros((nu,len(logpop)),dtype=bool)
        for u,h in enumerate(hist): self.seen[u,h]=True
        # Filled per expert and per block of users: every step is row-wise, so the result is
        # identical to processing whole matrices, but peak memory stays near the output size
        # (needed for Amazon Beauty: 22k users x 12k items x 3 experts).
        E,ni=len(score_files),len(logpop)
        self.z=np.empty((nu,E,ni),dtype=np.float32);self.lr=np.empty((nu,E,ni),dtype=np.float16)
        self.hard=np.empty((nu,E,hard_k),dtype=np.int64);self.conf=np.empty((nu,E),dtype=np.float32)
        for e,f in enumerate(score_files):
            raw=np.load(f,mmap_mode='r')
            for s in range(0,nu,chunk):
                z=normalize(np.asarray(raw[s:s+chunk]),self.seen[s:s+chunk]);lr,order=log_ranks(z)
                self.z[s:s+chunk,e]=z;self.lr[s:s+chunk,e]=lr;self.hard[s:s+chunk,e]=order[:,:hard_k]
                top=-np.sort(-z,axis=1)[:,:10];self.conf[s:s+chunk,e]=top[:,0]-top[:,9]
        self.user_x=user_features(hist,times,logpop,niche)


# ---------------------------------------------------------------- model

class GateNet(nn.Module):
    def __init__(self,n_in,n_experts,hidden=32,dropout=.1):
        super().__init__()
        self.n_in=n_in
        if n_in: self.net=nn.Sequential(nn.Linear(n_in,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,n_experts))
        else: self.logits=nn.Parameter(torch.zeros(n_experts))   # no inputs: learned global weights
        self.log_scale=nn.Parameter(torch.zeros(()))
    def forward(self,x):
        if self.n_in: return torch.softmax(self.net(x),-1)
        return torch.softmax(self.logits,-1).expand(*x.shape[:-1],-1)


class Fuser:
    """Builds gate inputs for (user, candidate) blocks and fuses expert scores."""
    def __init__(self,mode,norm,groups,item_pop,item_genre,n_items,user_mu,user_sd):
        self.mode=mode;self.norm=norm;self.groups=set(groups);self.n_items=n_items
        self.item_pop=torch.from_numpy(item_pop);self.item_genre=torch.from_numpy(item_genre)
        self.user_mu=torch.from_numpy(user_mu);self.user_sd=torch.from_numpy(user_sd)
        self.user_cols=[k for k,g in enumerate(['user_history']*3+['recency']*2) if g in self.groups]
    def n_features(self,n_experts):
        n=len(self.user_cols)+(n_experts if 'expert' in self.groups else 0)
        if self.mode=='candidate':
            n+=('item_pop' in self.groups)+(self.item_genre.shape[1] if 'item_genre' in self.groups else 0)
            n+=(2*n_experts+1) if 'expert' in self.groups else 0
        return n
    def block(self,view,users,items):
        """users [b] (np), items [b,c] (torch long) -> fusion scores s_hat [b,c,E], gate input."""
        u=torch.from_numpy(users)
        z=torch.from_numpy(view.z[users]).gather(2,items[:,None,:].expand(-1,view.z.shape[1],-1)).transpose(1,2)
        lr=torch.from_numpy(view.lr[users].astype(np.float32)).gather(2,items[:,None,:].expand(-1,view.lr.shape[1],-1)).transpose(1,2)
        masked=z<=MASKED/2
        s_hat=z if self.norm=='z' else torch.where(masked,torch.full_like(lr,MASKED),-lr)
        ux=((torch.from_numpy(view.user_x[users])-self.user_mu)/self.user_sd)[:,self.user_cols]
        parts=[ux]
        if 'expert' in self.groups: parts.append(torch.from_numpy(view.conf[users]))
        x=torch.cat(parts,-1)
        if self.mode=='user': return s_hat,x
        b,c=items.shape
        cols=[x[:,None,:].expand(b,c,-1)]
        if 'item_pop' in self.groups: cols.append(self.item_pop[items][...,None])
        if 'item_genre' in self.groups: cols.append(self.item_genre[items])
        if 'expert' in self.groups:
            lrn=lr/np.log(self.n_items)
            cols+=[z.clamp(-5,10)/5,lrn,lrn.std(-1,keepdim=True)]
        return s_hat,torch.cat(cols,-1)
    def fuse(self,gate,s_hat,x):
        w=gate(x)
        if self.mode=='user': w=w[:,None,:].expand_as(s_hat)
        return (w*s_hat).sum(-1),w


@torch.no_grad()
def full_scores(fuser,gate,view,users,chunk=64):
    """Fused scores over the full catalog for `users` (masked items stay masked)."""
    gate.eval();out=np.empty((len(users),fuser.n_items),dtype=np.float32)
    items_all=torch.arange(fuser.n_items)
    for s in range(0,len(users),chunk):
        ub=users[s:s+chunk];items=items_all.expand(len(ub),-1)
        sh,x=fuser.block(view,ub,items);f,_=fuser.fuse(gate,sh,x)
        out[s:s+chunk]=f.numpy()
    return out


@torch.no_grad()
def weights_for(fuser,gate,view,users,items):
    gate.eval();sh,x=fuser.block(view,users,torch.as_tensor(items,dtype=torch.long))
    return fuser.fuse(gate,sh,x)[1].numpy()


def candidates(view,users,truth,n_random,rng):
    """[target | union of experts' top-H unseen items | uniform unseen fill], fixed width.

    The hard-negative union is padded with uniform unseen items up to E*H slots, then
    n_random more uniform items are added. Uniform items may repeat a hard negative
    (rare; harmless for the softmax). The target and seen items are never negatives.
    """
    width=view.hard.shape[1]*view.hard.shape[2];n_items=view.seen.shape[1]
    rows=np.empty((len(users),1+width+n_random),dtype=np.int64)
    for r,(u,t) in enumerate(zip(users,truth)):
        h=np.unique(view.hard[u].ravel());h=h[h!=t]
        need=width+n_random-len(h);pool=np.empty(0,dtype=np.int64)
        while len(pool)<need:
            draw=rng.integers(n_items,size=2*need+64)
            pool=np.concatenate([pool,draw[~view.seen[u,draw]&(draw!=t)]])
        rows[r,0]=t;rows[r,1:]=np.concatenate([h,pool[:need]])
    return rows


def ndcg_at(scores,truth,hist,k=10):
    return rank_metrics(scores,truth,hist,ks=(k,))[0][f'NDCG@{k}']


def train_gate(fuser,view_fit,fit_users,view_sel,sel_users,n_experts,args,log):
    rng=np.random.default_rng(args.seed);torch.manual_seed(args.seed)
    gate=GateNet(fuser.n_features(n_experts),n_experts,args.hidden,args.dropout)
    opt=torch.optim.Adam(gate.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sel_truth=view_sel.truth[sel_users];sel_hist=[view_sel.hist[u] for u in sel_users]
    best=(-1.,None,0);bad=0
    for epoch in range(1,args.epochs+1):
        gate.train();order=rng.permutation(fit_users);losses=[]
        cand=candidates(view_fit,order,view_fit.truth[order],args.random_negatives,rng)
        for s in range(0,len(order),args.batch):
            ub=order[s:s+args.batch];items=torch.from_numpy(cand[s:s+args.batch])
            sh,x=fuser.block(view_fit,ub,items);f,_=fuser.fuse(gate,sh,x)
            loss=nn.functional.cross_entropy(f*gate.log_scale.exp(),torch.zeros(len(ub),dtype=torch.long))
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
        sel=ndcg_at(full_scores(fuser,gate,view_sel,sel_users),sel_truth,sel_hist)
        log.append({'mode':fuser.mode,'epoch':epoch,'loss':float(np.mean(losses)),'select_NDCG@10':sel})
        print(f'  {fuser.mode} gate epoch {epoch} loss={np.mean(losses):.4f} select NDCG@10={sel:.4f}',flush=True)
        if sel>best[0]+1e-5: best=(sel,{k:v.clone() for k,v in gate.state_dict().items()},epoch);bad=0
        else:
            bad+=1
            if bad>=args.patience: break
    gate.load_state_dict(best[1]);gate.eval()
    return gate,best[2],best[0]


def fixed_grid(view,users,n_experts,norm,step=.1):
    """Global weights on the simplex, chosen on selection NDCG@10."""
    sh=view.z[users] if norm=='z' else np.where(view.z[users]<=MASKED/2,MASKED,-view.lr[users].astype(np.float32))
    truth=view.truth[users];hist=[view.hist[u] for u in users]
    grid=np.round(np.arange(0,1+1e-9,step),6);best=(-1,None)
    for w in product(grid,repeat=n_experts-1):
        if sum(w)>1+1e-9: continue
        w=np.array([*w,1-sum(w)],dtype=np.float32)
        v=ndcg_at(np.einsum('e,uei->ui',w,sh),truth,hist)
        if v>best[0]: best=(v,w)
    return best[1]


# ---------------------------------------------------------------- main

def run(args):
    seed_everything(args.seed)
    out=Path(args.out)
    if (out/'metrics.csv').exists() and not args.overwrite:
        raise SystemExit(f'{out}/metrics.csv exists; pass --overwrite to replace it')
    out.mkdir(parents=True,exist_ok=True)
    experts=tuple(args.experts);E=len(experts)
    tr,histories,_,_,nu,ni=read_data(args.data)
    splits=heldout_splits(args.data,histories);times=history_times(args.data,tr)
    pop=np.bincount(tr.item_idx.to_numpy(),minlength=ni).astype(np.float64)
    logpop=np.log1p(pop);niche=pop<np.quantile(pop,.8)
    genre=item_categories(args.data,ni)
    item_pop=((logpop-logpop.mean())/logpop.std()).astype(np.float32)

    protocol='loo3' if 'fusion' in splits else 'loo2'
    def view(name):
        files=[Path(args.scores)/f'{e}_seed{args.score_seed}_{name}.npy' for e in experts]
        return View(name,*splits[name],times[name],files,logpop,niche,args.hard_negatives)
    print(f'protocol={protocol} experts={experts} mode(s)={args.methods} norm={args.norm}',flush=True)
    if protocol=='loo3':
        v_fit=view('fusion');v_sel=view('validation');fit_users=np.arange(nu);sel_users=np.arange(nu)
    else:
        v_fit=v_sel=view('validation')
        perm=np.random.default_rng(args.split_seed).permutation(nu);cut=int(args.fit_fraction*nu)
        fit_users=np.sort(perm[:cut]);sel_users=np.sort(perm[cut:])
    test_users=np.arange(nu)
    ux=v_fit.user_x[fit_users];mu=ux.mean(0).astype(np.float32);sd=(ux.std(0)+1e-6).astype(np.float32)

    def make_fuser(mode,groups=None):
        g=groups if groups is not None else [x for x in GROUPS if x not in args.drop_groups]
        return Fuser(mode,args.norm,g,item_pop,genre,ni,mu,sd)

    def s_hat(view,users):
        return view.z[users] if args.norm=='z' else np.where(view.z[users]<=MASKED/2,MASKED,-view.lr[users].astype(np.float32))

    sel_truth=v_sel.truth[sel_users];sel_hist=[v_sel.hist[u] for u in sel_users]
    test_hist,test_truth=splits['test']
    weights={};gates={};log=[];info={}
    if 'best_single' in args.methods:
        sel=[ndcg_at(s_hat(v_sel,sel_users)[:,k],sel_truth,sel_hist) for k in range(E)]
        k=int(np.argmax(sel));weights['best_single']=np.eye(E,dtype=np.float32)[k];info['best_single']=experts[k]
    if 'equal_weights' in args.methods: weights['equal_weights']=np.full(E,1/E,dtype=np.float32)
    if 'fixed_weights' in args.methods: weights['fixed_weights']=fixed_grid(v_sel,sel_users,E,args.norm)
    for mode,name in (('user','per_user_gate'),('candidate','candidate_gate')):
        if name not in args.methods: continue
        fz=make_fuser(mode)
        gate,ep,sel=train_gate(fz,v_fit,fit_users,v_sel,sel_users,E,args,log)
        gates[name]=(fz,gate);info[name]={'best_epoch':ep,'select_NDCG@10':sel,'n_features':fz.n_features(E)}
        torch.save({'state_dict':gate.state_dict(),'mode':mode,'norm':args.norm,'experts':experts,
                    'groups':sorted(fz.groups),'user_mu':mu,'user_sd':sd,'hidden':args.hidden,'dropout':args.dropout},
                   out/f'gate_{mode}.pt')

    # Evaluate one split at a time and free its expert evidence before loading the next,
    # so only one split's [users, experts, items] arrays are in memory (the test view is
    # built only after selection is scored). Evaluation uses no randomness, so results are
    # identical to scoring both splits together; needed for large catalogs (Amazon Beauty).
    rows={};npz={'truth':test_truth,'history_len':np.array([len(h) for h in test_hist]),
                 'target_popularity':pop[test_truth]}
    def evaluate(split,view_,users,truth,hist):
        for name in args.methods:
            if name in weights: sc=np.einsum('e,uei->ui',weights[name],s_hat(view_,users))
            else: sc=full_scores(*gates[name],view_,users)
            m,top=rank_metrics(sc,truth,hist)
            row={'method':name,'split':split,**m,'protocol':protocol,'norm':args.norm,'experts':'+'.join(experts),
                 'drop_groups':'+'.join(args.drop_groups) or 'none'}
            if name in weights: row.update({f'w_{e}':float(w) for e,w in zip(experts,weights[name])})
            if split=='test':
                npz[f'top20_{name}']=top
                if name in gates:
                    fz,gate=gates[name]
                    w_top=np.concatenate([weights_for(fz,gate,view_,users[s:s+256],top[s:s+256,:10]) for s in range(0,nu,256)])
                    w_tgt=np.concatenate([weights_for(fz,gate,view_,users[s:s+256],truth[s:s+256,None]) for s in range(0,nu,256)])[:,0]
                    npz[f'weights_top10_{name}']=w_top.astype(np.float16);npz[f'weights_target_{name}']=w_tgt
                    row.update({f'w_{e}':float(w_top[:,:,k].mean()) for k,e in enumerate(experts)})
            rows[(name,split)]=row
    evaluate('select',v_sel,sel_users,sel_truth,sel_hist)
    del v_fit,v_sel;gc.collect()
    v_test=view('test')
    evaluate('test',v_test,test_users,test_truth,test_hist)
    del v_test;gc.collect()
    rows=[rows[(name,split)] for name in args.methods for split in ('select','test')]
    res=pd.DataFrame(rows)
    res.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(log).to_csv(out/'gate_training_log.csv',index=False)
    np.savez_compressed(out/'test_predictions.npz',**npz)
    stats_file=Path(args.data)/'stats.json'
    cfg={**{k:(list(v) if isinstance(v,tuple) else v) for k,v in vars(args).items()},'protocol':protocol,
         'split_identity':split_identity(args.data),'n_fit_users':int(len(fit_users)),'n_select_users':int(len(sel_users)),
         'data_stats':json.loads(stats_file.read_text()) if stats_file.exists() else None,'info':info}
    (out/'config.json').write_text(json.dumps(cfg,indent=2,default=str))
    print(res[res.split=='test'].drop(columns=['split','protocol']).to_string(index=False,float_format='%.4f'))
    return res


def parser():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data',default='data/processed')
    ap.add_argument('--scores',default='results/scores',help='directory with {expert}_seed{S}_{split}.npy')
    ap.add_argument('--score-seed',type=int,default=42,help='seed in the expert score file names')
    ap.add_argument('--out',default='results/cand_gate/seed42')
    ap.add_argument('--seed',type=int,default=42,help='gate training seed')
    ap.add_argument('--split-seed',type=int,default=0,help='loo2 only: fit/select user partition seed')
    ap.add_argument('--fit-fraction',type=float,default=.7)
    ap.add_argument('--experts',nargs='+',default=list(EXPERTS),choices=EXPERTS)
    ap.add_argument('--methods',nargs='+',default=list(METHODS),choices=METHODS)
    ap.add_argument('--norm',choices=['z','rank'],default='z')
    ap.add_argument('--drop-groups',nargs='*',default=[],choices=GROUPS)
    ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--dropout',type=float,default=.1)
    ap.add_argument('--lr',type=float,default=3e-3)
    ap.add_argument('--weight-decay',type=float,default=1e-5)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--hard-negatives',type=int,default=50,help='top-H unseen items per expert')
    ap.add_argument('--random-negatives',type=int,default=100)
    ap.add_argument('--overwrite',action='store_true')
    return ap

if __name__=='__main__':
    run(parser().parse_args())
