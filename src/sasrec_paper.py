"""Replicate SASRec (Kang & McAuley, ICDM 2018) on MovieLens-1M under the paper's protocol.

Paper protocol: leave-one-out per user (last item = test, second-to-last = validation,
rest = train). Each held-out item is ranked against 100 random items the user has not
interacted with; metrics Hit@10 and NDCG@10. Paper result on ML-1M: Hit@10 0.8245,
NDCG@10 0.5905.

Paper / reference-code settings used here: d=50, 2 blocks, 1 head, max_len 200, dropout 0.2,
learned positions, embeddings scaled by sqrt(d), pre-LayerNorm queries, shared item
embedding for prediction, binary cross-entropy with one random negative per position,
Adam(lr 1e-3, betas .9/.98), batch 128, 200 epochs, one training window per user
(the last max_len items of the training history).

python -m src.sasrec_paper --tag v2_random_ties                 # train + evaluate
python -m src.sasrec_paper --data data/splits/v1_itemid --tag v1_itemid_ties
python -m src.sasrec_paper --score-prefix results/scores/sasrec_seed42 --tag main_sasrec
    # evaluate an existing model's saved score matrices under the paper protocol

Writes results/sasrec_paper/{tag}.csv (sampled-100 and full-ranking metrics for
validation and test), {tag}_log.json (validation curve) and {tag}.pt (best weights).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from .common import seed_everything, read_data, heldout_splits, rank_metrics, seen_matrix, sample_unseen

PAPER={'HR@10':0.8245,'NDCG@10':0.5905}


class Dropout(nn.Module):
    """Same as nn.Dropout; uses rand() masks, which are much faster than bernoulli on Windows CPU."""
    def __init__(self,p):
        super().__init__();self.p=p
    def forward(self,x):
        if not self.training or self.p==0: return x
        return x*(torch.rand_like(x)>=self.p)/(1-self.p)


class PaperSASRec(nn.Module):
    """SASRec as in the authors' reference implementation."""
    def __init__(self,n_items,d=50,max_len=200,blocks=2,heads=1,dropout=.2):
        super().__init__()
        assert d%heads==0
        self.d=d;self.heads=heads;self.max_len=max_len
        self.item=nn.Embedding(n_items+1,d,padding_idx=0)
        self.pos=nn.Embedding(max_len,d)
        self.drop=Dropout(dropout)
        self.attn_ln=nn.ModuleList([nn.LayerNorm(d,eps=1e-8) for _ in range(blocks)])
        self.qkv=nn.ModuleList([nn.Linear(d,3*d) for _ in range(blocks)])
        self.attn_out=nn.ModuleList([nn.Linear(d,d) for _ in range(blocks)])
        self.ffn_ln=nn.ModuleList([nn.LayerNorm(d,eps=1e-8) for _ in range(blocks)])
        self.ffn=nn.ModuleList([nn.Sequential(nn.Linear(d,d),Dropout(dropout),nn.ReLU(),nn.Linear(d,d),Dropout(dropout))
                                for _ in range(blocks)])
        self.last_ln=nn.LayerNorm(d,eps=1e-8)
        for name,p in self.named_parameters():
            if p.dim()>1: nn.init.xavier_normal_(p)
        with torch.no_grad(): self.item.weight[0].zero_()

    def attention(self,i,q_in,kv_in,blocked):
        b,l,d=q_in.shape;h=self.heads
        wq,wk,wv=self.qkv[i].weight.chunk(3);bq,bk,bv=self.qkv[i].bias.chunk(3)
        q=F.linear(q_in,wq,bq).view(b,l,h,d//h).transpose(1,2)
        k=F.linear(kv_in,wk,bk).view(b,l,h,d//h).transpose(1,2)
        v=F.linear(kv_in,wv,bv).view(b,l,h,d//h).transpose(1,2)
        w=(q@k.transpose(-1,-2))/(d//h)**.5
        w=self.drop(w.masked_fill(blocked,float('-inf')).softmax(-1))
        return self.attn_out[i]((w@v).transpose(1,2).reshape(b,l,d))

    def hidden(self,seq):
        """seq [b, l] of item ids + 1 (0 = padding, left-padded) -> [b, l, d]."""
        l=seq.shape[1]
        x=self.item(seq)*self.d**.5+self.pos(torch.arange(l,device=seq.device))[None]
        keep=seq.ne(0).unsqueeze(-1)
        x=self.drop(x)*keep
        # Causal mask only, as in the reference code; padded positions are zeroed after each block.
        blocked=torch.triu(torch.ones(l,l,dtype=torch.bool,device=seq.device),diagonal=1)
        for i in range(len(self.qkv)):
            q=self.attn_ln[i](x)
            x=q+self.attention(i,q,x,blocked)
            x=self.ffn_ln[i](x)
            x=(x+self.ffn[i](x))*keep
        return self.last_ln(x)

    def scores(self,seq):
        """Scores of every item for the next position: [b, n_items]."""
        return self.hidden(seq)[:,-1]@self.item.weight[1:].T


def left_pad(histories,max_len):
    out=np.zeros((len(histories),max_len),dtype=np.int64)
    for r,h in enumerate(histories):
        h=h[-max_len:]
        if h: out[r,-len(h):]=np.asarray(h)+1
    return out


def training_windows(histories,max_len):
    """Reference sampler: input = train[:-1], targets = train[1:], last max_len positions."""
    seq=left_pad([h[:-1] for h in histories],max_len)
    pos=left_pad([h[1:] for h in histories],max_len)
    return seq,pos


@torch.no_grad()
def full_scores(model,histories,max_len,batch=256):
    model.eval();out=[]
    for s in range(0,len(histories),batch):
        out.append(model.scores(torch.from_numpy(left_pad(histories[s:s+batch],max_len))).numpy())
    return np.concatenate(out)


def sampled_metrics(scores,truths,histories,n_neg=100,seed=0,k=10):
    """Paper protocol: rank the target among n_neg random items the user never interacted with.

    Negatives exclude the user's whole history and the target. Ties count against the target.
    Returns {'HR@k','NDCG@k'}.
    """
    rng=np.random.default_rng(seed)
    n_users,n_items=scores.shape
    seen=np.zeros((n_users,n_items),dtype=bool)
    for u,h in enumerate(histories): seen[u,h]=True
    seen[np.arange(n_users),truths]=True
    # n_neg distinct negatives per user, sampled uniformly without replacement.
    neg=np.empty((n_users,n_neg),dtype=np.int64)
    for u in range(n_users):
        pool=np.flatnonzero(~seen[u])
        neg[u]=rng.choice(pool,size=n_neg,replace=False)
    target=scores[np.arange(n_users),truths][:,None]
    rank=(scores[np.arange(n_users)[:,None],neg]>=target).sum(1)  # 0 = ranked first
    hit=rank<k
    return {f'HR@{k}':float(hit.mean()),f'NDCG@{k}':float(np.where(hit,1/np.log2(rank+2),0).mean())}


def evaluate(scores,truths,histories,repeats,seed):
    """Sampled-100 metrics averaged over `repeats` negative draws, plus full-ranking metrics."""
    draws=[sampled_metrics(scores,truths,histories,seed=seed+r) for r in range(repeats)]
    row={m:float(np.mean([d[m] for d in draws])) for m in draws[0]}
    row.update({m+'_std':float(np.std([d[m] for d in draws])) for m in draws[0]})
    full,_=rank_metrics(scores,truths,histories)
    row.update({'full_'+m:v for m,v in full.items()})
    return row


def train(args,out):
    seed_everything(args.seed);torch.set_num_threads(args.threads)
    tr,histories,valid,test,nu,ni=read_data(args.data)
    splits=heldout_splits(args.data,histories)
    model=PaperSASRec(ni,d=args.dim,max_len=args.max_len,blocks=args.blocks,heads=args.heads,dropout=args.dropout)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.98))
    seq_np,pos_np=training_windows(histories,args.max_len)
    seen=seen_matrix(tr,ni);rng=np.random.default_rng(args.seed)
    rows_u,cols=np.nonzero(pos_np)
    ckpt=out/f'{args.tag}_ckpt.pt';log=[];best={'NDCG@10':-1.,'epoch':0};start=1
    if args.resume and ckpt.exists():
        c=torch.load(ckpt,weights_only=False)
        model.load_state_dict(c['model']);opt.load_state_dict(c['opt']);log=c['log'];best=c['best']
        rng.bit_generator.state=c['rng'];torch.set_rng_state(c['torch_rng']);start=c['epoch']+1
        print(f'resumed after epoch {c["epoch"]}',flush=True)
    vh,vt=splits['validation'];t0=time.time()
    for epoch in range(start,args.epochs+1):
        model.train()
        neg_np=np.zeros_like(pos_np)
        neg_np[rows_u,cols]=sample_unseen(rows_u,seen,rng)+1  # fresh negatives every epoch
        perm=rng.permutation(nu);losses=[]
        for s in range(0,nu,args.batch):
            idx=perm[s:s+args.batch]
            seq,pos,neg=(torch.from_numpy(a[idx]) for a in (seq_np,pos_np,neg_np))
            h=model.hidden(seq)
            pl=(h*model.item(pos)).sum(-1);nl=(h*model.item(neg)).sum(-1)
            m=pos.ne(0)
            loss=(F.binary_cross_entropy_with_logits(pl[m],torch.ones_like(pl[m]))
                  +F.binary_cross_entropy_with_logits(nl[m],torch.zeros_like(nl[m])))
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        msg=f'epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} ({time.time()-t0:.0f}s)'
        if epoch%args.eval_every==0 or epoch==args.epochs:
            # Model selection on validation with one fixed negative draw (paper metric).
            v=sampled_metrics(full_scores(model,vh,args.max_len),vt,vh,seed=args.seed)
            log.append({'epoch':epoch,'loss':float(np.mean(losses)),**{'val_'+k:x for k,x in v.items()}})
            msg+=f" val HR@10={v['HR@10']:.4f} NDCG@10={v['NDCG@10']:.4f}"
            if v['NDCG@10']>best['NDCG@10']:
                best={'NDCG@10':v['NDCG@10'],'epoch':epoch};torch.save(model.state_dict(),out/f'{args.tag}.pt')
            tmp=ckpt.with_suffix('.tmp')
            torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'log':log,'best':best,'epoch':epoch,
                        'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()},tmp)
            tmp.replace(ckpt)
        print(msg,flush=True)
    print(f"best epoch {best['epoch']} (val sampled NDCG@10={best['NDCG@10']:.4f})",flush=True)
    model.load_state_dict(torch.load(out/f'{args.tag}.pt'))
    (out/f'{args.tag}_log.json').write_text(json.dumps({'config':vars(args)|{'data':str(args.data)},'evals':log},indent=1))
    return {split:full_scores(model,h,args.max_len) for split,(h,_) in splits.items()},splits,best['epoch']


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.score_prefix:
        _,histories,*_=read_data(args.data);splits=heldout_splits(args.data,histories)
        scores={s:np.load(f'{args.score_prefix}_{s}.npy') for s in splits};best_epoch=None
    else:
        scores,splits,best_epoch=train(args,out)
    rows=[]
    for split,(h,t) in splits.items():
        row={'tag':args.tag,'split':split,'best_epoch':best_epoch,**evaluate(scores[split],t,h,args.repeats,args.seed)}
        rows.append(row)
        print(f"{split.upper()}: sampled-100 HR@10={row['HR@10']:.4f}+-{row['HR@10_std']:.4f} "
              f"NDCG@10={row['NDCG@10']:.4f}+-{row['NDCG@10_std']:.4f} | full-ranking "
              f"Recall@10={row['full_Recall@10']:.4f} NDCG@10={row['full_NDCG@10']:.4f}",flush=True)
    print(f"paper (test, sampled-100): HR@10={PAPER['HR@10']} NDCG@10={PAPER['NDCG@10']}")
    pd.DataFrame(rows).to_csv(out/f'{args.tag}.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('data/processed'))
    p.add_argument('--out',default='results/sasrec_paper')
    p.add_argument('--tag',default='v2_random_ties')
    p.add_argument('--score-prefix',help='evaluate {prefix}_{split}.npy score matrices instead of training')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repeats',type=int,default=5,help='negative draws averaged for the sampled metrics')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--dim',type=int,default=50)
    p.add_argument('--blocks',type=int,default=2)
    p.add_argument('--heads',type=int,default=1)
    p.add_argument('--max-len',type=int,default=200)
    p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--resume',action='store_true')
    main(p.parse_args())
