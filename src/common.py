"""Shared data, reproducibility, ranking metrics, and negative sampling."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_data(path):
    path=Path(path)
    train=pd.read_csv(path/'train.csv')
    valid=pd.read_csv(path/'valid.csv')
    test=pd.read_csv(path/'test.csv')
    n_users=int(pd.read_csv(path/'user_map.csv').shape[0])
    n_items=int(pd.read_csv(path/'item_map.csv').shape[0])
    histories=[[] for _ in range(n_users)]
    # `seq` is the chronological order fixed in preprocessing (v2+: timestamp ties broken
    # randomly). v1 splits have no `seq`; fall back to their original item-id tie-break so
    # archived results stay reproducible.
    order=['user_idx','seq'] if 'seq' in train.columns else ['user_idx','timestamp','item_idx']
    for r in train.sort_values(order).itertuples():
        histories[int(r.user_idx)].append(int(r.item_idx))
    val_items=valid.sort_values('user_idx').item_idx.to_numpy(dtype=np.int64)
    test_items=test.sort_values('user_idx').item_idx.to_numpy(dtype=np.int64)
    assert len(val_items)==len(test_items)==n_users
    return train,histories,val_items,test_items,n_users,n_items


HELDOUT_FILES=(('fusion','fusion.csv'),('validation','valid.csv'),('test','test.csv'))


def heldout_splits(path, histories):
    """{split: (history, target_items)} in chronological order.

    Each split's history is train + every EARLIER held-out item of that user, so
    all models see exactly the same history and rank the same candidate set
    (all items minus that history). `fusion` exists only for protocol loo3.
    """
    path=Path(path);out={};hist=[list(h) for h in histories]
    for name,fname in HELDOUT_FILES:
        if not (path/fname).exists(): continue
        items=pd.read_csv(path/fname).sort_values('user_idx').item_idx.to_numpy(dtype=np.int64)
        assert len(items)==len(hist)
        out[name]=(hist,items)
        hist=[h+[int(i)] for h,i in zip(hist,items)]
    return out


def negative_samples(train, n_items, n_neg, seed):
    """Sample unobserved movies for pairwise-positive pointwise models.

    Exclude ALL held-out positives by passing a full interaction table if available;
    for train-only baseline this uses train positives and may sample held-outs at low rate.
    """
    rng=np.random.default_rng(seed)
    u=train.user_idx.to_numpy(np.int64);i=train.item_idx.to_numpy(np.int64)
    neg_u=np.repeat(u,n_neg);neg_i=sample_unseen(neg_u,seen_matrix(train,n_items),rng)
    return (np.concatenate([u,neg_u]),np.concatenate([i,neg_i]),
            np.concatenate([np.ones(len(u)),np.zeros(len(neg_u))]).astype(np.float32))


def seen_matrix(train, n_items):
    seen=np.zeros((int(train.user_idx.max())+1,n_items),dtype=bool)
    seen[train.user_idx.to_numpy(),train.item_idx.to_numpy()]=True
    return seen


def sample_unseen(users, seen, rng):
    """One uniformly random item per entry of `users` that the user has not interacted with."""
    j=rng.integers(seen.shape[1],size=len(users))
    bad=seen[users,j]
    while bad.any():
        j[bad]=rng.integers(seen.shape[1],size=int(bad.sum()))
        bad=seen[users,j]
    return j


def rank_metrics(scores, truths, seen, ks=(10,20), chunk=2048):
    """scores [users,items], truth [users], seen list of sets/lists to exclude.

    Every held-out target must remain an eligible candidate. Users are ranked in
    chunks so large catalogs (e.g. Amazon Beauty, 22k x 12k) fit in memory.
    """
    n_users,n_items=np.shape(scores)
    truths=np.asarray(truths)
    assert len(truths)==len(seen)==n_users
    max_k=min(max(ks),n_items)
    order=np.empty((n_users,max_k),dtype=np.int64)
    for start in range(0,n_users,chunk):
        end=min(start+chunk,n_users)
        block=np.array(scores[start:end],copy=True)
        for row,u in enumerate(range(start,end)):
            items=seen[u]
            if int(truths[u]) in items:
                raise ValueError(f'Target item is in masked history for user {u}')
            block[row,list(items)]=-np.inf
        # Full stable sort keeps deterministic tie handling.
        order[start:end]=np.argsort(-block,axis=1,kind='stable')[:,:max_k]
    metrics={}
    for k in ks:
        matches=(order[:,:min(k,n_items)]==truths[:,None])
        hits=matches.any(axis=1)
        ranks=np.argmax(matches,axis=1)+1
        ndcg=np.where(hits,1.0/np.log2(ranks+1),0.)
        metrics[f'Recall@{k}']=float(np.mean(hits))
        metrics[f'NDCG@{k}']=float(np.mean(ndcg))
    return metrics,order


def evaluate_model(score_fn, histories, truths, extra_seen=None, batch_size=128):
    all_scores=[]
    n=len(histories)
    for start in range(0,n,batch_size):
        users=np.arange(start,min(start+batch_size,n))
        all_scores.append(score_fn(users))
    scores=np.concatenate(all_scores)
    seen=[set(h) for h in histories]
    if extra_seen is not None:
        for u,it in enumerate(extra_seen): seen[u].add(int(it))
    metrics,order=rank_metrics(scores,truths,seen)
    return metrics,order
