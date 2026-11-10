"""Regression tests for the evaluation protocol (Phase 2 audit)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.common import rank_metrics, negative_samples, sample_unseen, seen_matrix
from src.preprocess import prepare

DATA=Path('data/processed')


def write_raw(raw):
    """Tiny MovieLens-format dataset: 6 users x 8 movies, many tied timestamps."""
    raw.mkdir()
    rows=[]
    for u in range(1,7):
        for k,i in enumerate(range(1,9)):
            rows.append(f'{u}::{i}::4::{1000+(k//3)}')   # 3 ratings per timestamp
    (raw/'ratings.dat').write_text('\n'.join(rows)+'\n',encoding='latin-1')
    (raw/'users.dat').write_text('\n'.join(f'{u}::M::25::1::00000' for u in range(1,7))+'\n',encoding='latin-1')
    (raw/'movies.dat').write_text('\n'.join(f'{i}::Movie {i} (2000)::Drama' for i in range(1,9))+'\n',encoding='latin-1')


def test_preprocess_is_deterministic_for_a_seed(tmp_path):
    write_raw(tmp_path/'raw')
    prepare(tmp_path/'raw',tmp_path/'a',seed=7);prepare(tmp_path/'raw',tmp_path/'b',seed=7)
    for f in ('train','valid','test'):
        assert (tmp_path/'a'/f'{f}.csv').read_bytes()==(tmp_path/'b'/f'{f}.csv').read_bytes()


def test_tie_break_is_not_item_id_order(tmp_path):
    write_raw(tmp_path/'raw')
    prepare(tmp_path/'raw',tmp_path/'p',seed=0)
    full=pd.concat([pd.read_csv(tmp_path/'p'/f'{s}.csv') for s in ('train','valid','test')])
    full=full.sort_values(['user_idx','seq'])
    # Within each tied timestamp group, order must not always be ascending item id.
    ascending=[g.item_idx.is_monotonic_increasing for _,g in full.groupby(['user_idx','timestamp'])]
    assert not all(ascending)
    # Different seeds may change which tied item is held out, but never the timestamps.
    prepare(tmp_path/'raw',tmp_path/'q',seed=1)
    for s in ('valid','test'):
        a=pd.read_csv(tmp_path/'p'/f'{s}.csv');b=pd.read_csv(tmp_path/'q'/f'{s}.csv')
        assert (a.timestamp.to_numpy()==b.timestamp.to_numpy()).all()


@pytest.mark.skipif(not (DATA/'train.csv').exists(),reason='processed data not available')
def test_processed_split_is_per_user_chronological():
    tr=pd.read_csv(DATA/'train.csv');va=pd.read_csv(DATA/'valid.csv');te=pd.read_csv(DATA/'test.csv')
    assert 'seq' in tr.columns, 'split predates the tie-break fix (no seq column)'
    last=tr.groupby('user_idx').agg(seq=('seq','max'),ts=('timestamp','max'))
    va=va.set_index('user_idx').loc[last.index];te=te.set_index('user_idx').loc[last.index]
    assert (last.seq<va.seq).all() and (va.seq<te.seq).all()
    assert (last.ts<=va.timestamp).all() and (va.timestamp<=te.timestamp).all()
    # A held-out movie never appears in that user's training history.
    pairs=set(zip(tr.user_idx,tr.item_idx))
    assert not any((u,i) in pairs for u,i in zip(va.index,va.item_idx))
    assert not any((u,i) in pairs for u,i in zip(te.index,te.item_idx))
    assert (va.item_idx.to_numpy()!=te.item_idx.to_numpy()).all()


def test_rank_metrics_never_recommends_masked_items():
    rng=np.random.default_rng(0)
    scores=rng.normal(size=(5,30));seen=[set(rng.choice(29,10,replace=False).tolist()) for _ in range(5)]
    truth=np.array([29]*5)
    _,order=rank_metrics(scores,truth,seen)
    for u in range(5): assert not set(order[u].tolist())&seen[u]
    with pytest.raises(ValueError): rank_metrics(scores,truth,[{29}]+seen[1:])


def test_negative_sampling_never_returns_a_known_positive():
    train=pd.DataFrame({'user_idx':[0,0,0,1,1],'item_idx':[0,1,2,3,4]})
    us,its,ys=negative_samples(train,6,n_neg=50,seed=0)
    seen=seen_matrix(train,6)
    assert not seen[us[ys==0],its[ys==0]].any()
    assert sample_unseen(np.zeros(1000,dtype=int),seen,np.random.default_rng(1)).min()>=3


def test_loo3_split_and_heldout_histories(tmp_path):
    from src.common import read_data, heldout_splits
    write_raw(tmp_path/'raw')
    out=tmp_path/'p3';stats=prepare(tmp_path/'raw',out,seed=0,protocol='loo3')
    assert stats['protocol']=='loo3' and stats['fusion']==stats['users']
    pd.DataFrame({'user_id':range(6),'user_idx':range(6)}).to_csv(out/'user_map.csv',index=False)
    _,hist,_,_,nu,_=read_data(out)
    splits=heldout_splits(out,hist)
    assert list(splits)==['fusion','validation','test']
    f,v,t=(splits[s] for s in splits)
    for u in range(nu):
        # Each split's history = train + all earlier held-out items, in order.
        assert f[0][u]==hist[u]
        assert v[0][u]==hist[u]+[int(f[1][u])]
        assert t[0][u]==hist[u]+[int(f[1][u]),int(v[1][u])]
        assert t[1][u] not in t[0][u]
    # Re-using a loo3 directory for loo2 would silently pick up fusion.csv: refuse.
    with pytest.raises(FileExistsError): prepare(tmp_path/'raw',out,seed=0,protocol='loo2')


def test_loo2_test_history_contains_validation_item():
    from src.common import heldout_splits
    if not (DATA/'train.csv').exists(): pytest.skip('processed data not available')
    from src.common import read_data
    _,hist,valid,test,_,_=read_data(DATA)
    splits=heldout_splits(DATA,hist)
    assert list(splits)==['validation','test']
    th=splits['test'][0]
    assert all(th[u][-1]==valid[u] and len(th[u])==len(hist[u])+1 for u in range(0,len(hist),97))
