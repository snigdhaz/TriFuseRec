import json
import numpy as np
import pandas as pd
import pytest
import torch
from src.preprocess import prepare
from src import cand_gate as cg


def make_dataset(tmp_path,protocol,n_users=30,n_items=25,seed=0):
    rng=np.random.default_rng(seed);raw=tmp_path/'raw';raw.mkdir()
    rows=[]
    for u in range(1,n_users+1):
        items=rng.choice(np.arange(1,n_items+1),size=rng.integers(8,15),replace=False)
        for k,i in enumerate(items): rows.append(f'{u}::{i}::4::{1000+60*k+int(rng.integers(0,2))}')
    (raw/'ratings.dat').write_text('\n'.join(rows)+'\n')
    (raw/'users.dat').write_text('\n'.join(f'{u}::F::25::3::00000' for u in range(1,n_users+1))+'\n')
    g=['Drama','Comedy','Action']
    (raw/'movies.dat').write_text('\n'.join(f'{i}::M{i} (1999)::{g[i%3]}|{g[(i+1)%3]}' for i in range(1,n_items+1))+'\n')
    data=tmp_path/'data';prepare(raw,data,seed=seed,protocol=protocol)
    return data


def fake_scores(data,scores_dir,seed=42):
    from src.common import read_data, heldout_splits
    _,hist,_,_,nu,ni=read_data(data);rng=np.random.default_rng(1)
    scores_dir.mkdir(exist_ok=True)
    for split,(h,truth) in heldout_splits(data,hist).items():
        for k,e in enumerate(cg.EXPERTS):
            s=rng.normal(size=(nu,ni)).astype(np.float32)
            s[np.arange(nu),truth]+=1.5*(k+1)*rng.random(nu)   # experts informative to different degrees
            np.save(scores_dir/f'{e}_seed{seed}_{split}.npy',s)


def test_normalize_handles_masks_and_zero_variance():
    scores=np.array([[1.,2.,3.,100.],[5.,5.,5.,5.]],dtype=np.float32)
    seen=np.array([[False,False,False,True],[False,False,False,False]])
    z=cg.normalize(scores,seen)
    assert z[0,3]==cg.MASKED
    assert np.isclose(z[0,:3].mean(),0,atol=1e-6) and np.isclose(z[0,:3].std(),1,atol=1e-5)
    assert (z[1]==0).all()                      # zero variance -> no information, no NaN
    lr,_=cg.log_ranks(z)
    assert lr[0,2]==0 and np.isclose(lr[0,3],np.log1p(3))  # best unseen item has rank 0, seen ranks last


def test_history_features_ignore_heldout_timestamps(tmp_path):
    data=make_dataset(tmp_path,'loo3')
    from src.common import read_data
    tr=read_data(data)[0]
    before=cg.history_times(data,tr)
    te=pd.read_csv(data/'test.csv');te['timestamp']+=10**7;te.to_csv(data/'test.csv',index=False)
    after=cg.history_times(data,tr)
    for split in before:   # the test target's timestamp is never a gate input
        for a,b in zip(before[split],after[split]): assert np.array_equal(a,b)


def test_gate_weights_are_a_distribution_per_pair():
    g=cg.GateNet(7,3,hidden=8,dropout=.2);x=torch.randn(4,11,7)
    w=g(x)
    assert (w>=0).all() and torch.allclose(w.sum(-1),torch.ones(4,11))
    g0=cg.GateNet(0,3);assert torch.allclose(g0(torch.zeros(5,0)).sum(-1),torch.ones(5))


def test_candidates_never_include_seen_or_duplicate_target(tmp_path):
    data=make_dataset(tmp_path,'loo2');fake_scores(data,tmp_path/'s')
    from src.common import read_data, heldout_splits
    tr,hist,_,_,nu,ni=read_data(data)
    sp=heldout_splits(data,hist);times=cg.history_times(data,tr)
    pop=np.bincount(tr.item_idx,minlength=ni).astype(float)
    v=cg.View('validation',*sp['validation'],times['validation'],
              [tmp_path/'s'/f'{e}_seed42_validation.npy' for e in cg.EXPERTS],np.log1p(pop),pop<np.median(pop),hard_k=3)
    users=np.arange(nu);c=cg.candidates(v,users,v.truth,5,np.random.default_rng(0))
    assert (c[:,0]==v.truth).all()
    for u in users:
        assert not v.seen[u,c[u]].any()
        assert (c[u,1:]!=v.truth[u]).all()


@pytest.mark.parametrize('protocol',['loo2','loo3'])
def test_end_to_end_smoke(tmp_path,protocol):
    data=make_dataset(tmp_path,protocol);fake_scores(data,tmp_path/'s')
    out=tmp_path/'out'
    args=cg.parser().parse_args(['--data',str(data),'--scores',str(tmp_path/'s'),'--out',str(out),
                                 '--epochs','2','--hard-negatives','3','--random-negatives','5','--batch','8'])
    res=cg.run(args)
    assert set(res.method)==set(cg.METHODS) and set(res.split)=={'select','test'}
    cfg=json.loads((out/'config.json').read_text())
    assert cfg['protocol']==protocol and len(cfg['split_identity'])==12
    pred=np.load(out/'test_predictions.npz')
    w=pred['weights_top10_candidate_gate'].astype(np.float32)
    assert np.allclose(w.sum(-1),1,atol=1e-2)
    wu=pred['weights_top10_per_user_gate'].astype(np.float32)
    assert np.allclose(wu,wu[:,:1,:],atol=1e-3)   # per-user gate: same weights for every item
    with pytest.raises(SystemExit): cg.run(args)  # never overwrite without --overwrite
    # Ablation flags: drop all inputs and one expert still runs.
    a2=cg.parser().parse_args(['--data',str(data),'--scores',str(tmp_path/'s'),'--out',str(tmp_path/'o2'),
                               '--epochs','1','--hard-negatives','2','--random-negatives','3','--norm','rank',
                               '--experts','sasrec','ngcf','--drop-groups',*cg.GROUPS])
    assert len(cg.run(a2))>0
