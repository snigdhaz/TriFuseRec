"""Per-user metrics and paired bootstrap confidence intervals."""
import numpy as np


def per_user_metrics(top,truth,k=10):
    """Per-user hit and NDCG at k from a ranked top list [users, >=k]."""
    m=top[:,:k]==np.asarray(truth)[:,None]
    hit=m.any(1)
    ndcg=np.where(hit,1/np.log2(np.argmax(m,1)+2),0.)
    return hit.astype(float),ndcg


def paired_bootstrap(a,b,n_boot=10000,seed=0,alpha=.05):
    """Paired bootstrap over users for mean(a - b).

    a and b must be per-user values for the SAME users and the SAME targets.
    Returns mean difference, (1-alpha) percentile CI, and a two-sided bootstrap
    p-value (share of resampled means on the other side of 0, doubled).
    """
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape: raise ValueError('paired bootstrap needs aligned per-user arrays')
    d=a-b;rng=np.random.default_rng(seed);n=len(d)
    means=np.empty(n_boot)
    for s in range(0,n_boot,500):
        idx=rng.integers(n,size=(min(500,n_boot-s),n))
        means[s:s+len(idx)]=d[idx].mean(1)
    lo,hi=np.quantile(means,[alpha/2,1-alpha/2])
    p=2*min((means<=0).mean(),(means>=0).mean())
    return {'mean_diff':float(d.mean()),'ci_low':float(lo),'ci_high':float(hi),'p_value':float(min(p,1.)),'n_users':int(n)}
