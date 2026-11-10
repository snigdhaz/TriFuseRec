"""MovieLens 1M -> deterministic per-user chronological leave-last-out implicit feedback.

Versions (recorded in stats.json as `preprocess_version`):
  v1  timestamp ties broken by item id (git <= b6701e3). Leaks "next id is larger".
  v2  timestamp ties broken randomly with --seed; adds a `seq` column.
Protocols:
  loo2  train | valid (2nd-to-last) | test (last)                         (default)
  loo3  train | fusion (3rd-to-last) | valid | test — a separate target for fitting
        fusion/gating models, so gate training never uses the selection labels.
NOTE: the split is per-user chronological, NOT a global time split: other users'
training interactions can be later than a user's test event.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


PREPROCESS_VERSION='v2'


def prepare(raw: Path, out: Path, seed: int = 42, protocol: str = 'loo2') -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ratings = pd.read_csv(raw / 'ratings.dat', sep='::', engine='python',
                          names=['user_id','item_id','rating','timestamp'], encoding='latin-1')
    users = pd.read_csv(raw / 'users.dat', sep='::', engine='python',
                        names=['user_id','gender','age','occupation','zip'], encoding='latin-1')
    movies = pd.read_csv(raw / 'movies.dat', sep='::', engine='python',
                         names=['item_id','title','genres'], encoding='latin-1')
    # Explicit rating values are ignored: every event is a positive interaction.
    ratings = ratings.drop(columns='rating').drop_duplicates(['user_id','item_id'], keep='last')
    # Fixed-point k-core filtering ensures each retained user/item has >= 5 events.
    while True:
        before = len(ratings)
        uc = ratings.user_id.value_counts()
        ic = ratings.item_id.value_counts()
        ratings = ratings[ratings.user_id.map(uc).ge(5) & ratings.item_id.map(ic).ge(5)].copy()
        if len(ratings) == before:
            break
    uid = sorted(ratings.user_id.unique().tolist())
    iid = sorted(ratings.item_id.unique().tolist())
    u_map = {u: n for n,u in enumerate(uid)}
    i_map = {i: n for n,i in enumerate(iid)}
    ratings['user_idx'] = ratings.user_id.map(u_map)
    ratings['item_idx'] = ratings.item_id.map(i_map)
    # ~53% of consecutive events share a timestamp (bulk rating sessions). Break ties
    # randomly (seeded) rather than by item id, otherwise sequential models can learn
    # the artificial rule "the next item has a larger id".
    ratings['tie'] = np.random.default_rng(seed).permutation(len(ratings))
    ratings = ratings.sort_values(['user_idx','timestamp','tie']).reset_index(drop=True)
    ratings['seq'] = ratings.groupby('user_idx').cumcount()  # chronological position per user
    pos = ratings.groupby('user_idx').cumcount(ascending=False)
    n_out={'loo2':2,'loo3':3}[protocol]
    if protocol=='loo2' and (out/'fusion.csv').exists():
        raise FileExistsError(f'{out}/fusion.csv is from a loo3 split; use a fresh --out directory')
    parts=[('train',pos>=n_out),('valid',pos==1),('test',pos==0)]
    if protocol=='loo3': parts.insert(1,('fusion',pos==2))
    for name, mask in parts:
        ratings.loc[mask, ['user_idx','item_idx','timestamp','seq']].to_csv(out / f'{name}.csv', index=False)
    users['user_idx'] = users.user_id.map(u_map)
    movies['item_idx'] = movies.item_id.map(i_map)
    users[users.user_idx.notna()].to_csv(out/'users.csv',index=False)
    movies[movies.item_idx.notna()].to_csv(out/'movies.csv',index=False)
    pd.DataFrame({'user_id':uid,'user_idx':range(len(uid))}).to_csv(out/'user_map.csv',index=False)
    pd.DataFrame({'item_id':iid,'item_idx':range(len(iid))}).to_csv(out/'item_map.csv',index=False)
    stats = {'preprocess_version':PREPROCESS_VERSION,'protocol':protocol,'tie_break_seed':seed,
             'users':len(uid),'items':len(iid),'train':int((pos>=n_out).sum()),
             **({'fusion':int((pos==2).sum())} if protocol=='loo3' else {}),
             'valid':int((pos==1).sum()),'test':int((pos==0).sum()),
             'train_mean_history':round(float((pos>=n_out).sum()/len(uid)),2)}
    (out/'stats.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path,default=Path('data/raw/ml-1m'))
    parser.add_argument('--out', type=Path,default=Path('data/processed'))
    parser.add_argument('--seed', type=int, default=42, help='seed for breaking timestamp ties')
    parser.add_argument('--protocol', choices=['loo2','loo3'], default='loo2')
    a=parser.parse_args()
    prepare(a.raw,a.out,a.seed,a.protocol)
