"""Amazon Beauty (McAuley 2014, 5-core) -> chronological leave-two-out implicit-feedback data.

Same protocol as src/preprocess.py (MovieLens-1M) so results are comparable:
  * every review is a positive interaction; star ratings are ignored
  * duplicate (user, item) pairs keep the latest event
  * fixed-point 5-core filtering on users and items
  * per user, ordered by timestamp: last -> test, second-to-last -> valid, rest -> train
  * review times are day-granular (~44% of users review their last two items on the same day);
    same-day ties are broken by a seeded random key (--tie-break random, default). Breaking them by
    item_idx (--tie-break item, the ML-1M rule) lets SASRec learn "next item has a larger ID".

Beauty has no user demographics, so DeepFM user fields are derived from TRAINING
interactions only (activity level, typical price tier, favourite sub-category).
Item fields come from product metadata (brand, price tier, category multi-hot).

Download (into data/raw/amazon-beauty/):
  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
  http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
"""
import argparse
import ast
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MIN_BRAND_ITEMS = 5      # rarer brands share one "other" code
MIN_CATEGORY_ITEMS = 5   # rarer category nodes are dropped from the multi-hot
PRICE_BINS = 10
ACTIVITY_BINS = 5


def read_reviews(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            rows.append((r['reviewerID'], r['asin'], int(r['unixReviewTime'])))
    return pd.DataFrame(rows, columns=['user_id', 'item_id', 'timestamp'])


def read_meta(path: Path, keep: set) -> pd.DataFrame:
    """Metadata lines are Python dict literals (not strict JSON); parse safely."""
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line[10:20] not in keep and "'asin'" in line[:10]:
                # fast path: {'asin': 'XXXXXXXXXX', ...  -> asin sits at [10:20]
                continue
            r = ast.literal_eval(line)
            if r['asin'] not in keep:
                continue
            paths = r.get('categories') or []
            nodes = sorted({c for p in paths for c in p if c != 'Beauty'})
            sub = next((p[1] for p in paths if len(p) > 1 and p[0] == 'Beauty'), '')
            rows.append({'item_id': r['asin'], 'title': r.get('title', ''),
                         'brand': (r.get('brand') or '').strip(), 'price': r.get('price', np.nan),
                         'subcategory': sub, 'categories': '|'.join(nodes)})
    return pd.DataFrame(rows)


def k_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    while True:
        before = len(df)
        uc = df.user_id.value_counts()
        ic = df.item_id.value_counts()
        df = df[df.user_id.map(uc).ge(k) & df.item_id.map(ic).ge(k)].copy()
        if len(df) == before:
            return df


def item_features(items: pd.DataFrame) -> pd.DataFrame:
    """Integer codes; 0 is reserved for unknown / rare in every field."""
    items = items.copy()
    items['brand'] = items.brand.fillna('')
    counts = items.loc[items.brand.ne(''), 'brand'].value_counts()
    frequent = sorted(counts[counts >= MIN_BRAND_ITEMS].index)
    brand_map = {b: n + 2 for n, b in enumerate(frequent)}            # 0 unknown, 1 rare
    items['brand_code'] = [0 if b == '' else brand_map.get(b, 1) for b in items.brand]
    price = pd.to_numeric(items.price, errors='coerce')
    known = price.notna()
    items['price_code'] = 0
    items.loc[known, 'price_code'] = pd.qcut(price[known], PRICE_BINS, labels=False,
                                             duplicates='drop').astype(int) + 1
    items['subcat_code'] = 0
    subs = sorted(s for s in items.subcategory.dropna().unique() if s)
    sub_map = {s: n + 1 for n, s in enumerate(subs)}
    items['subcat_code'] = items.subcategory.map(sub_map).fillna(0).astype(int)
    node_counts = Counter(c for cats in items.categories for c in cats.split('|') if c)
    kept = {c for c, n in node_counts.items() if n >= MIN_CATEGORY_ITEMS}
    items['categories'] = ['|'.join(c for c in cats.split('|') if c in kept) for cats in items.categories]
    return items


def user_features(train: pd.DataFrame, items: pd.DataFrame, n_users: int) -> pd.DataFrame:
    """Derived from training interactions only, so no validation/test leakage."""
    t = train.merge(items[['item_idx', 'price_code', 'subcat_code']], on='item_idx', how='left')
    length = t.groupby('user_idx').size().reindex(range(n_users), fill_value=0)
    # Equal lengths always share a bucket; heavy ties at 3-4 events may merge some quantile bins.
    activity = pd.qcut(length, ACTIVITY_BINS, labels=False, duplicates='drop').astype(int)

    def mode_nonzero(s):
        s = s[s > 0]
        return int(s.value_counts().sort_index().idxmax()) if len(s) else 0

    price_pref = t.groupby('user_idx').price_code.apply(
        lambda s: int(np.median(s[s > 0])) if (s > 0).any() else 0).reindex(range(n_users), fill_value=0)
    subcat_pref = t.groupby('user_idx').subcat_code.apply(mode_nonzero).reindex(range(n_users), fill_value=0)
    return pd.DataFrame({'user_idx': range(n_users), 'train_length': length.to_numpy(),
                         'activity_code': activity.to_numpy(), 'price_code': price_pref.to_numpy(),
                         'subcat_code': subcat_pref.to_numpy()})


def prepare(raw: Path, out: Path, tie_break: str = 'random', seed: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    reviews = read_reviews(raw / 'reviews_Beauty_5.json.gz')
    raw_events = len(reviews)
    reviews = reviews.sort_values('timestamp').drop_duplicates(['user_id', 'item_id'], keep='last')
    reviews = k_core(reviews, 5)

    uid = sorted(reviews.user_id.unique().tolist())
    iid = sorted(reviews.item_id.unique().tolist())
    u_map = {u: n for n, u in enumerate(uid)}
    i_map = {i: n for n, i in enumerate(iid)}
    reviews['user_idx'] = reviews.user_id.map(u_map)
    reviews['item_idx'] = reviews.item_id.map(i_map)
    if tie_break == 'random':
        reviews = reviews.sort_values(['user_idx', 'item_idx']).reset_index(drop=True)
        reviews['tie'] = np.random.default_rng(seed).permutation(len(reviews))
        reviews = reviews.sort_values(['user_idx', 'timestamp', 'tie']).drop(columns='tie').reset_index(drop=True)
    else:
        reviews = reviews.sort_values(['user_idx', 'timestamp', 'item_idx']).reset_index(drop=True)
    reviews['seq'] = reviews.groupby('user_idx').cumcount()  # read_data orders histories by this
    pos = reviews.groupby('user_idx').cumcount(ascending=False)
    splits = {'train': pos >= 2, 'valid': pos == 1, 'test': pos == 0}
    for name, mask in splits.items():
        reviews.loc[mask, ['user_idx', 'item_idx', 'timestamp', 'seq']].to_csv(out / f'{name}.csv', index=False)
    train = reviews.loc[splits['train'], ['user_idx', 'item_idx']]

    meta = read_meta(raw / 'meta_Beauty.json.gz', set(iid))
    items = pd.DataFrame({'item_id': iid, 'item_idx': range(len(iid))}).merge(meta, on='item_id', how='left')
    for col in ('title', 'brand', 'subcategory', 'categories'):
        items[col] = items[col].fillna('')
    items = item_features(items)
    items.to_csv(out / 'items.csv', index=False)
    users = user_features(train, items, len(uid))
    users.insert(0, 'user_id', uid)
    users.to_csv(out / 'users.csv', index=False)
    pd.DataFrame({'user_id': uid, 'user_idx': range(len(uid))}).to_csv(out / 'user_map.csv', index=False)
    pd.DataFrame({'item_id': iid, 'item_idx': range(len(iid))}).to_csv(out / 'item_map.csv', index=False)

    n_train = int(splits['train'].sum())
    lengths = users.train_length
    stats = {'dataset': 'amazon-beauty-5core-2014', 'tie_break': tie_break, 'raw_events': raw_events,
             'users': len(uid), 'items': len(iid), 'interactions': len(reviews),
             'train': n_train, 'valid': int(splits['valid'].sum()), 'test': int(splits['test'].sum()),
             'density_pct': round(100 * len(reviews) / (len(uid) * len(iid)), 4),
             'train_mean_history': round(n_train / len(uid), 2),
             'train_median_history': float(lengths.median()),
             'train_history_p90': float(lengths.quantile(.9)),
             'items_with_metadata': int(items.title.ne('').sum()),
             'items_with_brand': int(items.brand_code.gt(0).sum()),
             'items_with_price': int(items.price_code.gt(0).sum()),
             'brand_codes': int(items.brand_code.max() + 1),
             'category_nodes': len({c for cats in items.categories for c in cats.split('|') if c}),
             'subcategories': int(items.subcat_code.max())}
    (out / 'stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/raw/amazon-beauty'))
    parser.add_argument('--out', type=Path, default=Path('data/processed_beauty'))
    parser.add_argument('--tie-break', choices=['random', 'item'], default='random')
    parser.add_argument('--seed', type=int, default=0, help='seed for random same-day tie-breaking')
    a = parser.parse_args()
    prepare(a.raw, a.out, a.tie_break, a.seed)
