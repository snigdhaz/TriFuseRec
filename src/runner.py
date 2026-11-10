"""Multi-seed experiment runner (Phase 3).

    python -m src.runner run configs/main_loo3.json --dry-run      # print the plan
    python -m src.runner run configs/main_loo3.json --seeds 42      # one seed
    python -m src.runner run configs/main_loo3.json                 # all seeds in the config
    python -m src.runner aggregate configs/main_loo3.json           # mean ± std + paired bootstrap
    python -m src.runner run configs/ablations_loo3.json --stages ablation   # gate ablations
    python -m src.runner ablations configs/ablations_loo3.json      # ablation table

Layout:
  data/splits/{protocol}_tie{preprocess_seed}/      one split per preprocessing seed
  runs/{name}/seed{k}/                             train.py outputs for training seed k
  runs/{name}/seed{k}/fusion/                      cand_gate outputs
  runs/{name}/seed{k}/manifest_{stage}.json        config hash, split identity, git commit, status
  runs/{name}/aggregate/                           summary tables

Safety:
  * A stage whose manifest says `done` with the same config hash is skipped (resume).
  * A stage marked `running` (interrupted) is re-run from the start of that stage.
  * A `done` stage with a DIFFERENT config is never overwritten unless --force is given.
  * The preprocessing seed is fixed per experiment, so all training seeds share the
    same held-out targets and per-user results are paired across seeds and methods.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .stats import per_user_metrics, paired_bootstrap

MODELS=('sasrec','ngcf','deepfm')


def chash(obj):
    return hashlib.sha1(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:12]


def git_commit():
    try:
        sha=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        dirty=subprocess.run(['git','status','--porcelain','src'],capture_output=True,text=True).stdout.strip()
        return sha+('-dirty' if dirty else '')
    except Exception: return 'unknown'


def split_dir(cfg):
    p=cfg['preprocess']
    return Path(cfg.get('splits_root','data/splits'))/f"{p.get('protocol','loo2')}_tie{p['seed']}"


def run_dir(cfg,seed): return Path(cfg.get('runs_root','runs'))/cfg['name']/f'seed{seed}'


def flags(d):
    out=[]
    for k,v in d.items():
        k='--'+k.replace('_','-')
        if isinstance(v,bool):
            if v: out.append(k)
        elif isinstance(v,(list,tuple)): out+=[k,*map(str,v)]
        else: out+=[k,str(v)]
    return out


class Stage:
    def __init__(self,name,cmd,manifest,config,outputs):
        self.name=name;self.cmd=cmd;self.manifest=Path(manifest);self.config=config;self.outputs=outputs
    def state(self):
        if not self.manifest.exists(): return 'todo'
        m=json.loads(self.manifest.read_text())
        if m.get('status')!='done': return 'interrupted'
        if m.get('config_hash')!=chash(self.config): return 'conflict'
        if not all(Path(o).exists() for o in self.outputs): return 'interrupted'
        return 'done'
    def write(self,status,extra=None):
        self.manifest.parent.mkdir(parents=True,exist_ok=True)
        m={'stage':self.name,'status':status,'config':self.config,'config_hash':chash(self.config),
           'command':' '.join(self.cmd),'git_commit':git_commit(),'time':time.strftime('%Y-%m-%d %H:%M:%S'),
           'outputs':[str(o) for o in self.outputs],**(extra or {})}
        self.manifest.write_text(json.dumps(m,indent=2))


def plan(cfg,seeds):
    py=sys.executable;sd=split_dir(cfg);p=cfg['preprocess']
    stages=[Stage('preprocess',[py,'-m','src.preprocess','--raw',p.get('raw','data/raw/ml-1m'),'--out',str(sd),
                                '--seed',str(p['seed']),'--protocol',p.get('protocol','loo2')],
                  sd/'manifest_preprocess.json',p,[sd/'train.csv',sd/'test.csv',sd/'stats.json'])]
    for seed in seeds:
        rd=run_dir(cfg,seed)
        stages.append(Stage('popularity',[py,'-m','src.popularity','--data',str(sd),'--out',str(rd)],
                            rd/'manifest_popularity.json',{'split':str(sd)},[rd/'popularity.csv']))
        for m in MODELS:
            mc=cfg['models'][m]
            stages.append(Stage(f'train_{m}',[py,'-W','ignore','-m','src.train','--model',m,'--seed',str(seed),
                                              '--data',str(sd),'--out',str(rd),*flags(mc)],
                                rd/f'manifest_train_{m}.json',{'model':m,'seed':seed,'split':str(sd),**mc},
                                [rd/f'{m}_seed{seed}.pt',rd/f'{m}_seed{seed}.csv']))
        fc=cfg.get('fusion',{})
        stages.append(Stage('fusion',[py,'-W','ignore','-m','src.cand_gate','--data',str(sd),'--scores',str(rd/'scores'),
                                      '--score-seed',str(seed),'--seed',str(seed),'--out',str(rd/'fusion'),*flags(fc)],
                            rd/'manifest_fusion.json',{'seed':seed,'split':str(sd),**fc},[rd/'fusion'/'metrics.csv']))
        if cfg.get('ablations'):
            ab=json.loads(Path(cfg['ablations']).read_text())
            for v in ab['variants']:
                if v['name']==ab.get('reference') and not cfg.get('rerun_reference'): continue  # = fusion stage
                vc={**fc,'methods':ab['methods'],**v['args']}
                od=rd/'ablations'/v['name']
                stages.append(Stage(f"ablation_{v['name']}",[py,'-W','ignore','-m','src.cand_gate','--data',str(sd),
                                    '--scores',str(rd/'scores'),'--score-seed',str(seed),'--seed',str(seed),'--out',str(od),*flags(vc)],
                                    rd/f"manifest_ablation_{v['name']}.json",{'seed':seed,'split':str(sd),**vc},[od/'metrics.csv']))
    return stages


def run(args):
    cfg=json.loads(Path(args.config).read_text())
    seeds=args.seeds or cfg['seeds']
    stages=[s for s in plan(cfg,seeds) if not args.stages or s.name in args.stages or s.name.split('_')[0] in args.stages]
    for st in stages:
        state=st.state()
        print(f'[{state:>11}] {st.name:<14} {st.manifest.parent}')
        if args.dry_run:
            print('              '+' '.join(st.cmd));continue
        if state=='done': continue
        if state=='conflict' and not args.force:
            raise SystemExit(f'{st.manifest} was completed with a different config. Refusing to overwrite; '
                             'use a new experiment name or pass --force.')
        if state in ('conflict','interrupted') and '--overwrite' not in st.cmd and \
                (st.name=='fusion' or st.name.startswith('ablation_')) and Path(st.outputs[0]).exists():
            st.cmd=st.cmd+['--overwrite']
        st.write('running');t=time.time()
        log=st.manifest.with_suffix('.log')
        with open(log,'w') as fh: r=subprocess.run(st.cmd,stdout=fh,stderr=subprocess.STDOUT)
        if r.returncode!=0:
            st.write('failed',{'returncode':r.returncode,'log':str(log)})
            raise SystemExit(f'{st.name} failed (exit {r.returncode}); see {log}')
        extra={'seconds':round(time.time()-t,1),'log':str(log)}
        sd=split_dir(cfg)
        if (sd/'stats.json').exists(): extra['data_stats']=json.loads((sd/'stats.json').read_text())
        st.write('done',extra)


def aggregate(args):
    cfg=json.loads(Path(args.config).read_text())
    seeds=[s for s in (args.seeds or cfg['seeds']) if (run_dir(cfg,s)/'fusion'/'metrics.csv').exists()]
    if not seeds: raise SystemExit('no completed fusion stages found')
    out=Path(cfg.get('runs_root','runs'))/cfg['name']/'aggregate';out.mkdir(parents=True,exist_ok=True)
    metrics=pd.concat([pd.read_csv(run_dir(cfg,s)/'fusion'/'metrics.csv').assign(seed=s) for s in seeds])
    cols=['Recall@10','Recall@20','NDCG@10','NDCG@20']
    summ=metrics.groupby(['split','method'])[cols].agg(['mean','std','count'])
    summ.columns=[f'{a}_{b}' for a,b in summ.columns];summ=summ.reset_index()
    summ.to_csv(out/'summary_mean_std.csv',index=False)
    print(f'seeds: {seeds}');print(summ[summ.split=='test'].to_string(index=False,float_format='%.4f'))

    # Paired bootstrap on test: per-user metrics averaged over seeds (targets must be identical).
    preds={s:np.load(run_dir(cfg,s)/'fusion'/'test_predictions.npz') for s in seeds}
    truth=preds[seeds[0]]['truth']
    for s in seeds:
        if not np.array_equal(preds[s]['truth'],truth):
            print('held-out targets differ across seeds: skipping paired statistics');return
    methods=[f[len('top20_'):] for f in preds[seeds[0]].files if f.startswith('top20_')]
    per={k:{m:np.mean([per_user_metrics(preds[s][f'top20_{m}'],truth,k)[1] for s in seeds],0) for m in methods} for k in (10,20)}
    rec={m:np.mean([per_user_metrics(preds[s][f'top20_{m}'],truth,10)[0] for s in seeds],0) for m in methods}
    pairs=[('candidate_gate','best_single'),('candidate_gate','fixed_weights'),('candidate_gate','per_user_gate'),
           ('per_user_gate','fixed_weights'),('per_user_gate','best_single'),('fixed_weights','best_single')]
    rows=[]
    for a,b in pairs:
        if a not in methods or b not in methods: continue
        for name,va,vb in (('NDCG@10',per[10][a],per[10][b]),('NDCG@20',per[20][a],per[20][b]),('Recall@10',rec[a],rec[b])):
            rows.append({'comparison':f'{a} - {b}','metric':name,**paired_bootstrap(va,vb,n_boot=args.n_boot)})
    bs=pd.DataFrame(rows);bs.to_csv(out/'paired_bootstrap_test.csv',index=False)
    print('\nPaired bootstrap (test, per-user metrics averaged over seeds):')
    print(bs.to_string(index=False,float_format='%.4f'))


def ablation_summary(args):
    """Test metrics per ablation variant (mean ± std over seeds) and change vs the reference."""
    cfg=json.loads(Path(args.config).read_text());ab=json.loads(Path(cfg['ablations']).read_text())
    seeds=args.seeds or cfg['seeds'];rows=[]
    for v in ab['variants']:
        for s in seeds:
            f=run_dir(cfg,s)/('fusion' if v['name']==ab.get('reference') else f"ablations/{v['name']}")/'metrics.csv'
            if f.exists():
                d=pd.read_csv(f);d=d[(d.split=='test')&d.method.isin(ab['methods'])]
                rows.append(d.assign(variant=v['name'],seed=s))
    if not rows: raise SystemExit('no ablation results found')
    df=pd.concat(rows);cols=['Recall@10','NDCG@10','Recall@20','NDCG@20']
    g=df.groupby(['variant','method'])[cols].agg(['mean','std','count']);g.columns=[f'{a}_{b}' for a,b in g.columns]
    g=g.reset_index();ref=g[g.variant==ab['reference']].set_index('method')
    for c in cols:
        g[f'{c}_delta_vs_ref']=g.apply(lambda r:r[f'{c}_mean']-ref[f'{c}_mean'].get(r.method,np.nan),axis=1)
    out=Path(cfg.get('runs_root','runs'))/cfg['name']/'aggregate';out.mkdir(parents=True,exist_ok=True)
    g.to_csv(out/'ablation_summary.csv',index=False)
    print(g[['variant','method','NDCG@10_mean','NDCG@10_std','NDCG@10_count','NDCG@10_delta_vs_ref']].to_string(index=False,float_format='%.4f'))


def estimate(args):
    print('''Measured on this Mac (Apple-silicon GPU via MPS), ML-1M, dim 64, one seed:
  SASRec   ~60 s/epoch, early stop after ~8-14 epochs      ~10-15 min
  NGCF     ~5 s/epoch + eval, early stop after ~65-85 ep.  ~8-10 min
  DeepFM   ~45-60 s/epoch, early stop after ~11-14 epochs  ~12-15 min
  Fusion   per-user + candidate gate + baselines           ~5-15 min
  Total per seed ~40-55 min; 5 seeds ~3.5-4.5 h sequentially.
Disk per seed: score matrices 82 MB each (6 for loo2, 9 for loo3) + ~45 MB checkpoints
  -> ~0.6-0.8 GB per seed, ~3-4 GB for 5 seeds (runs/ is git-ignored).
Peak RAM: ~3-4 GB (fusion holds z-scores and ranks for 2-3 splits).''')


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('run');r.add_argument('config');r.add_argument('--seeds',type=int,nargs='*')
    r.add_argument('--stages',nargs='*',help='e.g. preprocess train_sasrec train fusion')
    r.add_argument('--dry-run',action='store_true');r.add_argument('--force',action='store_true')
    a=sub.add_parser('aggregate');a.add_argument('config');a.add_argument('--seeds',type=int,nargs='*')
    a.add_argument('--n-boot',type=int,default=10000)
    b=sub.add_parser('ablations');b.add_argument('config');b.add_argument('--seeds',type=int,nargs='*')
    sub.add_parser('estimate')
    args=ap.parse_args()
    {'run':run,'aggregate':aggregate,'ablations':ablation_summary,'estimate':estimate}[args.cmd](args)

if __name__=='__main__':
    main()
