"""Research tables and figures (Phases 5 and 7).

    python -m src.report --runs results/cand_gate/loo2_seed42 --tag loo2_seed42
    python -m src.report --runs runs/main_loo3/seed*/fusion --tag main_loo3 \\
        --data data/splits/loo3_tie42 --ablations runs/main_loo3/aggregate/ablation_summary.csv

--runs takes one cand_gate output directory per training seed. All runs must share
the same held-out targets (same split); per-user metrics are averaged over seeds
before the paired bootstrap. Writes results/analysis/report_{tag}/ (a new suffixed
directory if it already exists).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from .common import read_data, heldout_splits
from .cand_gate import history_times
from .compare_runs import unique_path
from .stats import per_user_metrics, paired_bootstrap

# Reference categorical palette (validated: adjacent CVD ΔE >= 9.1, normal ΔE >= 19.6).
# Three slots sit below 3:1 contrast, so every figure ships with a table in report.md.
SLOTS=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4']
SURFACE='#fcfcfb';INK='#0b0b0b';INK2='#52514e';GRID='#e4e3df'
METHOD_ORDER=['best_single','equal_weights','fixed_weights','per_user_gate','candidate_gate']
METHOD_LABEL={'best_single':'Best single','equal_weights':'Equal weights','fixed_weights':'Fixed weights',
              'per_user_gate':'Per-user gate','candidate_gate':'Candidate gate'}
METHOD_COLOR=dict(zip(METHOD_ORDER,SLOTS))
METHOD_MARKER=dict(zip(METHOD_ORDER,['o','v','s','D','^']))
EXPERT_COLOR={'deepfm':SLOTS[0],'sasrec':SLOTS[1],'ngcf':SLOTS[2]}
EXPERT_LABEL={'deepfm':'DeepFM','sasrec':'SASRec','ngcf':'NGCF'}
HIST_BINS=[0,50,100,200,500,np.inf];HIST_LABELS=['<50','50-99','100-199','200-499','500+']

plt.rcParams.update({'figure.facecolor':SURFACE,'axes.facecolor':SURFACE,'savefig.facecolor':SURFACE,
                     'axes.edgecolor':GRID,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
                     'text.color':INK,'axes.grid':True,'grid.color':GRID,'grid.linewidth':.8,
                     'axes.axisbelow':True,'axes.spines.top':False,'axes.spines.right':False,
                     'font.size':10,'axes.titlesize':11,'axes.titleweight':'semibold','legend.frameon':False})


def verdict(ci_low,ci_high):
    if ci_low>0: return 'positive (CI above 0)'
    if ci_high<0: return 'negative (CI below 0)'
    return 'inconclusive (CI spans 0)'


def md_table(df,floatfmt='.4f'):
    cols=list(df.columns)
    def fmt(v):
        if isinstance(v,(float,np.floating)): return 'n/a' if np.isnan(v) else format(v,floatfmt)
        return str(v)
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]+
                     ['| '+' | '.join(fmt(v) for v in r)+' |' for r in df.itertuples(index=False)])


def load_runs(paths):
    runs=[]
    for p in paths:
        p=Path(p)
        runs.append({'dir':p,'metrics':pd.read_csv(p/'metrics.csv'),'pred':np.load(p/'test_predictions.npz'),
                     'config':json.loads((p/'config.json').read_text())})
    truth=runs[0]['pred']['truth']
    for r in runs[1:]:
        if not np.array_equal(r['pred']['truth'],truth):
            raise SystemExit('runs have different held-out targets; they cannot be pooled or paired')
    return runs


def cohorts(data,truth,n_items):
    tr,hist,_,_,_,_=read_data(data)
    sp=heldout_splits(data,hist);times=history_times(data,tr)
    test_hist=sp['test'][0];first,_,last=times['test']
    n=np.array([len(h) for h in test_hist])
    rate=n/((last-first)/86400.+1.)
    pop=np.bincount(tr.item_idx,minlength=n_items)
    head=pop>=np.quantile(pop,.8)
    return pd.DataFrame({'history_len':pd.cut(n,HIST_BINS,labels=HIST_LABELS,right=False),
                         'activity':pd.qcut(rate,4,labels=['Q1 least','Q2','Q3','Q4 most']),
                         'target_item':np.where(head[truth],'head','tail')}),head


def save(fig,path):
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main(args):
    runs=load_runs(args.runs)
    out=Path(args.out_root)/f'report_{args.tag}'
    if out.exists(): out=unique_path(out)
    out.mkdir(parents=True)
    cfg=runs[0]['config'];data=args.data or cfg['data']
    pred0=runs[0]['pred'];truth=pred0['truth'];nu=len(truth)
    n_items=int(pd.read_csv(Path(data)/'item_map.csv').shape[0])
    methods=[m for m in METHOD_ORDER if f'top20_{m}' in pred0.files]
    coh,head=cohorts(data,truth,n_items)
    seeds=[r['config'].get('seed') for r in runs]
    md=[f'# TriFuseRec results: {args.tag}','',
        f"Protocol **{cfg['protocol']}**, split `{cfg['split_identity']}`, training seeds {seeds}, "
        f"normalization `{cfg['norm']}`, experts {cfg['experts']}.",
        'All methods rank the same candidates (all items minus the user\'s history) for the same test targets.']
    if cfg['protocol']=='loo2':
        md.append('**Preliminary:** loo2 arrangement (gates fit on validation targets of 70% of users, selected on the other 30%; '
                  'base models were early-stopped on validation of all users). The recommended loo3 protocol has not been run yet.')
    if len(runs)==1: md.append('**Single seed:** confidence intervals reflect user sampling only, not training variance.')
    md.append('')

    # ---- per-user metrics averaged over seeds
    pu={m:{} for m in methods}
    for m in methods:
        for k in (10,20):
            h,nd=zip(*[per_user_metrics(r['pred'][f'top20_{m}'],truth,k) for r in runs])
            pu[m][f'Recall@{k}']=np.mean(h,0);pu[m][f'NDCG@{k}']=np.mean(nd,0)
    rows=[]
    for m in methods:
        tops=[r['pred'][f'top20_{m}'][:,:10] for r in runs]
        rows.append({'method':METHOD_LABEL[m],**{k:float(v.mean()) for k,v in pu[m].items()},
                     'coverage@10':float(np.mean([len(np.unique(t))/n_items for t in tops])),
                     'long_tail_share@10':float(np.mean([1-head[t].mean() for t in tops])),
                     'tail_Recall@10':float(pu[m]['Recall@10'][coh.target_item.to_numpy()=='tail'].mean())})
    main_tab=pd.DataFrame(rows)
    if len(runs)>1:
        mm=pd.concat([r['metrics'] for r in runs]);mm=mm[mm.split=='test']
        sd=mm.groupby('method')[['Recall@10','NDCG@10']].std()
        main_tab['NDCG@10_std_over_seeds']=[sd.loc[m,'NDCG@10'] for m in methods]
    ref_best=main_tab.set_index('method').loc[METHOD_LABEL['best_single']] if 'best_single' in methods else None
    ref_fix=main_tab.set_index('method').loc[METHOD_LABEL['fixed_weights']] if 'fixed_weights' in methods else None
    for c in ('NDCG@10','Recall@10'):
        if ref_best is not None: main_tab[f'{c}_vs_best_single_%']=100*(main_tab[c]/ref_best[c]-1)
        if ref_fix is not None: main_tab[f'{c}_vs_fixed_%']=100*(main_tab[c]/ref_fix[c]-1)
    main_tab.to_csv(out/'methods_test.csv',index=False)
    info=cfg.get('info',{})
    md+=['## RQ1 — Does adaptive fusion beat the strongest single model and fixed fusion?','',
         f"Best single expert (chosen on the selection set): **{info.get('best_single','n/a')}**.",'',
         md_table(main_tab[['method','Recall@10','Recall@20','NDCG@10','NDCG@20']+
                           [c for c in main_tab.columns if c.endswith('_%')]]),'']

    # ---- paired bootstrap
    pairs=[('candidate_gate','best_single'),('candidate_gate','fixed_weights'),('candidate_gate','per_user_gate'),
           ('per_user_gate','fixed_weights'),('fixed_weights','best_single')]
    bs=[]
    for a,b in pairs:
        if a in methods and b in methods:
            for k in ('NDCG@10','Recall@10','NDCG@20'):
                r=paired_bootstrap(pu[a][k],pu[b][k],n_boot=args.n_boot)
                bs.append({'comparison':f'{METHOD_LABEL[a]} − {METHOD_LABEL[b]}','metric':k,**r,'verdict':verdict(r['ci_low'],r['ci_high'])})
    bs=pd.DataFrame(bs);bs.to_csv(out/'paired_bootstrap.csv',index=False)
    md+=['Paired bootstrap over users (10,000 resamples, 95% percentile CI):','',
         md_table(bs[['comparison','metric','mean_diff','ci_low','ci_high','p_value','verdict']]),'',
         '![paired differences](fig_paired_differences.png)','']
    if len(bs):
        sub=bs[bs.metric=='NDCG@10'].iloc[::-1]
        fig,ax=plt.subplots(figsize=(7,.55*len(sub)+1.2))
        y=np.arange(len(sub))
        ax.hlines(y,sub.ci_low,sub.ci_high,color=SLOTS[0],linewidth=2)
        ax.plot(sub.mean_diff,y,'o',color=SLOTS[0],markersize=8,markeredgecolor=SURFACE,markeredgewidth=2)
        ax.axvline(0,color=INK2,linewidth=1)
        ax.set_yticks(y,sub.comparison);ax.set_xlabel('Difference in test NDCG@10 (95% CI)')
        ax.set_title('Paired differences between fusion methods');ax.grid(axis='y',visible=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        save(fig,out/'fig_paired_differences.png')

    # ---- RQ2 cohorts
    md+=['## RQ2 — Which users benefit?','']
    ctabs=[]
    for col in ('history_len','activity','target_item'):
        t=pd.DataFrame({METHOD_LABEL[m]:pd.Series(pu[m]['Recall@10']).groupby(coh[col].values,observed=True).mean() for m in methods})
        t.insert(0,'users',coh[col].value_counts().reindex(t.index).to_numpy())
        t=t.reset_index().rename(columns={'index':col});ctabs.append(t)
        t.to_csv(out/f'cohort_{col}_recall10.csv',index=False)
    md+=['History length (= user degree in the interaction graph, incl. held-out history), test Recall@10:','',md_table(ctabs[0]),'',
         'Activity rate (interactions per active day) quartiles, test Recall@10:','',md_table(ctabs[1]),'',
         'Target item popularity (head = top 20% by train count); the tail row is **Tail Recall@10**:','',md_table(ctabs[2]),'',
         '![history cohorts](fig_history_cohorts.png)','']
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(len(HIST_LABELS))
    for m in methods:
        v=ctabs[0].set_index('history_len')[METHOD_LABEL[m]].reindex(HIST_LABELS).to_numpy()
        ax.plot(x,v,'-o',color=METHOD_COLOR[m],linewidth=2,markersize=8,markeredgecolor=SURFACE,markeredgewidth=2,label=METHOD_LABEL[m])
    ax.set_xticks(x,HIST_LABELS);ax.set_xlabel('History length (interactions)');ax.set_ylabel('Test Recall@10')
    ax.set_title('Accuracy by user history length');ax.grid(axis='x',visible=False)
    ax.legend(loc='upper right',ncol=1,fontsize=9)
    save(fig,out/'fig_history_cohorts.png')

    if 'best_single' in methods:
        miss=pu['best_single']['Recall@10']<.5   # missed by the best single expert (majority of seeds)
        t=pd.DataFrame([{'method':METHOD_LABEL[m],'users':int(miss.sum()),
                         'Recall@10 on users the best single model misses':float(pu[m]['Recall@10'][miss].mean()),
                         'Recall@10 on users it hits':float(pu[m]['Recall@10'][~miss].mean())} for m in methods])
        t.to_csv(out/'missed_by_best_single.csv',index=False)
        md+=['Users missed by the strongest single expert (gains here are what fusion can add; losses on hit users are its cost):','',md_table(t),'']

    # ---- RQ3 gate weights
    experts=cfg['experts'];md+=['## RQ3 — Does candidate-aware gating use complementary experts better than per-user gating?','']
    wt_rows=[];panels=[]
    for g in ('per_user_gate','candidate_gate'):
        key=f'weights_top10_{g}'
        if key not in pred0.files: continue
        w=np.mean([r['pred'][key].astype(np.float32) for r in runs],0)   # [U,10,E]
        wu=w.mean(1)
        df=pd.DataFrame(wu,columns=experts);df['history_len']=coh.history_len.to_numpy()
        gm=df.groupby('history_len',observed=True)[list(experts)].mean().reindex(HIST_LABELS)
        panels.append((g,gm))
        within=w.std(1).mean(0)   # how much weights vary across a user's top-10 items
        wt_rows.append({'gate':METHOD_LABEL[g],**{f'mean w_{e}':float(wu[:,k].mean()) for k,e in enumerate(experts)},
                        **{f'p10-p90 w_{e}':f'{np.quantile(wu[:,k],.1):.2f}-{np.quantile(wu[:,k],.9):.2f}' for k,e in enumerate(experts)},
                        'within-user std across top-10 items':float(within.mean())})
        gm.reset_index().to_csv(out/f'gate_weights_by_history_{g}.csv',index=False)
    if wt_rows:
        md+=['Gate-weight distributions on test (weights on each user\'s top-10 recommendations, averaged over seeds):','',
             md_table(pd.DataFrame(wt_rows)),'']
        for g,gm in panels:
            md+=[f'{METHOD_LABEL[g]} — mean weights by history length:','',md_table(gm.reset_index()),'']
        md+=['![gate weights](fig_gate_weights.png)','']
        fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),3.6),sharey=True,squeeze=False)
        for ax,(g,gm) in zip(axes[0],panels):
            left=np.zeros(len(gm))
            for e in experts:
                ax.barh(HIST_LABELS,gm[e].to_numpy(),left=left,color=EXPERT_COLOR[e],height=.6,
                        edgecolor=SURFACE,linewidth=2,label=EXPERT_LABEL[e])
                left+=gm[e].to_numpy()
            ax.set_xlim(0,1);ax.set_title(METHOD_LABEL[g]);ax.set_xlabel('Mean expert weight');ax.grid(axis='y',visible=False)
        axes[0][0].set_ylabel('History length')
        h,l=axes[0][0].get_legend_handles_labels()
        fig.legend(h,l,loc='lower center',ncol=len(experts),fontsize=9,bbox_to_anchor=(.5,0))
        fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'fig_gate_weights.png',dpi=160);plt.close(fig)
    if 'weights_target_candidate_gate' in pred0.files and 'best_single' in methods:
        wt=np.mean([r['pred']['weights_target_candidate_gate'] for r in runs],0)
        miss=pu['best_single']['Recall@10']<.5
        t=pd.DataFrame([{'users':lab,**{f'w_{e} on target item':float(wt[mask,k].mean()) for k,e in enumerate(experts)}}
                        for lab,mask in (('best single misses',miss),('best single hits',~miss))])
        md+=['Candidate-gate weights on the held-out target item (diagnostic only; targets are never gate inputs):','',md_table(t),'']

    # ---- RQ4 diversity / popularity
    md+=['## RQ4 — Effect on diversity and popularity bias','',
         md_table(main_tab[['method','NDCG@10','coverage@10','long_tail_share@10','tail_Recall@10']]),'',
         '![accuracy vs coverage](fig_accuracy_vs_coverage.png)','',
         '_Points that nearly coincide in the figure (e.g. fixed weights and the per-user gate) are separated in the table above._','']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for ax,col,lab in ((axes[0],'coverage@10','Catalog coverage@10'),(axes[1],'long_tail_share@10','Long-tail share of top-10')):
        for m in methods:
            r=main_tab[main_tab.method==METHOD_LABEL[m]].iloc[0]
            # Shape is a second encoding, so near-coincident points stay distinguishable.
            ax.plot(r[col],r['NDCG@10'],METHOD_MARKER[m],markersize=9,color=METHOD_COLOR[m],
                    markeredgecolor=SURFACE,markeredgewidth=1.5,label=METHOD_LABEL[m],linestyle='none')
        ax.margins(x=.15,y=.12);ax.set_xlabel(lab);ax.set_ylabel('Test NDCG@10')
    axes[0].set_title('Accuracy vs coverage');axes[1].set_title('Accuracy vs long-tail exposure')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=len(methods),fontsize=9,bbox_to_anchor=(.5,0))
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'fig_accuracy_vs_coverage.png',dpi=160);plt.close(fig)

    # ---- timestamp fix figure
    if args.timestamp_comparison and Path(args.timestamp_comparison).exists():
        tc=pd.read_csv(args.timestamp_comparison);tc=tc[tc.split=='test']
        md+=['## Timestamp tie-break fix (seed 42, loo2, not paired)','',
             md_table(tc[['method','Recall@10_old','Recall@10_new','Recall@10_rel%','NDCG@10_old','NDCG@10_new','NDCG@10_rel%']]),'',
             '![timestamp fix](fig_timestamp_fix.png)','']
        fig,ax=plt.subplots(figsize=(7,4));y=np.arange(len(tc));h=.36
        ax.barh(y+h/2,tc['Recall@10_old'],h,color=SLOTS[0],edgecolor=SURFACE,linewidth=2,label='Old (item-id tie-break)')
        ax.barh(y-h/2,tc['Recall@10_new'],h,color=SLOTS[1],edgecolor=SURFACE,linewidth=2,label='Corrected (random tie-break)')
        for yy,v in zip(y-h/2,tc['Recall@10_new']): ax.text(v+.004,yy,f'{v:.3f}',va='center',fontsize=8,color=INK2)
        ax.set_yticks(y,tc.method);ax.invert_yaxis();ax.set_xlabel('Test Recall@10');ax.grid(axis='y',visible=False)
        ax.set_title('Effect of the timestamp tie-break fix')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=9)
        save(fig,out/'fig_timestamp_fix.png')

    # ---- ablations
    if args.ablations and Path(args.ablations).exists():
        ab=pd.read_csv(args.ablations)
        md+=['## Ablations (test NDCG@10, mean over seeds; Δ vs full model)','',
             md_table(ab[['variant','method','NDCG@10_mean','NDCG@10_std','NDCG@10_count','NDCG@10_delta_vs_ref']]),'',
             '![ablations](fig_ablations.png)','']
        sub=ab[ab.method=='candidate_gate'].sort_values('NDCG@10_delta_vs_ref')
        fig,ax=plt.subplots(figsize=(7,.4*len(sub)+1.2))
        ax.barh(sub.variant,sub['NDCG@10_delta_vs_ref'],color=SLOTS[0],height=.6,edgecolor=SURFACE,linewidth=2)
        ax.axvline(0,color=INK2,linewidth=1);ax.set_xlabel('Δ test NDCG@10 vs full candidate gate')
        ax.set_title('Candidate-gate ablations');ax.grid(axis='y',visible=False)
        save(fig,out/'fig_ablations.png')
    else:
        md+=['## Ablations','','Not run yet. See `configs/ablations.json` and the commands in README.','']

    (out/'report.md').write_text('\n'.join(md))
    print(f'wrote {out}/report.md');print('\n'.join(md[:40]))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--tag',required=True)
    ap.add_argument('--data',default=None,help='defaults to the data dir in the run config')
    ap.add_argument('--out-root',default='results/analysis')
    ap.add_argument('--timestamp-comparison',default='results/analysis/timestamp_fix_comparison.csv')
    ap.add_argument('--ablations',default=None)
    ap.add_argument('--n-boot',type=int,default=10000)
    main(ap.parse_args())
