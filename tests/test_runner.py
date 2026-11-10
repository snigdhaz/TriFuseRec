import json
import pytest
from src import runner
from tests.test_cand_gate import make_dataset


def write_cfg(tmp_path,raw,fusion_epochs=1):
    cfg={'name':'smoke','splits_root':str(tmp_path/'splits'),'runs_root':str(tmp_path/'runs'),
         'preprocess':{'seed':0,'protocol':'loo3','raw':str(raw)},'seeds':[1,2],
         'models':{'sasrec':{'dim':8,'epochs':1,'cpu':True,'max_len':5},
                   'ngcf':{'dim':8,'epochs':1,'cpu':True,'ngcf_infer_graph':'history'},
                   'deepfm':{'dim':8,'epochs':1,'cpu':True}},
         'fusion':{'epochs':fusion_epochs,'hard_negatives':2,'random_negatives':3,'batch':8}}
    p=tmp_path/'cfg.json';p.write_text(json.dumps(cfg));return p


def args(**kw):
    base=dict(seeds=None,stages=None,dry_run=False,force=False,n_boot=200)
    base.update(kw);return type('A',(),base)


def test_runner_smoke_resume_and_no_overwrite(tmp_path,capsys):
    make_dataset(tmp_path,'loo2')            # only to create tmp_path/raw
    cfg=write_cfg(tmp_path,tmp_path/'raw')
    runner.run(args(config=cfg))
    for s in (1,2):
        d=tmp_path/'runs'/'smoke'/f'seed{s}'
        assert (d/'fusion'/'metrics.csv').exists()
        m=json.loads((d/'manifest_train_sasrec.json').read_text())
        assert m['status']=='done' and m['data_stats']['protocol']=='loo3'
    capsys.readouterr()
    runner.run(args(config=cfg))             # resume: everything already done
    assert capsys.readouterr().out.count('[       done]')==1+2*5
    runner.aggregate(args(config=cfg))
    agg=tmp_path/'runs'/'smoke'/'aggregate'
    assert (agg/'summary_mean_std.csv').exists() and (agg/'paired_bootstrap_test.csv').exists()
    cfg2=write_cfg(tmp_path,tmp_path/'raw',fusion_epochs=2)   # changed config, same name
    with pytest.raises(SystemExit): runner.run(args(config=cfg2,stages=['fusion']))


def test_runner_ablation_stages(tmp_path):
    make_dataset(tmp_path,'loo2')
    cfg=json.loads(write_cfg(tmp_path,tmp_path/'raw').read_text())
    ab={'reference':'full','methods':['fixed_weights','candidate_gate'],
        'variants':[{'name':'full','args':{}},{'name':'no_item_features','args':{'drop_groups':['item_pop','item_genre']}},
                    {'name':'no_ngcf','args':{'experts':['deepfm','sasrec']}}]}
    (tmp_path/'ab.json').write_text(json.dumps(ab));cfg['ablations']=str(tmp_path/'ab.json');cfg['seeds']=[1]
    p=tmp_path/'cfg_ab.json';p.write_text(json.dumps(cfg))
    runner.run(args(config=p))
    d=tmp_path/'runs'/'smoke'/'seed1'
    assert (d/'ablations'/'no_ngcf'/'metrics.csv').exists() and not (d/'ablations'/'full').exists()
    runner.ablation_summary(args(config=p))
    assert (tmp_path/'runs'/'smoke'/'aggregate'/'ablation_summary.csv').exists()
