#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'results/confirmatory_results_final_test_4b53f0d205ce.csv'
EXPECTED_SHA='de68c22d7ebbb970e67d65a806eb2ea9642e0b0927fe415add3ca1420c75eb95'
RUN_TAG='4b53f0d205ce'
RUN_FP='4b53f0d205ce3756e5ce9b19a709eebeb26e2c9b52481ddc724208fa32d41d94'
DATASETS={'ETTh1','ETTm2','Energy','Exchange','Jena','AAPL'}
TEACHERS={'QELM','VQC','QRC','QKRR','RFF','ESN'}
METHODS={'Standalone','NaiveKD','PersistenceGateKD','HardRejectSR','SoftSR'}
METRICS=['eval_MSE','eval_RMSE','eval_MAE','eval_MASE','eval_sMAPE']

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def validate_release_manifest():
    p=ROOT/'release_manifest_sha256.csv'
    if not p.exists(): raise AssertionError('release_manifest_sha256.csv missing')
    for r in csv.DictReader(p.open(encoding='utf-8')):
        target=ROOT/r['path']
        if not target.exists(): raise AssertionError(f'Missing release file: {r["path"]}')
        if target.stat().st_size != int(r['bytes']): raise AssertionError(f'Size mismatch: {r["path"]}')
        if sha256(target) != r['sha256']: raise AssertionError(f'Hash mismatch: {r["path"]}')

def validate_results():
    if sha256(RESULT)!=EXPECTED_SHA: raise AssertionError('Frozen result SHA-256 mismatch')
    df=pd.read_csv(RESULT)
    assert len(df)==900
    assert set(df.evaluation_split.astype(str))=={'final_test'}
    assert set(df.run_tag.astype(str))=={RUN_TAG}
    assert set(df.run_fingerprint.astype(str))=={RUN_FP}
    assert set(df.dataset)==DATASETS
    assert set(df.teacher)==TEACHERS
    assert set(df.method)==METHODS
    assert df[['dataset','seed','teacher','method']].duplicated().sum()==0
    assert df.groupby(['dataset','seed','teacher']).method.nunique().eq(5).all()
    assert df.groupby(['dataset','seed']).teacher.nunique().eq(6).all()
    assert set(df.seed)=={11,23,37,51,79}
    assert np.isfinite(df[METRICS+['gate']].to_numpy(float)).all()
    assert df.gate.between(0,1).all()
    wide=df.pivot(index=['dataset','seed','teacher'],columns='method',values=METRICS+['gate'])
    for idx,row in df[df.method.isin(['PersistenceGateKD','HardRejectSR','SoftSR'])].iterrows():
        key=(row.dataset,row.seed,row.teacher)
        ref='Standalone' if row.gate==0 else ('NaiveKD' if row.gate==1 else None)
        if ref:
            for m in METRICS:
                a=float(row[m]); b=float(wide.loc[key,(m,ref)])
                if not np.isclose(a,b,rtol=0,atol=1e-12):
                    raise AssertionError(f'Endpoint invariance failed: {key} {row.method} {m}')
    assert (df[df.method=='PersistenceGateKD'].gate==0).all()
    manifest=json.loads((ROOT/'results/manifests/run_manifest_4b53f0d205ce.json').read_text())
    assert manifest['status']=='complete' and manifest['run_tag']==RUN_TAG
    return df

def main():
    validate_release_manifest(); df=validate_results()
    print('Release verification: PASS')
    print(f'Rows={len(df)}, blocks={df.groupby(["dataset","seed","teacher"]).ngroups}, run_tag={RUN_TAG}')
if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'Release verification: FAIL: {exc}',file=sys.stderr); raise
