#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, shutil
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'notebooks/Quantum_CeNN_Reproducible_T4_5H_V3_FinalHoldout.ipynb'
VALID={'NONE','SMOKE','PROFILE','FULL'}

def patch_mode(nb, mode):
    pattern=re.compile(r'^RUN_MODE\s*=\s*["\'](?:NONE|SMOKE|PROFILE|FULL)["\']\s*$', re.M)
    hits=0
    for cell in nb.cells:
        if cell.cell_type!='code': continue
        new,n=pattern.subn(f'RUN_MODE = "{mode}"', cell.source)
        if n: cell.source=new; hits += n
    if hits != 1: raise RuntimeError(f'Expected one RUN_MODE assignment, found {hits}.')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=sorted(VALID), required=True)
    ap.add_argument('--workspace', type=Path)
    ap.add_argument('--confirm-final-test', action='store_true')
    ap.add_argument('--timeout', type=int, default=-1, help='Per-cell seconds; -1 disables timeout')
    args=ap.parse_args()
    if args.mode=='FULL' and not args.confirm_final_test:
        raise SystemExit('FULL refused. Re-run with --confirm-final-test after freezing the protocol.')
    workspace=(args.workspace or ROOT/'runs'/args.mode.lower()).resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    nb=nbformat.read(SOURCE, as_version=4); patch_mode(nb,args.mode)
    input_copy=workspace/f'Quantum_CeNN_{args.mode}_input.ipynb'
    output=workspace/f'Quantum_CeNN_{args.mode}_executed.ipynb'
    nbformat.write(nb,input_copy)
    ep=ExecutePreprocessor(timeout=args.timeout, kernel_name='python3', allow_errors=False)
    ep.preprocess(nb, {'metadata': {'path': str(workspace)}})
    nbformat.write(nb,output)
    print(f'Executed notebook: {output}')
if __name__=='__main__': main()
