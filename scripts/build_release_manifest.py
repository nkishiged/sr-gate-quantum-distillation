#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'release_manifest_sha256.csv'
EXCLUDE={OUT.resolve()}
def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
rows=[]
for p in sorted(x for x in ROOT.rglob('*') if x.is_file() and x.resolve() not in EXCLUDE and '.git' not in x.parts):
    rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)})
with OUT.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']); w.writeheader(); w.writerows(rows)
print(f'Wrote {OUT} with {len(rows)} entries')
