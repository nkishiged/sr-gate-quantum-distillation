from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify_release',ROOT/'scripts/verify_release.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_release_manifest():
    mod.validate_release_manifest()

def test_frozen_results():
    df=mod.validate_results()
    assert len(df)==900
