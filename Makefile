.PHONY: verify test analyze datasets smoke profile

verify:
	python scripts/verify_release.py

test:
	pytest -q

analyze:
	python scripts/corrected_analysis.py --results results/confirmatory_results_final_test_4b53f0d205ce.csv --output build/corrected_analysis

datasets:
	python scripts/download_datasets.py --output datasets --verify

smoke:
	python scripts/run_notebook.py --mode SMOKE

profile:
	python scripts/run_notebook.py --mode PROFILE
