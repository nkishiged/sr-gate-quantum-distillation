# Publish this artifact on GitHub and Zenodo

## 1. Verify the local artifact

```bash
python scripts/verify_release.py
pytest -q
```

Review `CITATION.cff` and `.zenodo.json`. Confirm author order, affiliations, ORCIDs, repository name, release date, and license before publication. When both files are present, Zenodo uses `.zenodo.json` for GitHub-release metadata, while GitHub uses `CITATION.cff` to display citation information.

## 2. Create and push the GitHub repository

Create an empty public repository, for example `sr-gate-distillation`, without generating a second README or license. Extract this ZIP, open a terminal in the extracted root, and run:

```bash
git init
git add .
git commit -m "Release reproducibility artifact v3.0.0"
git branch -M main
git remote add origin https://github.com/USERNAME/sr-gate-distillation.git
git push -u origin main
```

Record the immutable commit:

```bash
git rev-parse HEAD
```

## 3. Enable the repository in Zenodo

Sign in to Zenodo, link the GitHub account, open the GitHub integration page, and enable the repository. Enable it **before** publishing the GitHub release so the release can be ingested automatically.

## 4. Publish the GitHub release

Create tag `v3.0.0` from the verified commit and publish a GitHub release. Command-line alternative:

```bash
git tag -a v3.0.0 -m "Sealed final-holdout reproducibility artifact"
git push origin v3.0.0
gh release create v3.0.0   --title "v3.0.0 — Sealed final-holdout artifact"   --notes-file docs/RELEASE_NOTES_v3.0.0.md
```

GitHub automatically provides source ZIP and tar archives for the tagged release.

## 5. Confirm Zenodo ingestion

Wait for Zenodo to process the GitHub release, then open the generated software record and verify its metadata and DOI. Check that the archived release corresponds to tag `v3.0.0` and the recorded Git commit.

## 6. Update the manuscript

Insert the final values in the Data and Code Availability statement:

- GitHub repository URL;
- release tag `v3.0.0`;
- full Git commit SHA;
- Zenodo version DOI;
- optionally, the concept DOI for all versions.

Do not reuse a DOI or repository URL from another project.
