# Releasing

This project uses tag-based publishing to PyPI via GitHub Actions.

## One-time setup

1. Configure PyPI Trusted Publisher for this repository and workflow:
   - Repository: `diogoribeiro7/torch-namo-optim`
   - Workflow: `.github/workflows/publish.yml`
   - Environment: `pypi`
2. In GitHub repository settings, create an environment named `pypi`.
3. Optionally set required reviewers for the `pypi` environment.

## Release process

1. Ensure `main` is green (CI passing).
2. Update changelog entries under `## [Unreleased]` in `CHANGELOG.md`.
3. Bump version and roll changelog:

```bash
python scripts/bump_version.py 0.1.1
```

4. Commit and push:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore(release): 0.1.1"
git push
```

5. Create and push release tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

6. GitHub Actions `Publish` workflow will:
   - build and validate package distributions,
   - publish to PyPI,
   - create a GitHub Release with generated notes.

## Notes

- Tag format must be `vX.Y.Z` (for example: `v0.1.1`).
- Keep `pyproject.toml` version and tag aligned (`0.1.1` <-> `v0.1.1`).
