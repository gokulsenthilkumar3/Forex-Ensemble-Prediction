# Contributing

## Workflow

1. Create a focused branch from `main`.
2. Make small, reviewable commits.
3. Run tests and lint checks locally.
4. Open a pull request with labels, reviewer, and assignee.

This repository has recently used focused branches for model training, data/features, code quality, MLOps, and documentation improvements. [conversation_history:1]

## Local checks

```bash
pytest tests/ -v
flake8 src/ tests/ main.py predict.py --max-line-length=120 --ignore=E501,W503
black --check src/ tests/ main.py predict.py --line-length=120
isort --check-only src/ tests/ main.py predict.py
```

These match the CI workflow already present in the repository. [cite:39]

## Coding style

- Keep modules focused.
- Prefer docstrings on public functions.
- Use config-driven behavior where possible.
- Avoid committing large generated artifacts.

The repository already ignores outputs, model binaries, scratch files, logs, and MLflow run directories in ignore files. [cite:39]
