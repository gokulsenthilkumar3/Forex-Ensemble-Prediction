# Development Guide

This document summarizes the current code layout and development workflow.

## Code layout

```text
src/
├── data/
├── evaluation/
├── features/
├── mlops/
├── models/
└── utils/
```

The project also includes `main.py`, `predict.py`, `.github/workflows/ci.yml`, `tests/`, `Dockerfile`, and `config/features.yaml`. [cite:39]

## Main modules

- `src/data/loader.py`: data loading and required-column validation. [cite:39]
- `src/data/cleaner.py`: per-currency IQR outlier removal. [cite:39]
- `src/features/engineer.py`: feature engineering and config loading. [cite:39]
- `src/features/cross_currency.py`: cross-currency signals. [cite:39]
- `src/features/shap_ranking.py`: SHAP importance helpers. [cite:39]
- `src/models/tree_models.py`: XGBoost and LightGBM builders and trainers. [cite:39]
- `src/models/deep_learning.py`: GRU, LSTM, BiLSTM-Attn, Transformer, and TFT builders. [cite:39]
- `src/evaluation/metrics.py`: regression and directional accuracy metrics. [cite:39]
- `src/evaluation/visualize.py`: output charts. [cite:39]
- `src/mlops/`: MLflow and versioned artifacts. [cite:39]

## Local workflow

1. Install requirements.
2. Run tests.
3. Run formatting checks.
4. Run the training pipeline.
5. Open a PR from a focused branch.

This matches the repo's recent branch-based improvement workflow. [conversation_history:1]

## CI workflow

CI runs three stages in sequence: lint, tests, then Docker build. The configuration lives in `.github/workflows/ci.yml`. [cite:39]

## Suggested next improvements

- Add pre-commit hooks.
- Add API serving docs.
- Add release process docs.
- Add sample datasets or fixtures for faster local testing.
