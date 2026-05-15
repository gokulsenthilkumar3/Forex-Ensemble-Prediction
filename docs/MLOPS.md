# MLOps Guide

This guide documents the MLOps utilities currently present in the repository.

## Experiment tracking

The project includes `src/mlops/experiment.py` with helpers for setting up MLflow, starting runs, logging params, logging metrics, and logging artifacts. The helper is designed to degrade gracefully when `mlflow` is not installed. [cite:39]

## Versioned artifacts

`src/mlops/versioning.py` provides helpers to create timestamped run IDs, create versioned output folders, save models into both `outputs/<run_id>/` and `outputs/latest/`, and write a `manifest.json` file. [cite:39]

## Training helper

`src/mlops/train_with_tracking.py` wraps tree-model training in MLflow runs and logs metrics and model artifacts for XGBoost and LightGBM. [cite:39]

## Inference flow

`predict.py` supports choosing the model directory so you can run inference either from the latest artifacts or from a version-pinned run directory. It also applies the saved `per_currency_scalers.pkl` artifact during preprocessing. [cite:39]

## Container usage

The repository includes a multi-stage Dockerfile using Python 3.11 slim images, a non-root runtime user, and environment-variable-driven execution. It also exposes port `5000` for MLflow-related usage. [cite:39]

## CI integration

GitHub Actions validates linting, tests, and Docker image buildability on pushes and pull requests. [cite:39]
