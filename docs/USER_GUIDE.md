# User Guide

This guide explains how to install, train, test, and run inference with the Forex Ensemble Prediction project.

## Prerequisites

- Python 3.11 recommended. [cite:39]
- A forex CSV dataset containing `date`, `currency_code`, and `exchange_rate`. [cite:39]
- Optional: Docker for containerized runs, and MLflow for experiment browsing. [cite:39]

## Install

```bash
pip install -r requirements.txt
```

The pinned requirements include the training stack, testing tools, formatting tools, and MLflow support. [cite:39]

## Train a model

```bash
python main.py --data Forex_Data.csv --output outputs --config config/features.yaml
```

Useful flags:
- `--epochs 40`
- `--timesteps 15`
- `--test-ratio 0.2`
- `--log-level DEBUG`

These are supported directly by `main.py`. [cite:39]

## Run inference

```bash
python predict.py --input Forex_Data.csv --output predictions.csv --model lgb
```

To use a specific saved run:

```bash
python predict.py --input Forex_Data.csv --model-dir outputs/run_YYYYMMDD_HHMMSS --model xgb
```

`predict.py` can load `lgb`, `xgb`, or `stacking`. [cite:39]

## Run tests

```bash
pytest tests/ -v
```

The repository currently includes unit tests for cleaning, metrics, and feature engineering. [cite:39]

## Lint and formatting

```bash
flake8 src/ tests/ main.py predict.py --max-line-length=120 --ignore=E501,W503
black --check src/ tests/ main.py predict.py --line-length=120
isort --check-only src/ tests/ main.py predict.py
```

These commands match the CI workflow. [cite:39]

## Docker workflow

```bash
docker build -t forex-prediction .
docker run --rm -e DATA_PATH=Forex_Data.csv forex-prediction
```

The project ships with a multi-stage Dockerfile and `.dockerignore`. [cite:39]

## Outputs

After training, inspect `outputs/` for metrics CSVs, trained models, Keras artifacts, SHAP outputs, and charts. Versioned runs can also create `outputs/<run_id>/manifest.json` and `outputs/latest/` artifacts. [cite:39]
