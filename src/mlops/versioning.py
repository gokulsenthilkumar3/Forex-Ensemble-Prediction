"""
src/mlops/versioning.py
========================
Versioned artifact saving with run ID and timestamp.

Ensures every training run produces uniquely named artifacts
so models from different runs never overwrite each other.
"""

from __future__ import annotations
import os
import logging
import datetime
import joblib
import shutil

log = logging.getLogger(__name__)


def make_run_id() -> str:
    """
    Generate a timestamped run ID string.

    Returns
    -------
    String in format 'run_YYYYMMDD_HHMMSS', e.g. 'run_20260515_221500'.
    """
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def versioned_output_dir(base_dir: str, run_id: str) -> str:
    """
    Create and return a versioned subdirectory: base_dir/run_id/.

    Parameters
    ----------
    base_dir : Root output directory (e.g. 'outputs').
    run_id   : Unique run identifier from make_run_id().

    Returns
    -------
    Full path to the created versioned directory.
    """
    path = os.path.join(base_dir, run_id)
    os.makedirs(path, exist_ok=True)
    log.info(f"Versioned output dir: {path}")
    return path


def save_versioned_model(model, base_dir: str, name: str, run_id: str) -> str:
    """
    Save a model to both the versioned run directory and the 'latest' symlink.

    Artifacts are saved as:
        base_dir/<run_id>/<name>.pkl         <- versioned copy
        base_dir/latest/<name>.pkl           <- overwritten 'latest' copy

    Parameters
    ----------
    model    : Fitted sklearn/xgboost/lightgbm model.
    base_dir : Root output directory.
    name     : Model filename without extension.
    run_id   : Current run identifier.

    Returns
    -------
    Path to the versioned artifact.
    """
    run_dir    = os.path.join(base_dir, run_id)
    latest_dir = os.path.join(base_dir, "latest")
    os.makedirs(run_dir,    exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    versioned_path = os.path.join(run_dir, f"{name}.pkl")
    latest_path    = os.path.join(latest_dir, f"{name}.pkl")

    joblib.dump(model, versioned_path)
    joblib.dump(model, latest_path)    # overwrite latest
    log.info(f"Versioned model saved: {versioned_path}")
    log.info(f"Latest model updated:  {latest_path}")
    return versioned_path


def save_versioned_keras(model, base_dir: str, name: str, run_id: str) -> str:
    """
    Save a Keras model to the versioned run directory and 'latest'.

    Parameters
    ----------
    model    : Fitted tf.keras.Model.
    base_dir : Root output directory.
    name     : Model filename without extension.
    run_id   : Current run identifier.

    Returns
    -------
    Path to the versioned artifact.
    """
    run_dir    = os.path.join(base_dir, run_id)
    latest_dir = os.path.join(base_dir, "latest")
    os.makedirs(run_dir,    exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    versioned_path = os.path.join(run_dir,    f"{name}.keras")
    latest_path    = os.path.join(latest_dir, f"{name}.keras")

    model.save(versioned_path)
    model.save(latest_path)
    log.info(f"Versioned Keras model saved: {versioned_path}")
    return versioned_path


def write_run_manifest(base_dir: str, run_id: str, metadata: dict) -> str:
    """
    Write a JSON manifest file for the run with metadata.

    Parameters
    ----------
    base_dir : Root output directory.
    run_id   : Current run identifier.
    metadata : Dict of arbitrary key-value metadata (params, metrics, paths).

    Returns
    -------
    Path to the written manifest file.
    """
    import json
    run_dir  = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump({"run_id": run_id, **metadata}, f, indent=2, default=str)
    log.info(f"Run manifest written: {path}")
    return path
