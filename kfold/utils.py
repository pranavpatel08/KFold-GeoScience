"""Utilities for logging, seeding, hardware introspection, and IO.

Importing this module is the canonical way to silence TensorFlow's
C++/Python logging in this project. Other modules that use TF should
``import utils`` before importing tensorflow.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import json
import random
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import psutil


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def save_csv_dataframe(df, path: Path):
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def format_cv_table(cv_results: Dict[str, Dict[str, float]], model_order: Iterable[str]) -> str:
    headers = ["Model", "OOF R2", "OOF RMSE", "OOF MAE"]
    rows: List[List[str]] = []
    for name in model_order:
        if name not in cv_results:
            continue
        stats = cv_results[name]
        rows.append(
            [
                name,
                f"{stats['oof_r2']:.4f}",
                f"{stats['oof_rmse']:.2f}",
                f"{stats['oof_mae']:.2f}",
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        col_widths = [max(w, len(v)) for w, v in zip(col_widths, row)]

    def _line(left: str, mid: str, right: str, fill: str) -> str:
        return left + mid.join(fill * (w + 2) for w in col_widths) + right

    def _row(values: List[str]) -> str:
        return "║ " + " ║ ".join(v.ljust(w) for v, w in zip(values, col_widths)) + " ║"

    lines = [
        _line("╔", "╦", "╗", "═"),
        _row(headers),
        _line("╠", "╬", "╣", "═"),
    ]
    lines.extend(_row(r) for r in rows)
    lines.append(_line("╚", "╩", "╝", "═"))
    return "\n".join(lines)


def get_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cpu": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
    }
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        info["gpu"] = [gpu.name for gpu in gpus]
    return info


