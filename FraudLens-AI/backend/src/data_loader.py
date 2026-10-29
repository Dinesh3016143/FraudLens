"""Safe loading of CSV data (uploaded or bundled sample).

Uploaded files are only ever parsed as CSV text with pandas. They are never
executed, and cell contents are treated as plain, untrusted data.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

from .validation import ValidationResult, validate_dataframe

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_ROWS = 50_000
SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_transactions.csv"


class DataLoadError(Exception):
    """Raised for problems with the file itself (type, size, unreadable CSV)."""


def load_csv(source: Union[str, Path, BinaryIO], filename: str | None = None) -> ValidationResult:
    """Read a CSV from a path or file-like object, then validate it."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))
    if not str(name).lower().endswith(".csv"):
        raise DataLoadError("Only .csv files are supported.")

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path.name}")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise DataLoadError("File is too large (limit 5 MB).")
        handle: BinaryIO = open(path, "rb")
        close = True
    else:
        handle = source
        close = False

    try:
        data = handle.read(MAX_UPLOAD_BYTES + 1)
    finally:
        if close:
            handle.close()

    if len(data) > MAX_UPLOAD_BYTES:
        raise DataLoadError("File is too large (limit 5 MB).")

    try:
        import io

        raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=True)
    except Exception:
        # Deliberately generic: do not echo file contents into errors/logs.
        raise DataLoadError("Could not parse the file as a CSV.")

    if len(raw) > MAX_ROWS:
        raise DataLoadError(f"Too many rows (limit {MAX_ROWS:,}).")

    return validate_dataframe(raw)


def load_sample() -> ValidationResult:
    """Load the bundled synthetic sample dataset."""
    return load_csv(SAMPLE_PATH)
