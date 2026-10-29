"""Backend export helpers (Phase 1: CSV export of analyzed transactions)."""
from __future__ import annotations

import pandas as pd

DISCLAIMER = (
    "Risk scores are illustrative indicators produced by configurable rules. "
    "They are not probabilities of fraud and do not confirm fraud."
)


def analyzed_to_csv(df: pd.DataFrame, include_description: bool = False) -> bytes:
    """Return analyzed transactions as CSV bytes.

    The nested `findings` column is dropped (the `rules_triggered` column keeps the
    rule IDs). Free-text descriptions are excluded unless explicitly requested, to
    avoid exporting unnecessary information.
    """
    export = df.drop(columns=["findings"], errors="ignore").copy()
    if not include_description and "transaction_description" in export.columns:
        export = export.drop(columns=["transaction_description"])
    # Guard against spreadsheet formula injection from untrusted text cells.
    for col in export.select_dtypes(include=["object", "string"]).columns:
        export[col] = export[col].map(_neutralize)
    return export.to_csv(index=False).encode("utf-8")


def _neutralize(value):
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value
