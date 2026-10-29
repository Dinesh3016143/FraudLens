"""Backend validation of transaction data.

Takes a raw DataFrame (e.g. from an uploaded CSV) and returns:
  * a cleaned DataFrame with correct types (only rows that passed validation)
  * a list of per-row issues, so the user can see exactly what to correct
  * file-level errors (e.g. missing required columns)

Optional columns are never fabricated: if absent they are simply left out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import pandas as pd

REQUIRED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "amount",
    "currency",
    "account_avg_amount",
    "transactions_last_hour",
    "is_new_payee",
    "merchant_category",
    "channel",
]
OPTIONAL_COLUMNS = ["country_code", "transaction_description"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_TRUE = {"1", "true", "yes", "y"}
_FALSE = {"0", "false", "no", "n"}


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    row_issues: pd.DataFrame  # columns: row, transaction_id, column, problem
    file_errors: List[str] = field(default_factory=list)
    total_rows: int = 0

    @property
    def ok(self) -> bool:
        """True if the file is usable (no file-level errors and at least one valid row)."""
        return not self.file_errors and len(self.clean) > 0

    @property
    def invalid_row_count(self) -> int:
        if self.row_issues.empty:
            return 0
        return int(self.row_issues["row"].nunique())


def _empty_issues() -> pd.DataFrame:
    return pd.DataFrame(columns=["row", "transaction_id", "column", "problem"])


def _parse_binary(value) -> float:
    """Return 0/1, or NaN if the value is not a valid binary flag."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().lower()
    if text in _TRUE:
        return 1.0
    if text in _FALSE:
        return 0.0
    # tolerate "1.0" / "0.0" produced by spreadsheets
    try:
        num = float(text)
        if num in (0.0, 1.0):
            return num
    except ValueError:
        pass
    return float("nan")


def validate_dataframe(raw: pd.DataFrame) -> ValidationResult:
    """Validate and clean a raw transactions DataFrame."""
    if raw is None or raw.empty:
        return ValidationResult(
            clean=pd.DataFrame(columns=ALL_COLUMNS),
            row_issues=_empty_issues(),
            file_errors=["The dataset is empty."],
            total_rows=0,
        )

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return ValidationResult(
            clean=pd.DataFrame(columns=ALL_COLUMNS),
            row_issues=_empty_issues(),
            file_errors=[f"Missing required column(s): {', '.join(missing)}"],
            total_rows=len(df),
        )

    keep = [c for c in ALL_COLUMNS if c in df.columns]
    df = df[keep].reset_index(drop=True)
    total = len(df)
    issues: list[dict] = []

    def add(mask: pd.Series, column: str, problem: str) -> None:
        for idx in df.index[mask]:
            issues.append({
                "row": int(idx) + 2,  # +2 = 1-based row number plus the CSV header line
                "transaction_id": str(df.at[idx, "transaction_id"]),
                "column": column,
                "problem": problem,
            })

    # Text fields
    for col in ("transaction_id", "merchant_category", "channel"):
        df[col] = df[col].astype("string").str.strip()
        add(df[col].isna() | (df[col] == ""), col, "Missing value")
    df["currency"] = df["currency"].astype("string").str.strip().str.upper()
    add(df["currency"].isna() | (df["currency"] == ""), "currency", "Missing value")
    bad_cur = df["currency"].notna() & (df["currency"] != "") & ~df["currency"].fillna("").str.match(_CURRENCY_RE)
    add(bad_cur, "currency", "Currency must be a 3-letter code such as INR or USD")

    # Timestamp
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    add(ts.isna(), "timestamp", "Missing or invalid timestamp")
    df["timestamp"] = ts

    # Numeric fields
    amount = pd.to_numeric(df["amount"], errors="coerce")
    add(amount.isna(), "amount", "Missing or non-numeric amount")
    add(amount.notna() & (amount < 0), "amount", "Amount must not be negative")
    df["amount"] = amount

    avg = pd.to_numeric(df["account_avg_amount"], errors="coerce")
    add(avg.isna(), "account_avg_amount", "Missing or non-numeric baseline amount")
    add(avg.notna() & (avg < 0), "account_avg_amount", "Baseline amount must not be negative")
    df["account_avg_amount"] = avg

    vel = pd.to_numeric(df["transactions_last_hour"], errors="coerce")
    add(vel.isna(), "transactions_last_hour", "Missing or non-numeric count")
    add(vel.notna() & (vel < 0), "transactions_last_hour", "Count must not be negative")
    add(vel.notna() & (vel % 1 != 0), "transactions_last_hour", "Count must be a whole number")
    df["transactions_last_hour"] = vel

    # Binary field
    flag = df["is_new_payee"].map(_parse_binary)
    add(flag.isna(), "is_new_payee", "Must be 0/1 (or true/false, yes/no)")
    df["is_new_payee"] = flag

    # Duplicate transaction IDs (first occurrence kept, later ones flagged)
    has_id = df["transaction_id"].notna() & (df["transaction_id"] != "")
    dup = has_id & df["transaction_id"].duplicated(keep="first")
    add(dup, "transaction_id", "Duplicate transaction_id")

    # Optional text columns: keep as strings, never invent values
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    issues_df = pd.DataFrame(issues) if issues else _empty_issues()
    bad_rows = set(issues_df["row"] - 2) if not issues_df.empty else set()

    clean = df.loc[~df.index.isin(bad_rows)].copy()
    clean["amount"] = clean["amount"].astype(float)
    clean["account_avg_amount"] = clean["account_avg_amount"].astype(float)
    clean["transactions_last_hour"] = clean["transactions_last_hour"].astype(int)
    clean["is_new_payee"] = clean["is_new_payee"].astype(int)
    clean["transaction_id"] = clean["transaction_id"].astype(str)
    clean["merchant_category"] = clean["merchant_category"].astype(str)
    clean["channel"] = clean["channel"].astype(str)
    clean["currency"] = clean["currency"].astype(str)
    clean = clean.reset_index(drop=True)

    return ValidationResult(clean=clean, row_issues=issues_df, file_errors=[], total_rows=total)
