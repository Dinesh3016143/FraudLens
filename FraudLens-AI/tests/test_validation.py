import io

import pandas as pd
import pytest

from backend.src.data_loader import DataLoadError, load_csv, load_sample
from backend.src.validation import validate_dataframe


def make_row(**overrides):
    row = {
        "transaction_id": "T1", "timestamp": "2026-09-01 10:00:00", "amount": "100",
        "currency": "INR", "account_avg_amount": "100", "transactions_last_hour": "1",
        "is_new_payee": "0", "merchant_category": "groceries", "channel": "web",
    }
    row.update(overrides)
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


def test_valid_sample_csv_loads():
    res = load_sample()
    assert res.ok
    assert res.invalid_row_count == 0
    assert len(res.clean) == res.total_rows


def test_valid_rows_get_proper_types():
    res = validate_dataframe(frame(make_row()))
    assert res.ok
    assert res.clean["amount"].dtype.kind == "f"
    assert res.clean["is_new_payee"].iloc[0] == 1 or res.clean["is_new_payee"].iloc[0] == 0
    assert pd.api.types.is_datetime64_any_dtype(res.clean["timestamp"])


def test_missing_required_column():
    df = frame(make_row()).drop(columns=["amount"])
    res = validate_dataframe(df)
    assert not res.ok
    assert "amount" in res.file_errors[0]


def test_empty_dataset():
    res = validate_dataframe(pd.DataFrame())
    assert not res.ok
    assert res.file_errors


def test_invalid_numeric_field():
    res = validate_dataframe(frame(make_row(), make_row(transaction_id="T2", amount="abc")))
    assert len(res.clean) == 1
    assert (res.row_issues["column"] == "amount").any()


def test_negative_amount_rejected():
    res = validate_dataframe(frame(make_row(amount="-5")))
    assert len(res.clean) == 0
    assert "negative" in res.row_issues.iloc[0]["problem"].lower()


def test_duplicate_transaction_ids():
    res = validate_dataframe(frame(make_row(), make_row()))
    assert len(res.clean) == 1
    assert (res.row_issues["problem"] == "Duplicate transaction_id").any()


def test_invalid_binary_field():
    res = validate_dataframe(frame(make_row(is_new_payee="maybe")))
    assert len(res.clean) == 0
    assert (res.row_issues["column"] == "is_new_payee").any()


def test_binary_accepts_true_false_words():
    res = validate_dataframe(frame(make_row(is_new_payee="true"), make_row(transaction_id="T2", is_new_payee="No")))
    assert list(res.clean["is_new_payee"]) == [1, 0]


def test_invalid_timestamp():
    res = validate_dataframe(frame(make_row(timestamp="not a date")))
    assert len(res.clean) == 0
    assert (res.row_issues["column"] == "timestamp").any()


def test_invalid_currency():
    res = validate_dataframe(frame(make_row(currency="RUPEES")))
    assert len(res.clean) == 0


def test_row_numbers_point_to_csv_lines():
    res = validate_dataframe(frame(make_row(), make_row(transaction_id="T2", amount="x")))
    assert res.row_issues.iloc[0]["row"] == 3  # header is line 1, first data row line 2


def test_optional_columns_not_fabricated():
    res = validate_dataframe(frame(make_row()))
    assert "country_code" not in res.clean.columns


def test_loader_rejects_non_csv():
    with pytest.raises(DataLoadError):
        load_csv(io.BytesIO(b"a,b"), filename="data.exe")


def test_loader_rejects_oversized_file():
    big = io.BytesIO(b"a" * (5 * 1024 * 1024 + 10))
    with pytest.raises(DataLoadError):
        load_csv(big, filename="big.csv")


def test_loader_malformed_csv_does_not_crash():
    res_or_err = None
    try:
        res_or_err = load_csv(io.BytesIO(b"\x00\x01garbage"), filename="x.csv")
    except DataLoadError:
        return
    assert not res_or_err.ok
