import io

import pandas as pd

from backend.src.data_loader import load_sample
from backend.src.detection_engine import analyze
from backend.src.reporting import analyzed_to_csv


def test_csv_export_roundtrip_and_privacy_defaults():
    res = analyze(load_sample().clean)
    out = pd.read_csv(io.BytesIO(analyzed_to_csv(res.data)))
    assert len(out) == len(res.data)
    assert "risk_score" in out.columns
    assert "findings" not in out.columns
    assert "transaction_description" not in out.columns


def test_csv_export_can_include_descriptions_and_neutralizes_formulas():
    df = analyze(load_sample().clean).data.copy()
    df.loc[0, "transaction_description"] = "=HYPERLINK(\"http://x\")"
    out = pd.read_csv(io.BytesIO(analyzed_to_csv(df, include_description=True)))
    assert out.loc[0, "transaction_description"].startswith("'")
