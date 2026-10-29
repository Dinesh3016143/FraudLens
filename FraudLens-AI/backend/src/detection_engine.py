"""Deterministic backend detection rules and transparent risk scoring.

Everything here is rule-based and traceable: each triggered rule records its
identifier, an explanation, the evidence values used, a severity label and the
points it contributed. The final 0-100 score is an *illustrative risk
indicator*, NOT a probability of fraud.

The engine has no dependency on Streamlit or any AI model, so it can be tested
on its own.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import pandas as pd

RULESET_VERSION = "1.0.0"

# Rule identifiers
R_LARGE = "R001_LARGE_VS_BASELINE"
R_BURST = "R002_VELOCITY_BURST"
R_NEW_PAYEE = "R003_HIGH_VALUE_NEW_PAYEE"
R_DUPLICATE = "R004_POSSIBLE_DUPLICATE"
R_TIMING = "R005_UNUSUAL_TIMING"
R_DATA_QUALITY = "R006_DATA_QUALITY"

RULE_NAMES = {
    R_LARGE: "Large amount vs account baseline",
    R_BURST: "Rapid transaction burst",
    R_NEW_PAYEE: "High-value payment to new payee",
    R_DUPLICATE: "Possible duplicate record",
    R_TIMING: "Unusual transaction timing",
    R_DATA_QUALITY: "Data-quality review",
}


@dataclass
class DetectionConfig:
    """All thresholds and weights in one place. Everything is adjustable in the UI."""

    # R001 - amount compared with the account's own supplied baseline
    large_ratio_threshold: float = 5.0
    large_points: int = 30
    large_severity: str = "High"

    # R002 - transactions_last_hour (supplied by the data)
    burst_threshold: int = 5
    burst_points: int = 25
    burst_severity: str = "Medium"

    # R003 - new payee + high value. Ratio works for any currency; the absolute
    # amount only applies to `absolute_threshold_currency` (we never assume INR).
    new_payee_ratio_threshold: float = 3.0
    new_payee_absolute_amount: float = 25000.0
    absolute_threshold_currency: str = "INR"
    new_payee_points: int = 25
    new_payee_severity: str = "High"
    # If R001 also fired, the same "large amount" signal is partly counted twice,
    # so R003 points are multiplied by this factor (1.0 disables the discount).
    overlap_discount: float = 0.5

    # R004 - duplicates: same amount/currency/category/channel/baseline within window
    duplicate_window_minutes: int = 10
    duplicate_points: int = 15
    duplicate_severity: str = "Medium"

    # R005 - timing: night-time hours AND amount well above baseline
    night_start_hour: int = 0
    night_end_hour: int = 5  # exclusive
    night_min_ratio: float = 2.0
    timing_points: int = 15
    timing_severity: str = "Low"

    # R006 - data quality
    data_quality_points: int = 5
    data_quality_severity: str = "Low"

    # Risk level boundaries on the 0-100 score
    medium_threshold: int = 30
    high_threshold: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    data: pd.DataFrame  # original columns + risk_score, risk_level, rules_triggered, n_rules, findings
    config: DetectionConfig
    ruleset_version: str = RULESET_VERSION
    analyzed_at: str = ""
    currency_totals: Dict[str, float] = field(default_factory=dict)


def _finding(rule_id: str, severity: str, points: float, explanation: str, evidence: dict) -> dict:
    return {
        "rule_id": rule_id,
        "rule_name": RULE_NAMES[rule_id],
        "severity": severity,
        "points": round(float(points), 2),
        "explanation": explanation,
        "evidence": evidence,
    }


def _duplicate_partners(df: pd.DataFrame, cfg: DetectionConfig) -> Dict[int, List[str]]:
    """Map row index -> IDs of other rows that look like duplicates within the time window."""
    out: Dict[int, List[str]] = {}
    if len(df) < 2:
        return out
    keys = ["amount", "currency", "merchant_category", "channel", "account_avg_amount"]
    window = pd.Timedelta(minutes=cfg.duplicate_window_minutes)
    for _, group in df.groupby(keys, dropna=False):
        if len(group) < 2:
            continue
        rows = list(group.index)
        for i in rows:
            partners = [
                df.at[j, "transaction_id"]
                for j in rows
                if j != i and abs(df.at[i, "timestamp"] - df.at[j, "timestamp"]) <= window
            ]
            if partners:
                out[i] = partners
    return out


def _in_night_window(hour: int, start: int, end: int) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # window wraps past midnight


def level_for_score(score: float, cfg: DetectionConfig) -> str:
    if score <= 0:
        return "None"
    if score >= cfg.high_threshold:
        return "High"
    if score >= cfg.medium_threshold:
        return "Medium"
    return "Low"


def evaluate_row(row: pd.Series, partners: List[str], cfg: DetectionConfig) -> List[dict]:
    """Apply every rule to one transaction and return its findings."""
    findings: List[dict] = []
    amount = float(row["amount"])
    baseline = float(row["account_avg_amount"])
    currency = str(row["currency"])
    ratio = amount / baseline if baseline > 0 else None

    large_fired = False
    if ratio is not None and ratio >= cfg.large_ratio_threshold:
        large_fired = True
        findings.append(_finding(
            R_LARGE, cfg.large_severity, cfg.large_points,
            f"Amount is {ratio:.1f}x the account's supplied baseline "
            f"(threshold {cfg.large_ratio_threshold:g}x).",
            {"amount": amount, "account_avg_amount": baseline, "ratio": round(ratio, 2), "currency": currency},
        ))

    velocity = int(row["transactions_last_hour"])
    if velocity >= cfg.burst_threshold:
        findings.append(_finding(
            R_BURST, cfg.burst_severity, cfg.burst_points,
            f"{velocity} transactions in the last hour (threshold {cfg.burst_threshold}).",
            {"transactions_last_hour": velocity},
        ))

    if int(row["is_new_payee"]) == 1:
        by_ratio = ratio is not None and ratio >= cfg.new_payee_ratio_threshold
        by_abs = currency == cfg.absolute_threshold_currency and amount >= cfg.new_payee_absolute_amount
        if by_ratio or by_abs:
            points = cfg.new_payee_points
            note = ""
            if large_fired:
                points = points * cfg.overlap_discount
                note = f" Points reduced (x{cfg.overlap_discount:g}) because R001 already counted the large amount."
            reasons = []
            if by_ratio:
                reasons.append(f"amount is {ratio:.1f}x baseline (threshold {cfg.new_payee_ratio_threshold:g}x)")
            if by_abs:
                reasons.append(f"amount >= {cfg.new_payee_absolute_amount:g} {currency}")
            findings.append(_finding(
                R_NEW_PAYEE, cfg.new_payee_severity, points,
                "Payment to a newly identified payee and " + " and ".join(reasons) + "." + note,
                {"is_new_payee": 1, "amount": amount, "ratio": None if ratio is None else round(ratio, 2),
                 "currency": currency},
            ))

    if partners:
        findings.append(_finding(
            R_DUPLICATE, cfg.duplicate_severity, cfg.duplicate_points,
            f"Matches {len(partners)} other record(s) on amount, currency, category, channel and baseline "
            f"within {cfg.duplicate_window_minutes} minutes. Could be a legitimate repeat or a data duplicate.",
            {"matching_transaction_ids": partners[:5], "window_minutes": cfg.duplicate_window_minutes},
        ))

    ts = row["timestamp"]
    if pd.notna(ts) and ratio is not None and ratio >= cfg.night_min_ratio:
        if _in_night_window(int(ts.hour), cfg.night_start_hour, cfg.night_end_hour):
            findings.append(_finding(
                R_TIMING, cfg.timing_severity, cfg.timing_points,
                f"Transaction at {ts.strftime('%H:%M')} (night window "
                f"{cfg.night_start_hour:02d}:00-{cfg.night_end_hour:02d}:00) with amount {ratio:.1f}x baseline.",
                {"hour": int(ts.hour), "ratio": round(ratio, 2)},
            ))

    if baseline <= 0 and amount > 0:
        findings.append(_finding(
            R_DATA_QUALITY, cfg.data_quality_severity, cfg.data_quality_points,
            "Account baseline is zero, so ratio-based rules could not be evaluated. Review the source data.",
            {"account_avg_amount": baseline, "amount": amount},
        ))

    return findings


def score_findings(findings: List[dict]) -> float:
    """Sum rule points and clamp to the 0-100 range."""
    total = sum(f["points"] for f in findings)
    return float(max(0.0, min(100.0, total)))


def explain_score(findings: List[dict], cfg: DetectionConfig) -> str:
    """Plain-language explanation of how the score was calculated."""
    if not findings:
        return "No rules were triggered, so the illustrative risk score is 0."
    lines = [f"- {f['rule_id']}: +{f['points']:g} points ({f['severity']} severity)" for f in findings]
    raw = sum(f["points"] for f in findings)
    final = score_findings(findings)
    tail = f"Sum = {raw:g}"
    if raw > 100:
        tail += f", capped at 100"
    tail += (f". Levels: Low < {cfg.medium_threshold} <= Medium < {cfg.high_threshold} <= High. "
             "This is an illustrative indicator, not a probability of fraud.")
    return "\n".join(lines) + f"\n{tail}\nFinal score: {final:g}"


def analyze(df: pd.DataFrame, cfg: DetectionConfig | None = None) -> AnalysisResult:
    """Run all rules over a validated DataFrame and return the enriched results."""
    cfg = cfg or DetectionConfig()
    out = df.copy().reset_index(drop=True)

    if out.empty:
        for col, default in (("risk_score", 0.0), ("risk_level", "None"), ("rules_triggered", ""), ("n_rules", 0)):
            out[col] = default
        out["findings"] = [[] for _ in range(len(out))]
        return AnalysisResult(out, cfg, analyzed_at=pd.Timestamp.now().isoformat(timespec="seconds"))

    partners = _duplicate_partners(out, cfg)
    all_findings, scores, levels, rule_strs, counts = [], [], [], [], []
    for idx, row in out.iterrows():
        f = evaluate_row(row, partners.get(idx, []), cfg)
        s = score_findings(f)
        all_findings.append(f)
        scores.append(s)
        levels.append(level_for_score(s, cfg))
        rule_strs.append(";".join(x["rule_id"] for x in f))
        counts.append(len(f))

    out["risk_score"] = scores
    out["risk_level"] = levels
    out["rules_triggered"] = rule_strs
    out["n_rules"] = counts
    out["findings"] = all_findings

    totals = out.groupby("currency")["amount"].sum().to_dict()
    return AnalysisResult(
        data=out,
        config=cfg,
        analyzed_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        currency_totals={k: float(v) for k, v in totals.items()},
    )
