# FraudLens AI

Explainable transaction-monitoring dashboard for education and human-assisted investigation.
Flags are **for human review only**. A risk score is an illustrative indicator, **not** a probability of fraud, and nothing here confirms fraud. This is an educational prototype, not a certified banking security product.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Streamlit dashboard, synthetic data, CSV validation, deterministic rules, transparent score, filters, CSV download, tests | Implemented (see "Verification") |
| 2 | Local Gemma 3:4B via Ollama, structured response validation, fallback | Planned |
| 3 | Case workflow, notes, SQLite, investigation report export | Planned |
| 4 | Anomaly detection, scenario generator, evaluation tools | Planned |
| 5 | Docs, contribution guide, license, screenshots | Planned |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_sample_data.py   # regenerates data/sample_transactions.csv
streamlit run app.py
python -m pytest                         # run the tests
```

No API keys are needed. Everything runs locally.

## Data schema

Required columns: `transaction_id`, `timestamp`, `amount`, `currency` (3-letter code), `account_avg_amount`, `transactions_last_hour`, `is_new_payee` (0/1 or true/false), `merchant_category`, `channel`.
Optional: `country_code`, `transaction_description` (never fabricated when absent; excluded from exports by default).

Rows that fail validation are excluded and listed with their CSV line number and the problem. Uploads must be `.csv`, up to 5 MB and 50,000 rows. Currencies are never summed together.

## Detection rules (ruleset 1.0.0)

| ID | Rule | Default |
|---|---|---|
| R001 | Amount vs account baseline | ratio >= 5x, 30 pts |
| R002 | Velocity burst (`transactions_last_hour`) | >= 5, 25 pts |
| R003 | High-value payment to new payee | ratio >= 3x, or >= 25,000 INR, 25 pts (halved if R001 also fired, to limit double counting) |
| R004 | Possible duplicate | same amount/currency/category/channel/baseline within 10 min, 15 pts |
| R005 | Unusual timing | 00:00-05:00 and amount >= 2x baseline, 15 pts |
| R006 | Data quality (zero baseline) | 5 pts |

Score = sum of points, capped at 0-100. Levels: Low < 30 <= Medium < 60 <= High. All thresholds are adjustable in the sidebar.

## Layout

- `app.py` - root Streamlit launcher (`streamlit run app.py`)
- `frontend/app.py` - Streamlit pages, charts, filters, and upload UI
- `backend/src/validation.py`, `backend/src/data_loader.py` - safe CSV ingestion
- `backend/src/detection_engine.py` - rules and scoring (no Streamlit dependency)
- `backend/src/reporting.py` - CSV export (neutralizes spreadsheet formulas)
- `tests/` - pytest suite; `data/` - fictional sample data

The frontend and backend are folders in one Python application and run in the
same Streamlit process. The AI explanation integration is planned for Phase 2;
the rule-based detection and scores work independently of Ollama.

## Verification

Run in a development sandbox: the pytest suite passes (38 tests) and `app.py` compiles. The Streamlit UI itself has **not** been launched yet because the sandbox could not install Streamlit. Please run `streamlit run app.py` locally and work through the checklist below.

## Manual checklist

1. Overview shows 129 transactions, High/Medium/Low counts, and INR and USD totals shown separately.
2. Transaction Explorer: search `TXN2005`, filter by rule, change page size.
3. Alert Investigation: open `TXN2005`; confirm several rules, evidence JSON and the score breakdown.
4. Change a threshold in the sidebar and confirm scores update.
5. Upload a CSV with a bad row and confirm it is listed under "Rows requiring correction".
6. Reports: download the CSV; confirm no description column by default.
