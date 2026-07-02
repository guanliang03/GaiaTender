# Smart Tender System — v2

## Quick Start
```bash
pip install streamlit pandas altair matplotlib reportlab openpyxl "numpy<2" "scipy<1.12" pypdfium2 easyocr
streamlit run app.py
```

The database starts **completely empty** — no dummy data.  
On first launch you will see the onboarding screen with a CSV template to download and fill in.

---

## File Structure

```
tender_system/
├── app.py                    # Entry point — sidebar, onboarding gate, tabs
├── config.py                 # ALL constants: weights, thresholds, stage lists
├── database.py               # SQLite init + CRUD (no seed data)
├── ai_engine.py              # 5-rule scoring engine + analytics helpers
├── charts.py                 # Matplotlib charts + ReportLab PDF export
├── ui_onboarding.py          # Empty-state screen with CSV template download
├── ui_score_breakdown.py     # Reusable score breakdown widget
├── ui_tab_new_entry.py       # Tab 1: manual form + bulk upload
├── ui_tab_pipeline.py        # Tab 2: editable grid + workload charts
├── ui_tab_report.py          # Tab 3: full breakdown card + PDF download
└── ui_tab_performance.py     # Tab 4: analytics + relationship depth table
```

---

## AI Scoring Rules (5 Components, 100 pts max)

| # | Rule | Max | Logic |
|---|------|-----|-------|
| 1 | **Price** | 30 | Bid ratio vs historical winning ratio for this client (or market avg). Over-budget = 3 pts. |
| 2 | **Strategy** | 25 | Does the chosen attribute match what has won with this client? Falls back to market-wide data. |
| 3 | **Assignee** | 20 | Lead's personal win/loss ratio across all closed deals. New leads = neutral 50%. |
| 4 | **Relationship Depth** | 15 | Combines: how many closed deals with this client × quality of those deals (win rate). |
| 5 | **Value Fit** | 10 | Is the project value within ±40% of the median won deal size? Far outside = lower score. |

### Rule 4 — Relationship Depth in detail

| Depth tier | Closed deals | Multiplier |
|---|---|---|
| Strong | ≥ 3 | 1.00 |
| Developing | 1–2 | 0.65 |
| Thin / New | 0 | 0.20 |

Quality multiplier (applied on top of depth):

| Win rate with client | Multiplier |
|---|---|
| ≥ 50% | 1.00 |
| > 0% | 0.65 |
| 0% (never won) | 0.25 |

---

## Score Breakdown UI

Every tender shows a **full per-rule breakdown**:
- In **New Entry** → compact 5-tile strip updates live as you type.
- In **Report tab** → full expandable cards per rule showing:
  - Points scored / max
  - Verdict emoji + label
  - Plain-English rationale ("why this score was given")
  - Data source ("3 client wins used as benchmark")
  - Mini progress bar per rule

Weak-scoring rules (< 60%) auto-expand to draw attention.

---

## Changing Weights

All weights and thresholds are in `config.py`. Example — to make relationship
depth worth more at the cost of price:

```python
WEIGHT_PRICE        = 25   # was 30
WEIGHT_RELATIONSHIP = 20   # was 15
```

No other file needs to change.
