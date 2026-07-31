# ui_tab_report.py
# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Report: project card + FULL AI score breakdown + PDF download.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import base64
import os

import pandas as pd
import streamlit as st

from ai_engine import predict
from charts import build_pdf_report
from ui_score_breakdown import render_breakdown


def render(df_master: pd.DataFrame) -> None:
    st.header("📄 Tender Report")

    if df_master.empty:
        st.info("No tenders in the system yet. Add tenders via the *📝 New Entry* tab.")
        return

    sel = st.selectbox(
        "Select Project",
        df_master["project_name"].unique(),
        help="Choose a tender to generate its detailed report.",
    )
    row = df_master[df_master["project_name"] == sel].iloc[0]

    hist_str = _history_summary(row["client_name"], df_master)
    risk     = _risk_label(row["win_prob"])

    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(f"## {row['project_name']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Budget",          f"RM {row['value']:,.0f}")
        m2.metric("Client Win Rate", hist_str)
        m3.metric("Status",          row["status"])
        m4.metric("AI Probability",  f"{row['win_prob']:.0f}%")

        details = {
            "Client":                row["client_name"],
            "Lead":                  row["assignee"],
            "Key Driver / Attribute":row["primary_factor"],
            "Product Brand":         row.get("product_brand", "—") or "—",
            "Product Model":         row.get("product_model", "—") or "—",
            "Submission Method":     row.get("submission_method", "—") or "—",
            "Start Date":            str(row.get("starting_date", "—") or "—"),
            "Deadline":              str(row["deadline"]),
            "Overall Risk":          risk,
        }
        st.table(pd.DataFrame(details.items(), columns=["Metric", "Value"]))

    st.divider()

    # ── Full AI score breakdown ───────────────────────────────────────────────
    st.subheader("🧠 AI Score Breakdown — Why this probability?")
    result = predict(
        row["value"], row["client_name"],
        row["primary_factor"], row["assignee"], df_master,
        deadline=row.get("deadline"),
        product_brand=str(row.get("product_brand", "") or ""),
        product_model=str(row.get("product_model", "") or ""),
        project_name=str(row.get("project_name", "") or ""),
        status=row.get("status", ""),
    )
    render_breakdown(result, compact=False)

    st.divider()

    # ── Submission PDF attachment ──────────────────────────────────────────
    pdf_path = str(row.get("pdf_path", "") or "")
    if pdf_path and os.path.isfile(pdf_path):
        st.subheader("📎 Submission PDF")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode()
        filename = os.path.basename(pdf_path)

        col_dl, _ = st.columns([1, 3])
        col_dl.download_button(
            "⬇️ Download Submission PDF",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
        )

        # Inline preview (works in most browsers)
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="700px" style="border:none;border-radius:8px;"></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
    elif pdf_path:
        st.warning("⚠️ Submission PDF was recorded but the file could not be found on disk.")

    st.divider()

    # ── PDF download ──────────────────────────────────────────────────────
    rules_summary = result.insight_str
    pdf = build_pdf_report(row, risk, hist_str, None, "", rules_summary)
    st.download_button(
        "📥 Download AI Report (PDF)",
        data=pdf,
        file_name=f"report_{sel.replace(' ', '_')}.pdf",
        mime="application/pdf",
        type="primary",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _history_summary(client_name: str, df: pd.DataFrame) -> str:
    hist = df[
        (df["client_name"] == client_name)
        & (df["status"].isin(["Won", "Lost"]))
    ]
    if hist.empty:
        return "No History"
    total = len(hist)
    wins  = len(hist[hist["status"] == "Won"])
    return f"{wins / total * 100:.1f}%  ({wins}/{total} Won)"


def _risk_label(win_prob: float) -> str:
    if win_prob < 40:
        return "High Risk 🔴 (Consider Deprioritising)"
    if win_prob <= 70:
        return "Medium Risk 🟠"
    return "Low Risk 🟢 (High Potential - Focus)"
