# charts.py
# ─────────────────────────────────────────────────────────────────────────────
# Chart builders and PDF export.
# Returns figures / BytesIO objects — no Streamlit calls.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
from datetime import date
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from config import ATTRIBUTES


def build_attribute_chart(df: pd.DataFrame, title: str) -> Optional[plt.Figure]:
    """Diverging bar chart: wins (green, up) vs losses (red, down) per attribute."""
    completed = df[df["status"].isin(["Won", "Lost"])]
    if completed.empty:
        return None

    wins   = [len(completed[(completed["primary_factor"] == a) & (completed["status"] == "Won")])  for a in ATTRIBUTES]
    losses = [-len(completed[(completed["primary_factor"] == a) & (completed["status"] == "Lost")]) for a in ATTRIBUTES]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(ATTRIBUTES, wins,   color="#4CAF50", label="Win",  alpha=0.88)
    ax.bar(ATTRIBUTES, losses, color="#F44336", label="Loss", alpha=0.88)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize="small")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def fig_to_buffer(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def build_pdf_report(
    report_row: pd.Series,
    risk_label: str,
    hist_performance: str,
    chart_buffer: Optional[io.BytesIO],
    chart_title: str,
    rules_summary: str = "",
) -> io.BytesIO:
    """Single-page PDF report. Returns BytesIO ready for st.download_button."""
    out = io.BytesIO()
    c   = canvas.Canvas(out, pagesize=letter)
    w, h = letter

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, h - 50, "Lab Supply Tender Report")
    c.setFont("Helvetica", 11)
    c.drawString(50, h - 75, f"Generated: {date.today().strftime('%d %B %Y')}")
    c.line(50, h - 85, w - 50, h - 85)

    # Section 1 — Financials
    y = h - 110
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "1. Financials & Performance")
    y -= 22
    c.setFont("Helvetica", 11)

    for line in [
        f"Project  : {report_row['project_name']}",
        f"Client   : {report_row['client_name']}",
        f"Budget   : RM {report_row['value']:,.2f}",
        f"Product  : {report_row.get('product_brand', '')} {report_row.get('product_model', '')}".strip(),
        f"Client Win Rate: {hist_performance}",
        f"Lead     : {report_row['assignee']}",
        f"Key Driver: {report_row['primary_factor']}",
        f"Submission: {report_row.get('submission_method', 'N/A')}",
    ]:
        c.drawString(65, y, line)
        y -= 18

    # Section 2 — AI Prediction
    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "2. AI Prediction")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(65, y, f"Win Probability : {report_row['win_prob']}%")
    y -= 18
    c.drawString(65, y, f"Risk Level      : {risk_label}")

    if rules_summary:
        y -= 18
        c.setFont("Helvetica-Oblique", 10)
        # Wrap long rules summary string at ~90 chars
        words = rules_summary.split()
        line_buf, line_len = [], 0
        for word in words:
            if line_len + len(word) + 1 > 90:
                c.drawString(65, y, " ".join(line_buf))
                y -= 14
                line_buf, line_len = [word], len(word)
            else:
                line_buf.append(word)
                line_len += len(word) + 1
        if line_buf:
            c.drawString(65, y, " ".join(line_buf))
            y -= 14

    # Section 3 — Chart
    if chart_title:
        y -= 15
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, f"3. {chart_title}")
        y -= 200
        if chart_buffer:
            chart_buffer.seek(0)
            c.drawImage(ImageReader(chart_buffer), 90, y, width=400, height=190)

    c.save()
    out.seek(0)
    return out
