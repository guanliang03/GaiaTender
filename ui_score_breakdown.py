# ui_score_breakdown.py
# ─────────────────────────────────────────────────────────────────────────────
# Reusable Streamlit component that renders the full per-rule score breakdown
# from a PredictionResult.  Used in both the New Entry preview panel and the
# Report tab.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import streamlit as st

from ai_engine import PredictionResult


# Colour for each score band (% of max)
def _band_colour(pct: float) -> str:
    if pct >= 80:
        return "green"
    if pct >= 50:
        return "orange"
    return "red"


def render_breakdown(result: PredictionResult, compact: bool = False) -> None:
    """
    Render an expandable score breakdown card.

    Parameters
    ----------
    result  : PredictionResult returned by ai_engine.predict()
    compact : If True, shows a minimal one-line summary per rule (for the
              New Entry sidebar panel). If False, shows full rationale cards
              (for the Report tab).
    """
    # ── Top-line probability + risk badge + confidence ───────────────────────
    risk_col = result.risk_colour
    conf_colours = {
        "High":         "green",
        "Moderate":     "blue",
        "Low":          "orange",
        "Insufficient": "red",
    }
    conf_col = conf_colours.get(result.confidence_level, "gray")

    st.markdown(
        f"### :{risk_col}[{result.risk_level}]  —  **{result.probability}% Win Probability**  "
        f"&nbsp;&nbsp; :{conf_col}[🔍 Confidence: {result.confidence_level}]"
    )

    # ── Score bar ─────────────────────────────────────────────────────────────
    total_pts = sum(r.score for r in result.rules)
    total_max = result.total_max
    st.progress(total_pts / total_max if total_max else 0)
    st.caption(f"Total score: **{total_pts} / {total_max}** points")

    # ── Red flags callout ─────────────────────────────────────────────────────
    if result.red_flags:
        with st.container():
            st.warning(
                "**🚨 Red Flags Detected — Critically Weak Dimensions:**\n\n"
                + "\n".join(f"- {f}" for f in result.red_flags)
            )

    # ── Action summary ────────────────────────────────────────────────────────
    if result.action_summary:
        with st.expander("💡 AI Recommendation", expanded=True):
            st.markdown(result.action_summary)

    st.divider()

    if compact:
        _render_compact(result)
    else:
        _render_full(result)


def _render_compact(result: PredictionResult) -> None:
    """Metrics rendered in a 2-column grid to prevent overlapping in sidebar."""
    cols = st.columns(2)
    for i, rule in enumerate(result.rules):
        col = cols[i % 2]
        colour = _band_colour(rule.pct)
        col.metric(
            label=rule.name,
            value=f"{rule.score}/{rule.max_score}",
            help=rule.rationale,
        )
        col.caption(f":{colour}[{rule.verdict}]")


def _render_full(result: PredictionResult) -> None:
    """One expander card per rule with full rationale and data source."""
    for rule in result.rules:
        colour = _band_colour(rule.pct)
        pct_bar = int(rule.pct)

        with st.expander(
            f"{rule.verdict}  ·  **{rule.name}**  ·  {rule.score}/{rule.max_score} pts",
            expanded=(rule.pct < 60),   # auto-expand weak rules to draw attention
        ):
            # Mini progress bar for this rule's score
            st.progress(pct_bar / 100)

            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**Why:** {rule.rationale}")
            with c2:
                st.markdown(f"**Data used:**  \n_{rule.data_source}_")
