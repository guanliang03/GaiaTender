# ui_tab_performance.py
# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Performance: win/loss analytics, strategy rankings, client
# relationship depth table, and strategic recommendations.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ai_engine import (
    attribute_win_rates,
    client_relationship_table,
    strategic_recommendation,
)
from config import ATTRIBUTES


def render(df_master: pd.DataFrame) -> None:
    st.header("🧠 Performance & Strategy Monitor")

    completed = df_master[df_master["status"].isin(["Won", "Lost"])]

    if completed.empty:
        st.info(
            "No closed tenders yet. Once you have **Won** or **Lost** records, "
            "this tab will show analytics and recommendations."
        )
        return

    _render_kpis(completed)
    st.divider()
    _render_win_loss_chart(completed)
    st.divider()
    _render_insights(completed)
    st.divider()
    _render_relationship_table(df_master)


# ── KPI strip ─────────────────────────────────────────────────────────────────

def _render_kpis(completed: pd.DataFrame) -> None:
    total    = len(completed)
    wins     = len(completed[completed["status"] == "Won"])
    win_rate = wins / total * 100

    total_value_won  = completed.loc[completed["status"] == "Won", "value"].sum()
    avg_deal_won     = completed.loc[completed["status"] == "Won", "value"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall Win Rate",   f"{win_rate:.1f}%")
    k2.metric("Closed Tenders",     total)
    k3.metric("Won Deal Value",     f"RM {total_value_won:,.0f}")
    k4.metric("Avg Won Deal Size",  f"RM {avg_deal_won:,.0f}" if wins else "—")


# ── Normalised stacked bar chart ──────────────────────────────────────────────

def _render_win_loss_chart(completed: pd.DataFrame) -> None:
    st.subheader("📊 Win/Loss Analysis by Competitive Attribute")

    attr_perf = (
        completed
        .groupby(["primary_factor", "status"])
        .size()
        .reset_index(name="Count")
    )

    chart = (
        alt.Chart(attr_perf)
        .mark_bar()
        .encode(
            x=alt.X(
                "Count",
                stack="normalize",
                axis=alt.Axis(format="%", title="Win/Loss Ratio"),
            ),
            y=alt.Y("primary_factor", title="Attribute", sort=ATTRIBUTES),
            color=alt.Color(
                "status",
                scale=alt.Scale(domain=["Won", "Lost"], range=["#2ecc71", "#e74c3c"]),
                legend=alt.Legend(title="Result"),
            ),
            tooltip=["primary_factor", "status", "Count"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


# ── Insights & recommendations ────────────────────────────────────────────────

def _render_insights(completed: pd.DataFrame) -> None:
    st.subheader("💡 Strategic Insights")

    stats = attribute_win_rates(completed)
    if not stats:
        st.warning("Insufficient data to rank attributes.")
        return

    # Top 3 drivers
    top_3 = stats[:3]
    worst = stats[-1]

    st.markdown("#### 🏆 Top Performing Drivers")
    cols = st.columns(len(top_3))
    for i, (attr, rate, wins, total) in enumerate(top_3):
        with cols[i]:
            colour = "green" if rate >= 60 else ("orange" if rate >= 40 else "red")
            st.markdown(
                f"**#{i+1} — {attr}**  \n"
                f":{colour}[**{rate:.1f}% win rate**]  \n"
                f"{wins} won / {total} total"
            )

    # Main recommendation
    st.markdown("#### 🧭 Strategic Recommendation")
    st.info(strategic_recommendation(top_3[0][0]))

    # Action warning
    if worst[1] < 50:
        st.warning(
            f"⚠️ **Action needed:** Your win rate for **'{worst[0]}'** is only "
            f"**{worst[1]:.1f}%** ({worst[2]}/{worst[3]} deals). "
            "Consider improving proposal quality, credentials, or client engagement "
            "when this is the primary driver."
        )

    # Full ranked table
    with st.expander("📋 Full Attribute Ranking"):
        rows = [
            {
                "Rank": i + 1,
                "Attribute": attr,
                "Win Rate %": f"{rate:.1f}",
                "Wins": wins,
                "Total": total,
                "Lost": total - wins,
            }
            for i, (attr, rate, wins, total) in enumerate(stats)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Client relationship depth table ──────────────────────────────────────────

def _render_relationship_table(df: pd.DataFrame) -> None:
    st.subheader("🤝 Client Relationship Depth")

    rel_df = client_relationship_table(df)
    if rel_df.empty:
        st.info("No closed deals yet — relationship data will appear here once tenders are marked Won or Lost.")
        return

    st.markdown(
        "This table shows how deep our relationship is with each client, "
        "based on closed deal history, total contract value, and win rate. "
        "Depth is **value-weighted** (60% total contract value · 40% deal count).  \n"
        "**Strong** = composite depth ≥ 80% · **Developing** = 40–79% · **Thin** = < 40%"
    )

    # Colour-code the Depth column
    def _colour_depth(val: str) -> str:
        return {
            "Strong":     "background-color: #d4edda; color: #155724",
            "Developing": "background-color: #fff3cd; color: #856404",
            "Thin":       "background-color: #f8d7da; color: #721c24",
        }.get(val, "")

    styled = rel_df.style.map(_colour_depth, subset=["Depth"])
    st.dataframe(styled, hide_index=True, use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** Scores, predictions, and recommendations are generated by an AI "
        "scoring engine based on historical patterns. AI can make mistakes — always apply "
        "professional judgement before acting on any recommendation."
    )
