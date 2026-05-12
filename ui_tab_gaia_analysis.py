# ui_tab_gaia_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Gaia Data Analysis
# Deep-dive analytics sourced from the imported Sebutharga tender records.
#
# Sections:
#   1. KPI Summary Row
#   2. Win Rate by Salesperson
#   3. Win Rate by Institution / Client
#   4. Win Rate by Submission Method
#   5. Deal Value Distribution
#   6. Monthly Submission Trend
#   7. Raw data explorer
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st


# ── Colour palette ─────────────────────────────────────────────────────────────
_GREEN  = "#2ecc71"
_RED    = "#e74c3c"
_BLUE   = "#3498db"
_ORANGE = "#f39c12"
_PURPLE = "#9b59b6"
_GREY   = "#95a5a6"
_BG     = "#1a1a2e"
_CARD   = "#16213e"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _closed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].isin(["Won", "Lost"])].copy()


def _win_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return (group, total, wins, win_rate_pct) sorted by win_rate desc."""
    closed = _closed(df)
    if closed.empty:
        return pd.DataFrame()
    grp   = closed.groupby(group_col)
    total = grp.size().rename("Total")
    wins  = closed[closed["status"] == "Won"].groupby(group_col).size().rename("Won")
    tbl   = pd.concat([total, wins], axis=1).reset_index()
    tbl["Lost"]     = tbl["Total"] - tbl["Won"]
    tbl["Win Rate"] = (tbl["Won"] / tbl["Total"] * 100).round(1)
    return tbl.sort_values("Win Rate", ascending=False).reset_index(drop=True)


def _bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    color: str = _BLUE,
    ylabel: str = "Win Rate (%)",
    fmt: str = "{:.1f}%",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.set_facecolor(_CARD)

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, values, color=color, height=0.6, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color="white")
    ax.invert_yaxis()
    ax.set_xlabel(ylabel, color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=11, pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333355")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.grid(axis="x", color="#333355", linewidth=0.5, zorder=1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            fmt.format(val), va="center", ha="left", fontsize=8, color="white"
        )
    fig.tight_layout()
    return fig


def _donut(labels: list[str], sizes: list[float], colors: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4), facecolor=_BG)
    ax.set_facecolor(_BG)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors, pctdistance=0.8, wedgeprops=dict(width=0.5),
    )
    for t in texts:
        t.set_color("white"); t.set_fontsize(9)
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(8)
    ax.set_title(title, color="white", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


# ── Main render ────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame) -> None:
    st.subheader("🏢 Gaia Tender Data Analysis")

    if df.empty:
        st.info("No data yet. Run `python import_gaia.py` to load the Sebutharga CSV files.")
        return

    closed = _closed(df)

    if closed.empty:
        st.warning("No closed (Won/Lost) tenders found. Import the Gaia CSVs first.")
        return

    # ── 1. KPI row ─────────────────────────────────────────────────────────────
    total_all  = len(df)
    total_cl   = len(closed)
    total_won  = int((closed["status"] == "Won").sum())
    total_lost = int((closed["status"] == "Lost").sum())
    pending    = total_all - total_cl
    overall_wr = total_won / total_cl * 100 if total_cl else 0
    total_val  = closed[closed["status"] == "Won"]["value"].sum()
    avg_val    = closed[closed["status"] == "Won"]["value"].mean() if total_won else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Tenders",   f"{total_all:,}")
    k2.metric("Closed",          f"{total_cl:,}")
    k3.metric("Won",             f"{total_won:,}",  delta=f"{overall_wr:.1f}% win rate")
    k4.metric("Lost",            f"{total_lost:,}")
    k5.metric("Pending",         f"{pending:,}")
    k6.metric("Total Won Value", f"RM {total_val/1_000_000:.2f}M")

    st.divider()

    # ── 2. Win rate by Salesperson ─────────────────────────────────────────────
    st.markdown("### 👤 Win Rate by Salesperson")
    sp_tbl = _win_rate_table(df, "assignee")

    if not sp_tbl.empty:
        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            # Colour bars: green if above average, red if below
            colours = [_GREEN if wr >= overall_wr else _RED for wr in sp_tbl["Win Rate"]]
            fig = _bar_chart(
                sp_tbl["assignee"].tolist(),
                sp_tbl["Win Rate"].tolist(),
                "Win Rate by Salesperson (%)",
                color=colours,
                figsize=(7, max(4, len(sp_tbl) * 0.45)),
            )
            st.pyplot(fig)
            plt.close(fig)
        with col_table:
            display = sp_tbl.rename(columns={"assignee": "Salesperson"})
            st.dataframe(
                display[["Salesperson", "Total", "Won", "Lost", "Win Rate"]],
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── 3. Win rate by Institution ─────────────────────────────────────────────
    st.markdown("### 🏫 Win Rate by Institution")
    inst_tbl = _win_rate_table(df, "client_name")

    if not inst_tbl.empty:
        # Show top 15 by volume, not just win rate, for relevance
        top_inst = (
            inst_tbl.sort_values("Total", ascending=False)
            .head(15)
            .sort_values("Win Rate", ascending=False)
        )

        col_chart2, col_table2 = st.columns([3, 2])
        with col_chart2:
            colours2 = [_GREEN if wr >= overall_wr else _ORANGE for wr in top_inst["Win Rate"]]
            fig2 = _bar_chart(
                top_inst["client_name"].tolist(),
                top_inst["Win Rate"].tolist(),
                "Win Rate by Institution — Top 15 by Volume (%)",
                color=colours2,
                figsize=(7, max(4, len(top_inst) * 0.45)),
            )
            st.pyplot(fig2)
            plt.close(fig2)
        with col_table2:
            display2 = inst_tbl.rename(columns={"client_name": "Institution"})
            st.dataframe(
                display2[["Institution", "Total", "Won", "Lost", "Win Rate"]],
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── 4. Win / Lost outcome donut  ──────────────────────────────────────────
    st.markdown("### 📊 Outcome Breakdown")
    c_pie1, c_pie2 = st.columns(2)

    with c_pie1:
        fig3 = _donut(
            ["Won", "Lost"],
            [total_won, total_lost],
            [_GREEN, _RED],
            "Overall Outcome (Closed Tenders)",
        )
        st.pyplot(fig3)
        plt.close(fig3)

    with c_pie2:
        # Submission method breakdown from original data if available
        # We map Submission_Method → stored in primary_factor (workaround)
        # Instead: show won value distribution by assignee top 5
        top5 = (
            closed[closed["status"] == "Won"]
            .groupby("assignee")["value"].sum()
            .nlargest(5)
        )
        if not top5.empty:
            fig4 = _donut(
                top5.index.tolist(),
                top5.values.tolist(),
                [_GREEN, _BLUE, _ORANGE, _PURPLE, _GREY],
                "Won Contract Value — Top 5 Salespeople",
            )
            st.pyplot(fig4)
            plt.close(fig4)

    st.divider()

    # ── 5. Deal value distribution ─────────────────────────────────────────────
    st.markdown("### 💰 Deal Value Distribution")
    won_vals  = closed[closed["status"] == "Won"]["value"].dropna()
    lost_vals = closed[closed["status"] == "Lost"]["value"].dropna()

    if not won_vals.empty:
        fig5, ax5 = plt.subplots(figsize=(9, 4), facecolor=_BG)
        ax5.set_facecolor(_CARD)
        bins = np.linspace(0, min(closed["value"].max(), 2_000_000), 30)
        ax5.hist(won_vals,  bins=bins, alpha=0.7, color=_GREEN,  label="Won",  zorder=2)
        ax5.hist(lost_vals, bins=bins, alpha=0.7, color=_RED,    label="Lost", zorder=2)
        ax5.set_xlabel("Tender Value (RM)", color="white")
        ax5.set_ylabel("Count",            color="white")
        ax5.set_title("Distribution of Won vs Lost Tender Values", color="white", fontsize=11)
        ax5.tick_params(colors="white")
        ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"RM {x/1000:.0f}K"))
        for spine in ax5.spines.values():
            spine.set_color("#333355")
        ax5.legend(facecolor=_CARD, edgecolor="#333355", labelcolor="white")
        ax5.grid(axis="y", color="#333355", linewidth=0.5, zorder=1)
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    st.divider()

    # ── 6. Deadline / submission trend ─────────────────────────────────────────
    st.markdown("### 📅 Monthly Submission Trend")
    df_trend = df.copy()
    df_trend["deadline"] = pd.to_datetime(df_trend["deadline"], errors="coerce")
    df_trend["month"]    = df_trend["deadline"].dt.to_period("M")
    trend = df_trend.groupby(["month", "status"]).size().unstack(fill_value=0)

    if not trend.empty:
        trend.index = trend.index.astype(str)
        fig6, ax6 = plt.subplots(figsize=(11, 4), facecolor=_BG)
        ax6.set_facecolor(_CARD)

        if "Won"  in trend.columns:
            ax6.bar(trend.index, trend["Won"],  color=_GREEN, label="Won",    zorder=2)
        if "Lost" in trend.columns:
            ax6.bar(trend.index, trend["Lost"],
                    bottom=trend.get("Won", pd.Series(0, index=trend.index)),
                    color=_RED, label="Lost", zorder=2)
        other_cols = [c for c in trend.columns if c not in ("Won", "Lost")]
        if other_cols:
            base = trend.get("Won", pd.Series(0, index=trend.index)) + \
                   trend.get("Lost", pd.Series(0, index=trend.index))
            ax6.bar(trend.index, trend[other_cols].sum(axis=1),
                    bottom=base, color=_GREY, label="Other", zorder=2)

        ax6.set_xlabel("Month", color="white")
        ax6.set_ylabel("No. of Tenders", color="white")
        ax6.set_title("Tenders by Month (stacked: Won / Lost / Other)", color="white", fontsize=11)
        ax6.tick_params(colors="white", axis="both")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        for spine in ax6.spines.values():
            spine.set_color("#333355")
        ax6.legend(facecolor=_CARD, edgecolor="#333355", labelcolor="white")
        ax6.grid(axis="y", color="#333355", linewidth=0.5, zorder=1)
        fig6.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    st.divider()

    # ── 7. Raw data explorer ───────────────────────────────────────────────────
    st.markdown("### 🔍 Raw Data Explorer")
    filter_status   = st.multiselect("Filter by Status", ["Won", "Lost", "Submitted", "Qualified Lead"], default=["Won", "Lost"])
    filter_assignee = st.multiselect("Filter by Salesperson", sorted(df["assignee"].dropna().unique().tolist()))

    fdf = df.copy()
    if filter_status:
        fdf = fdf[fdf["status"].isin(filter_status)]
    if filter_assignee:
        fdf = fdf[fdf["assignee"].isin(filter_assignee)]

    st.dataframe(
        fdf[["project_name", "client_name", "assignee", "value", "status", "deadline", "primary_factor"]]
          .rename(columns={
              "project_name":   "Project",
              "client_name":    "Institution",
              "assignee":       "Salesperson",
              "value":          "Value (RM)",
              "status":         "Status",
              "deadline":       "Deadline",
              "primary_factor": "Key Driver",
          })
          .sort_values("Value (RM)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing {len(fdf):,} of {len(df):,} records.")
