# ui_tab_gaia_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Gaia Data Analysis
# Deep-dive analytics sourced from the imported Sebutharga tender records.
#
# Sections:
#   1. KPI Summary Row
#   2. Win Rate by Salesperson
#   3. Win Rate by Institution / Client
#   4. Outcome Breakdown (donut charts)
#   5. Deal Value Distribution
#   6. Monthly Submission Trend
#   7. Raw data explorer
#
# All charts use Plotly for interactive hover tooltips.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
_GRID   = "#333355"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_CARD,
    font=dict(color="white", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(
        bgcolor="#0f3460",
        bordercolor="#3498db",
        font_size=13,
        font_family="Inter, sans-serif",
    ),
)


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
    tbl["Won"]      = tbl["Won"].fillna(0).astype(int)
    tbl["Lost"]     = tbl["Total"] - tbl["Won"]
    tbl["Win Rate"] = (tbl["Won"] / tbl["Total"] * 100).round(1)
    return tbl.sort_values("Win Rate", ascending=False).reset_index(drop=True)


def _plotly_hbar(
    tbl: pd.DataFrame,
    label_col: str,
    overall_wr: float,
    title: str,
    above_color: str = _GREEN,
    below_color: str = _RED,
    height_per_row: int = 38,
) -> go.Figure:
    """Horizontal bar chart with hover showing rate, won, lost, total."""
    colours = [above_color if wr >= overall_wr else below_color for wr in tbl["Win Rate"]]
    labels  = tbl[label_col].tolist()
    rates   = tbl["Win Rate"].tolist()
    wons    = tbl["Won"].tolist()
    losts   = tbl["Lost"].tolist()
    totals  = tbl["Total"].tolist()

    hover = [
        f"<b>{lbl}</b><br>"
        f"Win Rate: <b>{wr:.1f}%</b><br>"
        f"Won: {w} | Lost: {l} | Total: {t}"
        for lbl, wr, w, l, t in zip(labels, rates, wons, losts, totals)
    ]

    fig = go.Figure(go.Bar(
        x=rates,
        y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
        textfont=dict(color="white", size=11),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))

    chart_h = max(280, len(labels) * height_per_row + 80)
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(
            title="Win Rate (%)",
            color="white",
            gridcolor=_GRID,
            ticksuffix="%",
            range=[0, max(rates) * 1.2 if rates else 100],
        ),
        yaxis=dict(
            color="white",
            autorange="reversed",
            tickfont=dict(size=10),
        ),
        height=chart_h,
        showlegend=False,
    )
    return fig


def _plotly_donut(
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
) -> go.Figure:
    """Donut chart with hover showing label, value, and percentage."""
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color=_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color="white", size=12),
        hovertemplate="<b>%{label}</b><br>Count / Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=13)),
        legend=dict(font=dict(color="white"), bgcolor=_CARD),
        height=380,
    )
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
            fig = _plotly_hbar(
                sp_tbl.rename(columns={"assignee": "Salesperson"}),
                "Salesperson",
                overall_wr,
                "Win Rate by Salesperson (%)",
                above_color=_GREEN,
                below_color=_RED,
            )
            st.plotly_chart(fig, use_container_width=True)
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
        top_inst = (
            inst_tbl.sort_values("Total", ascending=False)
            .head(15)
            .sort_values("Win Rate", ascending=False)
        )

        col_chart2, col_table2 = st.columns([3, 2])
        with col_chart2:
            fig2 = _plotly_hbar(
                top_inst.rename(columns={"client_name": "Institution"}),
                "Institution",
                overall_wr,
                "Win Rate by Institution — Top 15 by Volume (%)",
                above_color=_GREEN,
                below_color=_ORANGE,
            )
            st.plotly_chart(fig2, use_container_width=True)
        with col_table2:
            display2 = inst_tbl.rename(columns={"client_name": "Institution"})
            st.dataframe(
                display2[["Institution", "Total", "Won", "Lost", "Win Rate"]],
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── 4. Outcome donut charts ────────────────────────────────────────────────
    st.markdown("### 📊 Outcome Breakdown")
    c_pie1, c_pie2 = st.columns(2)

    with c_pie1:
        fig3 = _plotly_donut(
            ["Won", "Lost"],
            [total_won, total_lost],
            [_GREEN, _RED],
            "Overall Outcome (Closed Tenders)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c_pie2:
        top5 = (
            closed[closed["status"] == "Won"]
            .groupby("assignee")["value"].sum()
            .nlargest(5)
        )
        if not top5.empty:
            fig4 = _plotly_donut(
                top5.index.tolist(),
                top5.values.tolist(),
                [_GREEN, _BLUE, _ORANGE, _PURPLE, _GREY],
                "Won Contract Value — Top 5 Salespeople",
            )
            # Override hover for value chart to show RM
            fig4.update_traces(
                hovertemplate="<b>%{label}</b><br>Won Value: RM %{value:,.0f}<br>Share: %{percent}<extra></extra>"
            )
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── 5. Deal value distribution ─────────────────────────────────────────────
    st.markdown("### 💰 Deal Value Distribution")
    won_vals  = closed[closed["status"] == "Won"]["value"].dropna()
    lost_vals = closed[closed["status"] == "Lost"]["value"].dropna()

    if not won_vals.empty:
        cap = min(closed["value"].max(), 2_000_000)
        bins = np.linspace(0, cap, 31)

        won_counts,  won_edges  = np.histogram(won_vals,  bins=bins)
        lost_counts, lost_edges = np.histogram(lost_vals, bins=bins)

        def _bin_label(edges: np.ndarray, i: int) -> str:
            lo = edges[i] / 1000
            hi = edges[i + 1] / 1000
            return f"RM {lo:.0f}K – {hi:.0f}K"

        won_hover  = [f"<b>{_bin_label(won_edges,  i)}</b><br>Won:  {c} tenders" for i, c in enumerate(won_counts)]
        lost_hover = [f"<b>{_bin_label(lost_edges, i)}</b><br>Lost: {c} tenders" for i, c in enumerate(lost_counts)]
        bin_centres = (bins[:-1] + bins[1:]) / 2

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=bin_centres,
            y=won_counts,
            name="Won",
            marker_color=_GREEN,
            opacity=0.8,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=won_hover,
        ))
        fig5.add_trace(go.Bar(
            x=bin_centres,
            y=lost_counts,
            name="Lost",
            marker_color=_RED,
            opacity=0.8,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=lost_hover,
        ))
        fig5.update_layout(
            **_PLOTLY_LAYOUT,
            title=dict(text="Distribution of Won vs Lost Tender Values", font=dict(size=14)),
            barmode="overlay",
            xaxis=dict(
                title="Tender Value (RM)",
                color="white",
                gridcolor=_GRID,
                tickformat=",.0f",
                tickprefix="RM ",
            ),
            yaxis=dict(title="Count", color="white", gridcolor=_GRID),
            legend=dict(font=dict(color="white"), bgcolor=_CARD),
            height=380,
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # ── 6. Monthly submission trend ────────────────────────────────────────────
    st.markdown("### 📅 Monthly Submission Trend")
    df_trend = df.copy()
    df_trend["deadline"] = pd.to_datetime(df_trend["deadline"], errors="coerce")
    # Drop outlier dates before year 2000 (e.g. 1990 bad data entries)
    df_trend = df_trend[df_trend["deadline"].dt.year >= 2000]
    df_trend["month"]    = df_trend["deadline"].dt.to_period("M")
    trend = df_trend.groupby(["month", "status"]).size().unstack(fill_value=0)

    if not trend.empty:
        trend.index = trend.index.astype(str)
        months = trend.index.tolist()

        won_col  = trend["Won"].tolist()  if "Won"  in trend.columns else [0] * len(months)
        lost_col = trend["Lost"].tolist() if "Lost" in trend.columns else [0] * len(months)
        other_cols = [c for c in trend.columns if c not in ("Won", "Lost")]
        other_col = trend[other_cols].sum(axis=1).tolist() if other_cols else [0] * len(months)

        won_hover   = [f"<b>{m}</b><br>Won: {w}" for m, w in zip(months, won_col)]
        lost_hover  = [f"<b>{m}</b><br>Lost: {l}" for m, l in zip(months, lost_col)]
        other_hover = [f"<b>{m}</b><br>Other/Pending: {o}" for m, o in zip(months, other_col)]

        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=months, y=won_col,
            name="Won",
            marker_color=_GREEN,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=won_hover,
        ))
        fig6.add_trace(go.Bar(
            x=months, y=lost_col,
            name="Lost",
            marker_color=_RED,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=lost_hover,
        ))
        if any(o > 0 for o in other_col):
            fig6.add_trace(go.Bar(
                x=months, y=other_col,
                name="Other",
                marker_color=_GREY,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=other_hover,
            ))
        fig6.update_layout(
            **_PLOTLY_LAYOUT,
            title=dict(text="Tenders by Month (stacked: Won / Lost / Other)", font=dict(size=14)),
            barmode="stack",
            xaxis=dict(
                title="Month",
                color="white",
                gridcolor=_GRID,
                tickangle=-45,
            ),
            yaxis=dict(title="No. of Tenders", color="white", gridcolor=_GRID),
            legend=dict(font=dict(color="white"), bgcolor=_CARD),
            height=400,
        )
        st.plotly_chart(fig6, use_container_width=True)

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
