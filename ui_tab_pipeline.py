# ui_tab_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Pipeline: editable grid, save/delete, workload charts.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import ATTRIBUTES, PIPELINE_STAGES
from database import delete_tender, update_tender


def render(df_master: pd.DataFrame) -> None:
    st.subheader("Pipeline & Workflow")

    if df_master.empty:
        st.info("No tenders in the system yet. Add tenders via the *📝 New Entry* tab.")
        return

    edited = _render_editor(df_master)
    _render_controls(edited, df_master)

    st.divider()
    _render_workload(df_master)


def _render_editor(df: pd.DataFrame) -> pd.DataFrame:
    # All columns except "id" — id is kept in the df for save/delete but never shown
    visible_cols = [c for c in df.columns if c != "id"]
    return st.data_editor(
        df,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        column_order=visible_cols,
        column_config={
            "value":            st.column_config.NumberColumn("Budget",   format="RM %d"),
            "win_prob":         st.column_config.ProgressColumn(
                                    "AI Prob %", format="%d%%", min_value=0, max_value=100
                                ),
            "status":           st.column_config.SelectboxColumn(
                                    "Status", options=PIPELINE_STAGES, required=True
                                ),
            "primary_factor":   st.column_config.TextColumn(
                                    "Key Driver", disabled=True, width="medium"
                                ),
            "starting_date":    st.column_config.DateColumn("Start Date", disabled=True),
            "submission_method":st.column_config.TextColumn("Submission", disabled=True),
            "product_brand":    st.column_config.TextColumn("Brand",      disabled=True),
            "product_model":    st.column_config.TextColumn("Model",      disabled=True),
        },
    )


def _render_controls(edited: pd.DataFrame, original: pd.DataFrame) -> None:
    col_save, col_del = st.columns([1, 5])

    if col_save.button("💾 Save Changes", type="primary"):
        for _, row in edited.iterrows():
            update_tender(
                row["id"], row["project_name"], row["client_name"],
                row["value"], row["status"],
                row["primary_factor"], row["assignee"], str(row["deadline"]),
                row.get("starting_date"), row.get("submission_method", ""),
                row.get("product_brand", ""), row.get("product_model", ""),
            )
        from database import recalculate_all_probabilities
        from ai_engine import predict
        recalculate_all_probabilities(predict)
        st.success("Changes saved.")
        st.rerun()

    with col_del:
        with st.popover("🗑️ Delete a Tender"):
            options = {
                f"[{r['id']}] {r['project_name']} — {r['client_name']}": r["id"]
                for _, r in original.iterrows()
            }
            sel = st.selectbox("Select tender to delete", list(options.keys()))
            if st.button("⚠️ Confirm Delete", type="primary"):
                delete_tender(options[sel])
                st.success("Tender deleted.")
                st.rerun()


def _render_workload(df: pd.DataFrame) -> None:
    active = df[~df["status"].isin(["Won", "Lost", "Untracked"])]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Active Workload by Assignee**")
        if not active.empty:
            st.bar_chart(active["assignee"].value_counts(), color="#36a2eb")
        else:
            st.info("No active tenders.")

        st.markdown("**⚠️ Low-Probability Tenders (< 40%)**  \n*Consider withdrawing or deprioritising to save effort.*")
        if not active.empty:
            lp = active[active["win_prob"] < 40][["project_name", "client_name", "win_prob", "assignee"]].sort_values("win_prob")
            if not lp.empty:
                st.dataframe(lp, hide_index=True, use_container_width=True)
            else:
                st.success("No low-probability active tenders. 🎉")

    with col_r:
        st.markdown("**🔥 High-Potential Tenders (≥ 70%)**  \n*Focus your team's effort here.*")
        if not active.empty:
            hp = active[active["win_prob"] >= 70][["project_name", "client_name", "win_prob", "assignee"]].sort_values("win_prob", ascending=False)
            if not hp.empty:
                st.dataframe(hp, hide_index=True, use_container_width=True)
            else:
                st.info("No high-potential active tenders at the moment.")
        else:
            st.info("No active tenders.")
