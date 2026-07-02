# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Tender Prediction and Groundwork Tracking System — entry point.
#
# Run with:  streamlit run app.py
#
# Module map
# ──────────
# config.py                → Constants (weights, thresholds, stage lists)
# database.py              → SQLite CRUD; starts empty (no dummy data)
# ai_engine.py             → 5-rule scoring engine + analytics helpers
# charts.py                → Matplotlib charts + ReportLab PDF
# ui_onboarding.py         → Empty-state screen with CSV template download
# ui_score_breakdown.py    → Reusable score breakdown component
# ui_tab_new_entry.py      → Tab 1: manual form + CSV/Excel bulk upload
# ui_tab_pipeline.py       → Tab 2: editable grid, save/delete, workload
# ui_tab_report.py         → Tab 3: full AI breakdown card + PDF download
# ui_tab_performance.py    → Tab 4: win/loss analytics + relationship table
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from database import add_staff, db_is_empty, get_all_staff, init_db, load_tenders, delete_staff
from ui_onboarding import render as render_onboarding
import ui_tab_new_entry   as tab_new
import ui_tab_pipeline    as tab_pipeline
import ui_tab_report      as tab_report
import ui_tab_performance as tab_perf
import ui_tab_gaia_analysis as tab_gaia


# ── Bootstrap ─────────────────────────────────────────────────────────────────

init_db()

st.set_page_config(
    page_title="Tender Prediction and Groundwork Tracking System",
    layout="wide",
    page_icon="📋",
)

# ── Sidebar: Staff management ─────────────────────────────────────────────────

st.sidebar.title("👥 Sales Support Team")

staff_list = get_all_staff()

if staff_list:
    st.sidebar.markdown("**Current Staff**")
    for name in staff_list:
        col1, col2 = st.sidebar.columns([5, 1])
        col1.caption(f"• {name}")
        if col2.button("🚫", key=f"del_staff_{name}", help=f"Remove {name}"):
            delete_staff(name)
            st.rerun()
    st.sidebar.divider()

with st.sidebar.form("add_staff_form", clear_on_submit=True):
    st.markdown("**Add Staff Member**")
    new_name = st.text_input("Full Name")
    new_dept = st.selectbox("Department", ["Material Science", "Life Science", "Service", "Analytical Science"])
    if st.form_submit_button("➕ Add", use_container_width=True):
        if new_name.strip():
            if add_staff(new_name.strip(), new_dept):
                st.rerun()
            else:
                st.error(f"'{new_name}' already exists.")
        else:
            st.error("Name cannot be blank.")

# ── Onboarding gate ───────────────────────────────────────────────────────────
# If the DB is completely empty, show the onboarding screen instead of tabs.

if db_is_empty():
    render_onboarding()
    st.stop()

# ── Load shared state ─────────────────────────────────────────────────────────

staff_list = get_all_staff()
df_master  = load_tenders()

# ── Main tabs ─────────────────────────────────────────────────────────────────

st.title("📋 Tender Prediction and Groundwork Tracking System")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 New Entry",
    "📋 Pipeline",
    "📄 Report",
    "🧠 Performance",
    "🏢 Gaia Analysis",
])

with tab1:
    tab_new.render(df_master, staff_list)

with tab2:
    tab_pipeline.render(df_master)

with tab3:
    tab_report.render(df_master)

with tab4:
    tab_perf.render(df_master)

with tab5:
    tab_gaia.render(df_master)
