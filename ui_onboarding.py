# ui_onboarding.py
# ─────────────────────────────────────────────────────────────────────────────
# Shown when the database has no data yet.
# Provides a CSV template download and a quick-start guide.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from config import ATTRIBUTES, CSV_TEMPLATE_COLUMNS, PIPELINE_STAGES


_EXAMPLE_ROWS = [
    {
        "project_name":   "Chemical Reagents Q1",
        "client_name":    "Ministry of Health",
        "value":          150000,
        "bid_amount":     142000,
        "primary_factor": "Price",
        "assignee":       "Ali",
        "deadline":       "2025-06-30",
        "status":         "Drafting Proposal",
    },
    {
        "project_name":   "Lab Equipment Maintenance",
        "client_name":    "University Hospital",
        "value":          80000,
        "bid_amount":     76000,
        "primary_factor": "Technical Capability",
        "assignee":       "Sarah",
        "deadline":       "2025-08-15",
        "status":         "Qualified Lead",
    },
]


def _build_template_csv() -> bytes:
    """Return a CSV bytes object: 2 example rows so users see the format."""
    df = pd.DataFrame(_EXAMPLE_ROWS, columns=CSV_TEMPLATE_COLUMNS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def render() -> None:
    """Render the full-page onboarding screen."""
    st.title("📋 Smart Tender System")
    st.markdown("---")

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.subheader("👋 Welcome — No data found yet")
        st.markdown(
            """
            The system starts with a **clean slate** — no dummy data.  
            You have two ways to get started:

            **Option A — Upload your existing tenders via CSV**  
            Download the template below, fill it in, then upload it on the  
            *📝 New Entry → Upload* tab.

            **Option B — Add tenders one by one**  
            Use the *📝 New Entry → Manual Entry* form to enter deals directly.

            Once you have at least a few **Won** and **Lost** tenders in the system,  
            the AI engine will begin producing meaningful predictions.
            """
        )

        st.download_button(
            label="⬇️ Download CSV Template",
            data=_build_template_csv(),
            file_name="tender_template.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
        st.markdown("---")
        st.write("**Ready to upload?**")
        from database import load_tenders, get_all_staff
        from ui_tab_new_entry import _render_upload
        _render_upload(pd.DataFrame(), get_all_staff())
    with col_right:
        st.subheader("📖 CSV Column Guide")

        guide = pd.DataFrame(
            [
                ["project_name",   "Text",          "Name of the project / tender"],
                ["client_name",    "Text",          "Name of the client organisation"],
                ["value",          "Number (RM)",   "Estimated project budget"],
                ["bid_amount",     "Number (RM)",   "Your proposed bid"],
                ["primary_factor", "See below",     "Chosen competitive strategy"],
                ["assignee",       "Text",          "Lead staff member's name"],
                ["deadline",       "YYYY-MM-DD",    "Submission deadline"],
                ["status",         "See below",     "Current pipeline stage"],
            ],
            columns=["Column", "Format", "Description"],
        )
        st.dataframe(guide, hide_index=True, use_container_width=True)

        st.markdown("**Allowed values for `primary_factor`:**")
        st.code("\n".join(ATTRIBUTES))

        st.markdown("**Allowed values for `status`:**")
        st.code("\n".join(PIPELINE_STAGES))

    st.markdown("---")
    st.caption(
        "💡 Tip: The AI prediction engine learns from your **Won** and **Lost** records. "
        "Import historical data first for the most accurate scores."
    )
