# ui_tab_new_entry.py
# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — New Entry: manual form + live AI breakdown + CSV/Excel bulk upload.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import streamlit as st

from ai_engine import predict, pick_best_attribute
from config import ATTRIBUTES, CSV_TEMPLATE_COLUMNS, PIPELINE_STAGES
from database import add_tender, recalculate_all_probabilities
from ui_score_breakdown import render_breakdown


def render(df_master: pd.DataFrame, staff_list: list[str]) -> None:
    st.subheader("Add New Tender")
    tab_manual, tab_upload = st.tabs(["Manual Entry", "Bulk Upload (CSV / Excel)"])

    with tab_manual:
        _render_manual(df_master, staff_list)

    with tab_upload:
        _render_upload(df_master, staff_list)


# ── Manual entry ──────────────────────────────────────────────────────────────

def _render_manual(df_master: pd.DataFrame, staff_list: list[str]) -> None:
    if not staff_list:
        st.warning("⚠️ No staff members found. Add at least one staff member in the sidebar before creating a tender.")
        return

    with st.container(border=True):
        col_form, col_ai = st.columns([3, 2], gap="large")

        with col_form:
            st.markdown("#### Tender Details")
            c1, c2 = st.columns(2)

            s_proj   = c1.text_input("Project Name *")
            s_cl     = c1.text_input("Client Name *")
            s_start  = c1.date_input("Start Date", value=date.today())
            s_dat    = c1.date_input("Deadline *", value=date.today())
            s_stat   = c1.selectbox("Initial Status", PIPELINE_STAGES, index=0)
            s_method = c1.selectbox("Submission Method", ["Online Bidding", "Email", "Hardcopy by Hand", "Hardcopy by Courier"])

            s_stf  = c2.selectbox("Assigned Lead *", staff_list)
            s_val  = c2.number_input("Estimated Budget (RM) *", min_value=0, step=1000)
            s_brand = c2.text_input("Product Brand")
            s_model = c2.text_input("Product Model")

            # ── Key Driver: Auto (AI) or Manual (user) ────────────────────────────
            driver_mode = c2.radio(
                "Key Driver Selection",
                ["🤖 Auto (AI)", "✏️ Manual"],
                horizontal=True,
                help="Auto lets the AI pick the best attribute; Manual lets you override.",
            )

            s_fac    = ATTRIBUTES[0]
            best_res = None

            if driver_mode == "✏️ Manual":
                s_fac = c2.selectbox("Key Driver *", ATTRIBUTES)
                if s_cl and s_val > 0:
                    best_res = predict(s_val, s_cl, s_fac, s_stf, df_master, deadline=s_dat)
                    c2.caption(f"AI score for this driver: **{best_res.probability}%** · Confidence: {best_res.confidence_level}")
                else:
                    c2.caption("Fill in Client & Budget to preview AI score.")
            else:
                if s_cl and s_val > 0:
                    s_fac, best_res = pick_best_attribute(
                        s_val, s_cl, s_stf, df_master, ATTRIBUTES, deadline=s_dat
                    )
                    c2.info(f"🤖 AI selected Key Driver: **{s_fac}** · Confidence: {best_res.confidence_level}")
                else:
                    c2.info("🤖 AI will choose Key Driver once fields are filled.")

            submitted = st.button("➕ Add Tender", type="primary", use_container_width=True)
            if submitted:
                if not (s_proj and s_cl and s_val > 0):
                    st.error("Please fill in all required fields (marked with *).")
                else:
                    if not best_res:
                        best_res = predict(s_val, s_cl, s_fac, s_stf, df_master, deadline=s_dat)
                    add_tender(s_proj, s_cl, s_val, best_res.probability,
                               s_stat, s_fac, s_stf, s_dat, s_start,
                               s_method, s_brand, s_model)
                    if best_res.probability >= 70:
                        st.success(f"✅ Tender added! 🔥 High potential (**{best_res.probability}%**). Focus your team's effort here.")
                    elif best_res.probability < 40:
                        st.warning(f"✅ Tender added, but ⚠️ low probability (**{best_res.probability}%**). Consider deprioritising.")
                    else:
                        st.success(f"✅ Tender added with a predicted win probability of **{best_res.probability}%**.")
                    time.sleep(1.5)
                    st.rerun()

        # ── Live AI preview panel ─────────────────────────────────────────────
        with col_ai:
            st.markdown("#### 🤖 Live AI Prediction")
            if s_proj and s_cl and s_val > 0 and best_res:
                render_breakdown(best_res, compact=True)
            else:
                st.info("Fill in the form to see the AI score breakdown.")





# ── Bulk upload ───────────────────────────────────────────────────────────────

_COLUMN_ALIASES: dict[str, list[str]] = {
    "project_name":   ["project_name",   "Project Name",  "Project",
                       "Bidding_Title",   "Bidding Title"],
    "client_name":    ["client_name",    "Client Name",   "Client",
                       "Institution",    "Institutions/University"],
    "value":          ["value",          "Value",         "Budget", "Est. Value",
                       "Amount_Value",   "Amount Value"],
    "bid_amount":     ["bid_amount",     "Bid Amount",    "My Bid Amount", "Bid",
                       "Amount_Value",   "Amount Value"],
    "primary_factor": ["primary_factor", "Primary Factor","Key Driver",
                       "Key Driver / Strategy", "Product_Model", "Product Model"],
    "assignee":       ["assignee",       "Assignee",      "Lead", "Staff",
                       "SalesPerson",    "Sales Person"],
    "deadline":       ["deadline",       "Deadline",      "Date",
                       "Due_Date",       "Due Date"],
    "status":         ["status",         "Status",        "Success"],
}

# Gaia Sebutharga CSV: row 0 is a title, row 1 is the real header.
# These are the positional column names after reading with header=1.
_GAIA_COL_MAP = {
    "No":                        None,
    "Starting Date":             "starting_date",
    "Due Date":                  "deadline",
    "Institutions/University":   "client_name",
    "Reference No":              None,
    "Bidding Title":             "project_name",
    "SalesPerson":               "assignee",
    "Product Brand":             "product_brand",
    "Product Model":             "product_model",
    "Amount Value":              "value",
    "Submitted Date":            None,
    "Submission Method":         "submission_method",
    "Success":                   "status",
    "Winning Company":           None,
    "Remark":                    None,
}


def _is_gaia_format(df: pd.DataFrame) -> bool:
    """Return True when the first column looks like the Sebutharga title row."""
    first_col = str(df.columns[0])
    return "Tender Sebutharga" in first_col or "Sebutharga" in first_col


def _parse_gaia_csv(raw_bytes: bytes) -> pd.DataFrame:
    """
    Re-read the CSV with header on row 1, map Gaia columns to system schema,
    parse Amount Value, filter to Won/Lost only.
    """
    import io, re
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc,
                             header=1, on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue

    # Rename to system schema using _GAIA_COL_MAP
    rename = {old: new for old, new in _GAIA_COL_MAP.items()
              if old in df.columns and new is not None}
    df = df.rename(columns=rename)

    # Parse Amount Value → numeric float
    def _parse_rm(val):
        if pd.isna(val):
            return 0.0
        return float(re.sub(r"[^\d.]", "", str(val)) or 0)

    if "value" in df.columns:
        df["value"] = df["value"].apply(_parse_rm)

    # Map Success column → pipeline status; Untracked for blank outcomes
    if "status" in df.columns:
        def _map(v):
            if pd.isna(v):
                return "Untracked"
            s = str(v).strip().lower()
            return "Won" if s == "yes" else ("Lost" if s == "no" else "Untracked")
        df["status"] = df["status"].apply(_map)

    return df



def _render_upload(df_master: pd.DataFrame, staff_list: list[str]) -> None:
    st.markdown(
        "Upload a CSV or Excel file that matches the template. "
        "Any column not provided will use a safe default."
    )

    uploaded = st.file_uploader("Choose file", type=["csv", "xlsx", "xls"])
    if not uploaded:
        return

    st.markdown("---")
    if st.button("⬆️ Import Tenders", type="primary"):
        try:
            if uploaded.name.endswith(".csv"):
                # Try UTF-8 first; fall back to cp1252 for Windows-generated files
                raw_bytes = uploaded.read()
                for enc in ("utf-8", "cp1252", "latin-1"):
                    try:
                        import io
                        raw = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, on_bad_lines="skip")
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                else:
                    st.error("Could not decode the CSV file. Please save it as UTF-8 and try again.")
                    return

                # ── Detect Gaia Sebutharga format and re-parse properly ────────
                if _is_gaia_format(raw):
                    data = _parse_gaia_csv(raw_bytes)
                    st.info(
                        f"📋 Gaia Sebutharga format detected — "
                        f"importing **{len(data)}** closed (Won/Lost) tenders."
                    )
                else:
                    data = _normalise_columns(raw)
            else:
                raw  = pd.read_excel(uploaded)
                data = _normalise_columns(raw)

            if "project_name" not in data.columns or "client_name" not in data.columns:
                st.error(
                    f"Missing required columns. "
                    f"Found in file: {list(data.columns)}"
                )
                return

            errors: list[str] = []
            count = 0

            for i, row in data.iterrows():
                try:
                    p_name   = str(row["project_name"]).strip()
                    c_name   = str(row["client_name"]).strip()
                    val      = float(row.get("value", 0))
                    raw_fac  = row.get("primary_factor", ATTRIBUTES[0])
                    fac      = raw_fac if raw_fac in ATTRIBUTES else ATTRIBUTES[0]
                    raw_stat = row.get("status", "Qualified Lead")
                    status   = raw_stat if raw_stat in PIPELINE_STAGES else "Qualified Lead"
                    assignee = str(row.get("assignee", staff_list[0] if staff_list else "Unassigned")).strip()
                    s_date   = row.get("starting_date", None)
                    s_method = str(row.get("submission_method", "") or "")
                    p_brand  = str(row.get("product_brand", "") or "")
                    p_model  = str(row.get("product_model", "") or "")
                    if assignee and assignee not in staff_list:
                        from database import add_staff
                        add_staff(assignee, "Imported")
                        staff_list.append(assignee)

                    try:
                        deadline = pd.to_datetime(row.get("deadline", date.today())).date()
                    except Exception:
                        deadline = date.today()

                    try:
                        starting_date = pd.to_datetime(s_date).date() if s_date and str(s_date) not in ("nan", "None", "") else None
                    except Exception:
                        starting_date = None

                    if not p_name or not c_name or val <= 0:
                        errors.append(f"Row {i + 2}: skipped — missing name or zero value.")
                        continue

                    best_fac, best_res = pick_best_attribute(
                        val, c_name, assignee, df_master, ATTRIBUTES, deadline=deadline
                    )
                    fac = best_fac
                    result = best_res

                    add_tender(p_name, c_name, val, result.probability,
                               status, fac, assignee, deadline,
                               starting_date, s_method, p_brand, p_model)
                    count += 1

                except Exception as row_err:
                    errors.append(f"Row {i + 2}: {row_err}")

            if count:
                with st.spinner("Recalculating AI scores against full dataset..."):
                    recalculate_all_probabilities(predict)
                st.success(f"✅ Successfully imported **{count}** tender(s) and recalculated all AI scores.")
            if errors:
                with st.expander(f"⚠️ {len(errors)} row(s) had issues"):
                    for e in errors:
                        st.write(e)

            time.sleep(1)
            st.rerun()

        except Exception as exc:
            st.error(f"Import failed: {exc}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map: dict[str, str] = {}
    for target, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[alias] = target
                break
    return df.rename(columns=col_map)
