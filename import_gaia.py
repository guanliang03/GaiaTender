# import_gaia.py
# ─────────────────────────────────────────────────────────────────────────────
# One-shot import script for tender data (supports .xlsx and .csv).
#
# Reads sample file:
#   • Names of individuals or organizations prohibited....xlsx
#   • 2024 - Tender Sebutharga Summary(2024).csv (if present)
#   • 2025 - Tender Sebutharga Summary(2025).csv (if present)
#
# Maps columns → tenders schema:
#   Bidding Title / Bidding_Title   → project_name
#   Institutions / Institution      → client_name
#   Amount Value / Amount_Value     → value
#   Starting Date / Starting_Date   → starting_date
#   SalesPerson                     → assignee
#   Due Date / Due_Date             → deadline
#   Submission / Submission_Method  → submission_method
#   Product Brand / Product_Brand   → product_brand
#   Product Model / Product_Model   → product_model
#   Success                         → status  (Yes→Won, No→Lost, blank→Submitted)
#
# Usage:
#   python import_gaia.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ── Bootstrap path so we can import project modules ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import ATTRIBUTES, DB_FILE
from database import add_staff, add_tender, get_all_staff, init_db, load_tenders, recalculate_all_probabilities
from ai_engine import pick_best_attribute, predict

# ── Files to import ────────────────────────────────────────────────────────────
DATA_FILES = [
    Path("2024 - Tender Sebutharga Summary(2024).csv"),
    Path("2025 - Tender Sebutharga Summary(2025).csv"),
]

# ── Column mapping dictionary ─────────────────────────────────────────────────
COLUMN_ALIAS_MAP = {
    "bidding_title": "project_name",
    "bidding title": "project_name",
    "bidding title ": "project_name",
    "institution": "client_name",
    "institutions": "client_name",
    "institutions/university": "client_name",
    "salesperson": "assignee",
    "sales person": "assignee",
    "product_brand": "product_brand",
    "product brand": "product_brand",
    "product_model": "product_model",
    "product model": "product_model",
    "amount_value": "value",
    "amount value": "value",
    "starting_date": "starting_date",
    "starting date": "starting_date",
    "due_date": "deadline",
    "due date": "deadline",
    "submission_method": "submission_method",
    "submission method": "submission_method",
    "submission method ": "submission_method",
    "submission": "submission_method",
    "success": "status",
}


# ── Status mapping ─────────────────────────────────────────────────────────────
def _map_status(val) -> str:
    """
    Map raw Success column to a pipeline stage.
    Blank outcome → 'Submitted'.
    """
    if pd.isna(val):
        return "Submitted"
    s = str(val).strip().lower()
    if s == "yes":
        return "Won"
    if s == "no":
        return "Lost"
    return "Submitted"


# ── Amount parser: "RM218,530.00" → 218530.0 ──────────────────────────────────
def _parse_amount(val) -> float:
    if pd.isna(val):
        return 0.0
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# ── Date parser ───────────────────────────────────────────────────────────────
def _parse_date(val) -> date | None:
    """Parse a date value, returning None if blank or unparseable."""
    if pd.isna(val):
        return None
    s = str(val).strip().rstrip(",")
    s = s.replace(",", "")
    try:
        parsed = pd.to_datetime(s, dayfirst=True)
        if parsed.year < 2000:
            return None
        return parsed.date()
    except Exception:
        return None


# ── Keyword → ATTRIBUTE mapper ────────────────────────────────────────────────
_KEYWORD_MAP: dict[str, str] = {
    "reagent": "Price",
    "consumable": "Price",
    "chemical": "Price",
    "kit": "Price",
    "freeze": "Technical Capability",
    "spectro": "Technical Capability",
    "microscop": "Technical Capability",
    "sequenc": "Technical Capability",
    "chromatograph": "Technical Capability",
    "centrifug": "Technical Capability",
    "pcr": "Technical Capability",
    "analyser": "Technical Capability",
    "analyzer": "Technical Capability",
    "flow cytom": "Technical Capability",
    "nmr": "Technical Capability",
    "mass spec": "Technical Capability",
    "deliver": "Delivery & Timeline",
    "courier": "Delivery & Timeline",
    "install": "Delivery & Timeline",
    "commission": "Delivery & Timeline",
    "service": "Relationship & Reputation",
    "maintenance": "Relationship & Reputation",
    "warranty": "Relationship & Reputation",
    "support": "Relationship & Reputation",
}

def _guess_factor(product_model: str, bidding_title: str) -> str:
    combined = f"{product_model} {bidding_title}".lower()
    for kw, attr in _KEYWORD_MAP.items():
        if kw in combined:
            return attr
    return ATTRIBUTES[0]


# ── Import routine ────────────────────────────────────────────────────────────

def import_file(file_path: Path, df_existing: pd.DataFrame, staff_set: set[str]) -> tuple[int, int, list[str]]:
    """
    Import one CSV or Excel file into the database.
    Returns (rows_imported, rows_skipped, error_messages).
    """
    print(f"\n{'='*60}")
    print(f"  Importing: {file_path.name}")
    print(f"{'='*60}")

    if file_path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(file_path)
    else:
        # Check if CSV has a title row (e.g. first row doesn't contain the 'No' column header)
        first_row = pd.read_csv(file_path, nrows=1, encoding="cp1252", on_bad_lines="skip")
        has_no = any(c.strip().lower() == "no" for c in first_row.columns)
        header_row = 0 if has_no else 1
        raw = pd.read_csv(file_path, encoding="cp1252", on_bad_lines="skip", header=header_row)

    # Normalize column headers
    norm_cols = {}
    for col in raw.columns:
        c_clean = str(col).strip().lower()
        if c_clean in COLUMN_ALIAS_MAP:
            norm_cols[col] = COLUMN_ALIAS_MAP[c_clean]
    raw = raw.rename(columns=norm_cols)

    count = 0
    skipped = 0
    errors: list[str] = []
    df_history = df_existing.copy()

    for i, row in raw.iterrows():
        project_name = str(row.get("project_name", "")).strip()
        client_name = str(row.get("client_name", "")).strip()

        if not project_name or project_name.lower() in ("nan", "none", ""):
            skipped += 1
            continue
        if not client_name or client_name.lower() in ("nan", "none", ""):
            skipped += 1
            continue

        status = _map_status(row.get("status"))
        amount = _parse_amount(row.get("value"))
        if amount <= 0:
            errors.append(f"Row {i+2}: '{project_name[:40]}' — zero or unparseable amount, skipped.")
            skipped += 1
            continue

        deadline = _parse_date(row.get("deadline")) or date.today()
        starting_date = _parse_date(row.get("starting_date"))
        assignee = str(row.get("assignee", "Unassigned")).strip().title()
        if not assignee or assignee.lower() in ("nan", "none", ""):
            assignee = "Unassigned"

        product_brand = str(row.get("product_brand", "") or "").strip()
        product_brand = "" if product_brand.lower() == "nan" else product_brand
        product_model = str(row.get("product_model", "") or "").strip()
        product_model = "" if product_model.lower() == "nan" else product_model
        sub_method = str(row.get("submission_method", "") or "").strip()
        sub_method = "" if sub_method.lower() == "nan" else sub_method

        # Auto-register staff member
        if assignee not in staff_set:
            added = add_staff(assignee, "Sales")
            if added:
                staff_set.add(assignee)
                print(f"  [+] New staff registered: {assignee}")

        initial_factor = _guess_factor(product_model, project_name)

        try:
            best_fac, best_res = pick_best_attribute(
                amount, client_name, assignee, df_history, ATTRIBUTES,
                deadline=deadline,
                product_brand=product_brand,
                product_model=product_model,
                project_name=project_name,
            )
        except Exception as e:
            best_fac = initial_factor
            best_res = None
            errors.append(f"Row {i+2}: AI scoring fallback ({e}).")

        win_prob = best_res.probability if best_res else 50.0

        try:
            inserted = add_tender(
                project=project_name,
                client_name=client_name,
                value=amount,
                win_prob=win_prob,
                status=status,
                factor=best_fac,
                assignee=assignee,
                deadline=deadline,
                starting_date=starting_date,
                submission_method=sub_method,
                product_brand=product_brand,
                product_model=product_model,
            )
            if inserted:
                count += 1
                print(f"  [OK][{count:>2}] {project_name[:40]:<40} | {client_name:<10} | {status:<9} | {assignee:<8} | RM{amount:>10,.0f}")
                new_row = {
                    "project_name": project_name, "client_name": client_name,
                    "value": amount, "win_prob": win_prob, "status": status,
                    "primary_factor": best_fac, "assignee": assignee, "deadline": str(deadline),
                }
                df_history = pd.concat([df_history, pd.DataFrame([new_row])], ignore_index=True)
            else:
                skipped += 1
                errors.append(f"Row {i+2}: Skipped — duplicate entry.")
        except Exception as e:
            errors.append(f"Row {i+2}: DB insert failed — {e}")
            skipped += 1

    return count, skipped, errors


def _wipe_tenders() -> None:
    """Delete ALL documents from the tenders and staff collections in Firestore."""
    from database import db

    while True:
        docs = list(db.collection("tenders").limit(100).stream())
        if not docs:
            break
        for doc in docs:
            doc.reference.delete()

    while True:
        docs = list(db.collection("staff").limit(100).stream())
        if not docs:
            break
        for doc in docs:
            doc.reference.delete()

    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except Exception:
            pass


def main() -> None:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("\n[*] Initialising database ...")
    init_db()

    print("[*] Wiping existing tenders & staff for a clean re-import ...")
    _wipe_tenders()

    df_existing = load_tenders()
    staff_set = set(get_all_staff())

    total_imported = 0
    total_skipped = 0
    all_errors: list[str] = []

    # Import primary target file first
    for file_path in DATA_FILES:
        if not file_path.exists():
            continue

        n_ok, n_skip, errs = import_file(file_path, df_existing, staff_set)
        total_imported += n_ok
        total_skipped += n_skip
        all_errors.extend(errs)

        df_existing = load_tenders()

    # Recalculate AI probabilities across full fresh dataset
    if total_imported > 0:
        print("\n[*] Recalculating AI win probabilities for fresh dataset ...")
        recalculate_all_probabilities(predict)

    print(f"\n{'='*60}")
    print(f"  IMPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  [OK]  Rows imported : {total_imported}")
    print(f"  [--]  Rows skipped  : {total_skipped}")
    print(f"  [!!]  Errors        : {len(all_errors)}")
    existing_count_after = len(load_tenders())
    print(f"  [DB]  Total DB rows : {existing_count_after}")

    if all_errors:
        print(f"\n  -- Error log --")
        for e in all_errors:
            print(f"    {e}")

    print(f"\n  Run 'streamlit run app.py' to view the system.\n")


if __name__ == "__main__":
    main()
