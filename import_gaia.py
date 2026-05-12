# import_gaia.py
# ─────────────────────────────────────────────────────────────────────────────
# One-shot import script for Gaia's real tender data.
#
# Reads both CSVs:
#   • 2024 - Tender Sebutharga Summary(2024).csv
#   • 2025 - Tender Sebutharga Summary(2025).csv
#
# Maps CSV columns → tenders schema:
#   Bidding_Title   → project_name
#   Institution     → client_name
#   Amount_Value    → value  (parsed from "RM218,530.00" → 218530.0)
#   Starting_Date   → starting_date
#   SalesPerson     → assignee
#   Due_Date        → deadline
#   Submission_Method → submission_method
#   Product_Brand   → product_brand
#   Product_Model   → product_model
#   Success         → status  (Yes→Won, No→Lost)
#                     Rows with no outcome (blank Success) are SKIPPED.
#
# Usage:
#   python import_gaia.py
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ── Bootstrap path so we can import project modules ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import ATTRIBUTES, PIPELINE_STAGES
from database import add_staff, add_tender, get_all_staff, init_db, load_tenders
from ai_engine import pick_best_attribute

# ── Files to import (both years) ──────────────────────────────────────────────
CSV_FILES = [
    Path("2024 - Tender Sebutharga Summary(2024).csv"),
    Path("2025 - Tender Sebutharga Summary(2025).csv"),
]

# ── Column header row is row index 1 (0-indexed) in the raw CSV ───────────────
# Row 0 = "2025 - Tender Sebutharga Summary" (title)
# Row 1 = actual column headers
STANDARD_COLS = [
    "No", "Starting_Date", "Due_Date", "Institution", "Reference_No",
    "Bidding_Title", "SalesPerson", "Product_Brand", "Product_Model",
    "Amount_Value", "Submitted_Date", "Submission_Method",
    "Success", "Winning_Company", "Remark",
]

# ── Status mapping ─────────────────────────────────────────────────────────────
def _map_status(val) -> str:
    """
    Map raw Success column to a pipeline stage.
    Blank / unknown outcome → 'Untracked'.
    """
    if pd.isna(val):
        return "Untracked"
    s = str(val).strip().lower()
    if s == "yes":
        return "Won"
    if s == "no":
        return "Lost"
    return "Untracked"


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
    """Parse a date value, returning None if blank or unparseable.
    Handles formats like '23 December, 2024' and standard d/m/yyyy.
    Rejects any result before year 2000 as a parsing artefact.
    """
    if pd.isna(val):
        return None
    s = str(val).strip().rstrip(",")   # remove trailing comma if any
    # Remove mid-string commas so "23 December, 2024" → "23 December 2024"
    s = s.replace(",", "")
    try:
        parsed = pd.to_datetime(s, dayfirst=True)
        if parsed.year < 2000:          # implausible — treat as no date
            return None
        return parsed.date()
    except Exception:
        return None


# ── Keyword → ATTRIBUTE mapper ────────────────────────────────────────────────
# The CSV has no "primary_factor" column — we derive a starting guess from
# the product model / title keywords, then let the AI pick the best attribute.
_KEYWORD_MAP: dict[str, str] = {
    # Price-sensitive commodity items
    "reagent": "Price",
    "consumable": "Price",
    "chemical": "Price",
    "kit": "Price",
    # Technical / specialised equipment
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
    # Delivery / logistics
    "deliver": "Delivery & Timeline",
    "courier": "Delivery & Timeline",
    "install": "Delivery & Timeline",
    "commission": "Delivery & Timeline",
    # Relationship / brand
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
    return ATTRIBUTES[0]   # fallback: first attribute


# ── Main import routine ────────────────────────────────────────────────────────

def import_csv(csv_path: Path, df_existing: pd.DataFrame, staff_set: set[str]) -> tuple[int, int, list[str]]:
    """
    Import one CSV file into the database.

    Returns (rows_imported, rows_skipped, error_messages).
    """
    print(f"\n{'='*60}")
    print(f"  Importing: {csv_path.name}")
    print(f"{'='*60}")

    raw = pd.read_csv(csv_path, encoding="cp1252", on_bad_lines="skip", header=1)

    # Rename to our standard columns (only as many as we have)
    rename = {old: new for old, new in zip(raw.columns, STANDARD_COLS)}
    raw = raw.rename(columns=rename)

    # Drop any rows where Bidding_Title AND Institution are both blank
    raw = raw.dropna(subset=["Bidding_Title", "Institution"], how="all")

    count    = 0
    skipped  = 0
    errors: list[str] = []

    # Keep a snapshot so the AI can use accumulated records within this import
    df_history = df_existing.copy()

    for i, row in raw.iterrows():
        project_name = str(row.get("Bidding_Title", "")).strip()
        client_name  = str(row.get("Institution",   "")).strip()

        if not project_name or project_name == "nan":
            skipped += 1
            continue
        if not client_name or client_name == "nan":
            skipped += 1
            continue

        # ── Map outcome status ───────────────────────────────────────────────────────────
        status = _map_status(row.get("Success"))
        amount  = _parse_amount(row.get("Amount_Value"))
        if amount <= 0:
            errors.append(f"Row {i+2}: '{project_name[:40]}' — zero/unparseable amount, skipped.")
            skipped += 1
            continue
        deadline      = _parse_date(row.get("Due_Date")) or date.today()
        starting_date = _parse_date(row.get("Starting_Date"))   # None if blank/bad
        assignee     = str(row.get("SalesPerson", "Unassigned")).strip().title()
        if not assignee or assignee in ("Nan", ""):
            assignee = "Unassigned"
        product_brand  = str(row.get("Product_Brand", "") or "").strip()
        product_brand  = "" if product_brand == "nan" else product_brand
        product_model  = str(row.get("Product_Model", "") or "").strip()
        product_model  = "" if product_model == "nan" else product_model
        sub_method     = str(row.get("Submission_Method", "") or "").strip()
        sub_method     = "" if sub_method == "nan" else sub_method

        # Auto-register new staff
        if assignee not in staff_set:
            added = add_staff(assignee, "Sales")
            if added:
                staff_set.add(assignee)
                print(f"  [+] New staff registered: {assignee}")

        # Derive a starting factor guess, then let AI optimise
        product_model = str(row.get("Product_Model", "")).strip()
        initial_factor = _guess_factor(product_model, project_name)

        try:
            best_fac, best_res = pick_best_attribute(
                amount,              # value
                client_name, assignee,
                df_history, ATTRIBUTES,
            )
        except Exception as e:
            best_fac = initial_factor
            best_res = None
            errors.append(f"Row {i+2}: AI scoring failed ({e}), using keyword guess.")

        win_prob = best_res.probability if best_res else 50

        try:
            add_tender(
                project      = project_name,
                client_name  = client_name,
                value        = amount,
                win_prob     = win_prob,
                status       = status,
                factor       = best_fac,
                assignee     = assignee,
                deadline     = deadline,
                starting_date= starting_date,
                submission_method = sub_method,
                product_brand= product_brand,
                product_model= product_model,
            )
            count += 1
            print(f"  [OK][{count:>3}] {project_name[:55]:<55} | {status:<9} | {assignee:<12} | RM{amount:>12,.0f}")

            # Add this record to local history so next rows can learn from it
            new_row = {
                "project_name": project_name, "client_name": client_name,
                "value": amount, "bid_amount": amount, "win_prob": win_prob,
                "status": status, "primary_factor": best_fac,
                "assignee": assignee, "deadline": str(deadline),
            }
            df_history = pd.concat(
                [df_history, pd.DataFrame([new_row])], ignore_index=True
            )

        except Exception as e:
            errors.append(f"Row {i+2}: DB insert failed — {e}")
            skipped += 1

    return count, skipped, errors


def _wipe_tenders() -> None:
    """Delete ALL rows from the tenders table so a re-import starts clean."""
    import sqlite3
    from config import DB_FILE
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM tenders")
    conn.execute("DELETE FROM staff")
    conn.commit()
    conn.close()


def main() -> None:
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("\n[*] Initialising database ...")
    init_db()

    print("[*] Wiping existing tenders & staff for a clean re-import ...")
    _wipe_tenders()

    existing_count_before = 0
    print(f"[*] DB cleared. Starting fresh import (Won/Lost only) ...")

    df_existing = load_tenders()
    staff_set   = set(get_all_staff())

    total_imported = 0
    total_skipped  = 0
    all_errors: list[str] = []

    for csv_file in CSV_FILES:
        if not csv_file.exists():
            print(f"\n[!] File not found, skipping: {csv_file}")
            continue

        n_ok, n_skip, errs = import_csv(csv_file, df_existing, staff_set)
        total_imported += n_ok
        total_skipped  += n_skip
        all_errors.extend(errs)

        # Reload so next CSV can benefit from what this one added
        df_existing = load_tenders()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  IMPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  [OK]  Rows imported : {total_imported}")
    print(f"  [--]  Rows skipped  : {total_skipped}")
    print(f"  [!!]  Errors        : {len(all_errors)}")
    existing_count_after = len(load_tenders())
    print(f"  [DB]  Total DB rows : {existing_count_after}  (was {existing_count_before})")

    if all_errors:
        print(f"\n  -- Error log --")
        for e in all_errors:
            print(f"    {e}")

    print(f"\n  Run 'streamlit run app.py' to view the system.\n")


if __name__ == "__main__":
    main()
