# database.py
# ─────────────────────────────────────────────────────────────────────────────
# Firebase Firestore schema and CRUD helpers.
# No seed / dummy data — the DB starts empty.
# Zero Streamlit dependency (except dynamic initialisation support).
# ─────────────────────────────────────────────────────────────────────────────

import os
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

from config import FIREBASE_SERVICE_ACCOUNT_KEY

# ── Safe Firebase Admin SDK Initialisation ────────────────────────────────────

if not firebase_admin._apps:
    try:
        # Check if the service account key file exists
        if os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY):
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY)
            firebase_admin.initialize_app(cred)
        else:
            # Fallback: Attempt to use environment / default credentials (ADC)
            firebase_admin.initialize_app()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialise Firebase Admin SDK.\n"
            f"Please verify that the service account JSON file is placed at "
            f"'{FIREBASE_SERVICE_ACCOUNT_KEY}' or credentials are set in the environment.\n"
            f"Error details: {e}"
        )

db = firestore.client()


# ── Schema / Connection Initialisation ────────────────────────────────────────

def init_db() -> None:
    """
    Ensure the Firestore client connection is active.
    Collections are created implicitly on write, so no DDL migration is required.
    """
    # The client is already initialised. We perform a quick ping/check.
    try:
        db.collections()
    except Exception as e:
        raise ConnectionError(f"Could not connect to Firebase Firestore: {e}")


# ── Read helpers ──────────────────────────────────────────────────────────────

def load_tenders() -> pd.DataFrame:
    """
    Load all tenders from the Firestore 'tenders' collection.
    Returns a pandas DataFrame sorted by the 'created_at' field descending.
    """
    tenders_ref = db.collection("tenders")
    docs = tenders_ref.stream()

    rows = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        rows.append(data)

    df = pd.DataFrame(rows)

    required_cols = [
        "id", "project_name", "client_name", "value", "win_prob", "status",
        "primary_factor", "assignee", "starting_date", "deadline",
        "submission_method", "product_brand", "product_model", "pdf_path"
    ]

    if df.empty:
        # Return empty DataFrame with correct column structure
        return pd.DataFrame(columns=required_cols)

    # Sort in memory by created_at descending if available (SQLite had "ORDER BY id DESC")
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.sort_values(by="created_at", ascending=False)

    # Convert deadline and starting_date fields back to datetime.date objects for the app
    for col in ("deadline", "starting_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Ensure all required columns are present (fill missing ones with None/NaN)
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Reorder columns to align with the expected schema
    return df[required_cols]


def get_all_staff() -> list[str]:
    """
    Load all staff names from the Firestore 'staff' collection.
    Returns a list of names sorted alphabetically.
    """
    staff_ref = db.collection("staff")
    docs = staff_ref.stream()
    names = []
    for doc in docs:
        data = doc.to_dict()
        if "name" in data:
            names.append(data["name"])
    return sorted(names)


def db_is_empty() -> bool:
    """True when both tenders and staff collections have zero documents."""
    tenders_empty = len(db.collection("tenders").limit(1).get()) == 0
    staff_empty = len(db.collection("staff").limit(1).get()) == 0
    return tenders_empty and staff_empty


# ── Write helpers ─────────────────────────────────────────────────────────────

def add_staff(name: str, department: str) -> bool:
    """
    Insert a new staff member (name normalised to Title Case).
    Uses the normalised name as document ID to guarantee uniqueness.
    Returns False on duplicate.
    """
    norm_name = name.strip().title()
    doc_ref = db.collection("staff").document(norm_name)
    
    # Check if duplicate exists
    if doc_ref.get().exists:
        return False
        
    doc_ref.set({
        "name": norm_name,
        "department": department
    })
    return True


def delete_staff(name: str) -> None:
    """Delete a staff member by name."""
    norm_name = name.strip().title()
    db.collection("staff").document(norm_name).delete()


def add_tender(
    project: str,
    client_name: str,
    value: float,
    win_prob: float,
    status: str,
    factor: str,
    assignee: str,
    deadline,
    starting_date=None,
    submission_method: str = "",
    product_brand: str = "",
    product_model: str = "",
    pdf_path: str = "",
) -> bool:
    """
    Insert a tender document.
    Silently skips if an identical (project_name, client_name, deadline) already exists.
    Returns True when inserted, False when skipped as duplicate.
    """
    p_name = project.strip()
    c_name = client_name.strip()
    dl_str = str(deadline)

    # Replicate unique constraint query
    duplicates = db.collection("tenders") \
        .where("project_name", "==", p_name) \
        .where("client_name", "==", c_name) \
        .where("deadline", "==", dl_str) \
        .limit(1).get()

    if len(duplicates) > 0:
        return False

    # Insert new document with automatic ID and server creation timestamp
    doc_ref = db.collection("tenders").document()
    doc_ref.set({
        "project_name": p_name,
        "client_name": c_name,
        "value": float(value),
        "win_prob": float(win_prob),
        "status": status,
        "primary_factor": factor,
        "assignee": assignee.strip().title(),
        "starting_date": str(starting_date) if starting_date else None,
        "deadline": dl_str,
        "submission_method": submission_method,
        "product_brand": product_brand,
        "product_model": product_model,
        "pdf_path": pdf_path or "",
        "created_at": firestore.SERVER_TIMESTAMP
    })
    return True


def update_tender(
    id: str,
    project: str,
    client_name: str,
    value: float,
    status: str,
    factor: str,
    assignee: str,
    deadline,
    starting_date=None,
    submission_method: str = "",
    product_brand: str = "",
    product_model: str = "",
    pdf_path: str | None = None,
) -> None:
    """Update an existing tender document by Firestore ID."""
    doc_ref = db.collection("tenders").document(id)
    update_data = {
        "project_name": project,
        "client_name": client_name,
        "value": float(value),
        "status": status,
        "primary_factor": factor,
        "assignee": assignee,
        "starting_date": str(starting_date) if starting_date else None,
        "deadline": str(deadline),
        "submission_method": submission_method,
        "product_brand": product_brand,
        "product_model": product_model
    }
    if pdf_path is not None:
        update_data["pdf_path"] = pdf_path

    doc_ref.update(update_data)


def delete_tender(id: str) -> None:
    """Delete a tender document by Firestore ID."""
    db.collection("tenders").document(id).delete()


def recalculate_all_probabilities(predict_fn) -> None:
    """
    Re-run the AI engine on every tender and persist the new win_prob.
    Also re-evaluates and updates the best Key Driver (primary_factor).
    Called after a bulk import or pipeline save so scores reflect the full dataset.
    """
    df = load_tenders()
    if df.empty:
        return
        
    from config import ATTRIBUTES
    
    for _, row in df.iterrows():
        best_prob = -1
        best_fac = ATTRIBUTES[0]

        for fac in ATTRIBUTES:
            res = predict_fn(
                row["value"], row["client_name"],
                fac, row["assignee"], df,
            )
            if res.probability > best_prob:
                best_prob = res.probability
                best_fac = fac

        # Update in Firestore
        doc_ref = db.collection("tenders").document(row["id"])
        doc_ref.update({
            "win_prob": float(best_prob),
            "primary_factor": best_fac
        })
