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
def clean_private_key(pk: str) -> str:
    # Resolve any literal "\n" / "\r" text into real characters
    pk = pk.replace("\\n", "\n").replace("\\r", "\r")
    
    header = "-----BEGIN PRIVATE KEY-----"
    footer = "-----END PRIVATE KEY-----"
    
    pk_clean = pk.strip()
    if pk_clean.startswith(header):
        pk_clean = pk_clean[len(header):]
    if pk_clean.endswith(footer):
        pk_clean = pk_clean[:-len(footer)]
    
    # Strip any characters that are not valid base64 components (whitespaces, newlines, etc.)
    import re
    payload = re.sub(r'[^A-Za-z0-9+/=]', '', pk_clean)
    
    # Reconstruct standard 64-character per line PEM format
    lines = [payload[i:i+64] for i in range(0, len(payload), 64)]
    return f"{header}\n" + "\n".join(lines) + f"\n{footer}\n"


# ── Safe Firebase Admin SDK Initialisation ────────────────────────────────────

if not firebase_admin._apps:
    try:
        # 1. Check if the service account key file exists locally
        if os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY):
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY)
            firebase_admin.initialize_app(cred)
        else:
            # 2. Check if credentials can be loaded from Streamlit secrets
            firebase_creds = None
            try:
                import streamlit as st
                if "firebase" in st.secrets:
                    sec = st.secrets["firebase"]
                    if isinstance(sec, str):
                        import json
                        firebase_creds = json.loads(sec)
                    elif hasattr(sec, "to_dict"):  # Streamlit Secrets object can be converted to dict
                        firebase_creds = sec.to_dict()
                    elif isinstance(sec, dict):
                        firebase_creds = sec
                    else:
                        # Fallback try dict conversion
                        firebase_creds = dict(sec)
                    
                    # If it's a dictionary with a single key 'credentials' containing the JSON string
                    if firebase_creds and "credentials" in firebase_creds and len(firebase_creds) == 1:
                        import json
                        firebase_creds = json.loads(firebase_creds["credentials"])
            except Exception:
                pass

            if firebase_creds:
                if isinstance(firebase_creds, dict) and "private_key" in firebase_creds:
                    pk = firebase_creds["private_key"]
                    if isinstance(pk, str):
                        # Clean and format the private key to be 100% correct
                        firebase_creds["private_key"] = clean_private_key(pk)
                
                cred = credentials.Certificate(firebase_creds)
                firebase_admin.initialize_app(cred)
            else:
                # 3. Fallback: Attempt to use environment / default credentials (ADC)
                firebase_admin.initialize_app()
    except Exception as e:
        keys_info = ""
        debug_info = ""
        try:
            if 'firebase_creds' in locals() and firebase_creds:
                keys_info = f" Parsed keys: {list(firebase_creds.keys())}."
                pk_raw = firebase_creds.get("private_key", "")
                if isinstance(pk_raw, str):
                    # Mask letters and digits to preserve security
                    masked_raw = "".join(c if not c.isalnum() else "X" for c in pk_raw)
                    debug_info = (
                        f"\n[Debug] Raw private key length: {len(pk_raw)}. "
                        f"Ends with (masked): {repr(masked_raw[-100:])}."
                    )
        except Exception as debug_err:
            debug_info = f"\n[Debug] Failed to gather debug info: {debug_err}"
            
        raise RuntimeError(
            f"Failed to initialise Firebase Admin SDK.{keys_info}{debug_info}\n"
            f"Please verify that the service account JSON file is placed at "
            f"'{FIREBASE_SERVICE_ACCOUNT_KEY}', defined in Streamlit secrets, or credentials are set in the environment.\n"
            f"Error details: {e}"
        )

try:
    db = firestore.client()
except Exception as e:
    raise RuntimeError(
        "Could not initialize Firebase Firestore client. "
        "This usually means the Firebase credentials were not found or are invalid.\n\n"
        "If you are running on Streamlit Cloud, please make sure you have added your service account credentials "
        "to the 'Secrets' manager in your Streamlit Cloud Dashboard (Settings > Secrets) in the following format:\n\n"
        "[firebase]\n"
        "type = \"service_account\"\n"
        "project_id = \"your-project-id\"\n"
        "private_key_id = \"your-private-key-id\"\n"
        "private_key = \"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n\"\n"
        "client_email = \"your-client-email\"\n"
        "...\n\n"
        f"Original error details: {e}"
    )


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

    # Automatically treat "Submitted" tenders as "Untracked" if the deadline has passed
    if "status" in df.columns and "deadline" in df.columns:
        from datetime import date
        today = date.today()
        mask = (df["status"] == "Submitted") & (df["deadline"].apply(lambda d: isinstance(d, date) and d < today))
        df.loc[mask, "status"] = "Untracked"

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


def recalculate_all_probabilities(predict_fn=None) -> None:
    """
    Re-run the AI engine on every tender and persist the new win_prob.
    Also re-evaluates and updates the best Key Driver (primary_factor).
    Called after a bulk import or pipeline save so scores reflect the full dataset.
    """
    df = load_tenders()
    if df.empty:
        return

    from config import ATTRIBUTES
    from ai_engine import pick_best_attribute, predict

    for _, row in df.iterrows():
        deadline = row.get("deadline")
        p_brand  = str(row.get("product_brand", "") or "")
        p_model  = str(row.get("product_model", "") or "")
        p_name   = str(row.get("project_name", "") or "")

        best_fac, best_res = pick_best_attribute(
            project_value=row["value"],
            client_name=row["client_name"],
            assignee=row["assignee"],
            history=df,
            attributes=ATTRIBUTES,
            deadline=deadline,
            product_brand=p_brand,
            product_model=p_model,
            project_name=p_name,
            status=row.get("status", ""),
        )

        # Update in Firestore
        doc_ref = db.collection("tenders").document(row["id"])
        doc_ref.update({
            "win_prob": float(best_res.probability),
            "primary_factor": best_fac
        })
