# database.py
# ─────────────────────────────────────────────────────────────────────────────
# SQLite schema, CRUD helpers.
# No seed / dummy data — the DB starts empty.
# Zero Streamlit dependency.
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3

import pandas as pd

from config import DB_FILE


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they do not already exist. Migrate schema if needed."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS tenders (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name      TEXT    NOT NULL,
            client_name       TEXT    NOT NULL,
            value             REAL    NOT NULL,
            win_prob          REAL    DEFAULT 0,
            status            TEXT    NOT NULL DEFAULT 'Qualified Lead',
            primary_factor    TEXT    NOT NULL,
            assignee          TEXT    NOT NULL,
            starting_date     DATE,
            deadline          DATE    NOT NULL,
            submission_method TEXT,
            product_brand     TEXT,
            product_model     TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS staff (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT UNIQUE NOT NULL,
            department TEXT
        )
    """)

    # ── Migrate: remove bid_amount / add new columns if upgrading old DB ──────
    c.execute("PRAGMA table_info(tenders)")
    existing_cols = {row[1] for row in c.fetchall()}

    if "bid_amount" in existing_cols:
        # Recreate table without bid_amount, preserving all data
        c.execute("""
            CREATE TABLE IF NOT EXISTS tenders_new (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name      TEXT    NOT NULL,
                client_name       TEXT    NOT NULL,
                value             REAL    NOT NULL,
                win_prob          REAL    DEFAULT 0,
                status            TEXT    NOT NULL DEFAULT 'Qualified Lead',
                primary_factor    TEXT    NOT NULL,
                assignee          TEXT    NOT NULL,
                starting_date     DATE,
                deadline          DATE    NOT NULL,
                submission_method TEXT,
                product_brand     TEXT,
                product_model     TEXT
            )
        """)
        c.execute("""
            INSERT INTO tenders_new
                (id, project_name, client_name, value, win_prob, status,
                 primary_factor, assignee, deadline)
            SELECT id, project_name, client_name, value, win_prob, status,
                   primary_factor, assignee, deadline
            FROM tenders
        """)
        c.execute("DROP TABLE tenders")
        c.execute("ALTER TABLE tenders_new RENAME TO tenders")

    else:
        # Add any missing new columns to an existing up-to-date table
        for col, typedef in [
            ("starting_date",     "DATE"),
            ("submission_method", "TEXT"),
            ("product_brand",     "TEXT"),
            ("product_model",     "TEXT"),
        ]:
            if col not in existing_cols:
                c.execute(f"ALTER TABLE tenders ADD COLUMN {col} {typedef}")

    conn.commit()
    conn.close()


# ── Read helpers ──────────────────────────────────────────────────────────────

def load_tenders() -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM tenders ORDER BY id DESC", conn)
    conn.close()
    if not df.empty:
        for col in ("deadline", "starting_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def get_all_staff() -> list[str]:
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT name FROM staff ORDER BY name", conn)
    conn.close()
    return df["name"].tolist()


def db_is_empty() -> bool:
    """True when both tenders and staff tables have zero rows."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count(*) FROM tenders")
    t = c.fetchone()[0]
    c.execute("SELECT count(*) FROM staff")
    s = c.fetchone()[0]
    conn.close()
    return t == 0 and s == 0


# ── Write helpers ─────────────────────────────────────────────────────────────

def add_staff(name: str, department: str) -> bool:
    """Insert a new staff member. Returns False on duplicate name."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            "INSERT INTO staff (name, department) VALUES (?, ?)", (name, department)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_staff(name: str) -> None:
    """Delete a staff member by name."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM staff WHERE name=?", (name,))
    conn.commit()
    conn.close()


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
) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO tenders
           (project_name, client_name, value, win_prob, status, primary_factor,
            assignee, starting_date, deadline, submission_method,
            product_brand, product_model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (project, client_name, value, win_prob, status, factor, assignee,
         str(starting_date) if starting_date else None,
         str(deadline), submission_method, product_brand, product_model),
    )
    conn.commit()
    conn.close()


def update_tender(
    id: int,
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
) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """UPDATE tenders
           SET project_name=?, client_name=?, value=?, status=?,
               primary_factor=?, assignee=?, starting_date=?, deadline=?,
               submission_method=?, product_brand=?, product_model=?
           WHERE id=?""",
        (project, client_name, value, status, factor, assignee,
         str(starting_date) if starting_date else None,
         str(deadline), submission_method, product_brand, product_model, id),
    )
    conn.commit()
    conn.close()


def delete_tender(id: int) -> None:
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM tenders WHERE id=?", (id,))
    conn.commit()
    conn.close()


def recalculate_all_probabilities(predict_fn) -> None:
    """
    Re-run the AI engine on every tender and persist the new win_prob.
    Also re-evaluates and updates the best Key Driver (primary_factor).
    Called after a bulk import or pipeline save so scores reflect the full dataset.
    """
    df = load_tenders()
    if df.empty:
        return
    conn = sqlite3.connect(DB_FILE)
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

        conn.execute(
            "UPDATE tenders SET win_prob=?, primary_factor=? WHERE id=?",
            (best_prob, best_fac, row["id"]),
        )
    conn.commit()
    conn.close()
