import io, sys, sqlite3
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from config import DB_FILE

conn = sqlite3.connect(DB_FILE)

# ── Step 1: normalise all tenders.assignee to Title Case ──────────────────────
rows = conn.execute("SELECT id, assignee FROM tenders").fetchall()
for row_id, assignee in rows:
    normalised = assignee.strip().title()
    if normalised != assignee:
        print(f"  tender #{row_id}: '{assignee}' -> '{normalised}'")
        conn.execute("UPDATE tenders SET assignee=? WHERE id=?", (normalised, row_id))

# ── Step 2: figure out which staff names are duplicates after title-casing ────
staff_rows = conn.execute("SELECT id, name FROM staff ORDER BY id").fetchall()

# group by normalised name
from collections import defaultdict
groups = defaultdict(list)
for staff_id, name in staff_rows:
    groups[name.strip().title()].append((staff_id, name))

# ── Step 3: for each group, keep the lowest id, delete the rest ───────────────
for norm, members in groups.items():
    keep_id = members[0][0]
    # Rename the keeper if needed
    if members[0][1] != norm:
        print(f"  staff #{keep_id}: rename '{members[0][1]}' -> '{norm}'")
        conn.execute("UPDATE staff SET name=? WHERE id=?", (norm, keep_id))
    # Delete duplicates
    for dup_id, dup_name in members[1:]:
        print(f"  staff #{dup_id}: DELETE '{dup_name}' (duplicate of '{norm}')")
        conn.execute("DELETE FROM staff WHERE id=?", (dup_id,))

conn.commit()

print("\nStaff table after fix:")
for r in conn.execute("SELECT id, name FROM staff ORDER BY name").fetchall():
    print(f"  [{r[0]}] {r[1]}")

print("\nDistinct assignees in tenders:")
for r in conn.execute("SELECT DISTINCT assignee FROM tenders ORDER BY assignee").fetchall():
    print(f"  {r[0]}")

conn.close()
print("\nDone.")
