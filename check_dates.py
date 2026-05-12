import sqlite3, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
from config import DB_FILE
conn = sqlite3.connect(DB_FILE)

rows = conn.execute("SELECT id, starting_date, deadline, project_name FROM tenders WHERE project_name LIKE '%FREEZE-DRY%'").fetchall()
for r in rows:
    print(f"id={r[0]}  start={r[1]}  deadline={r[2]}")
    print(f"  {r[3][:70]}")

bad = conn.execute("SELECT COUNT(*) FROM tenders WHERE starting_date < '2000-01-01' AND starting_date IS NOT NULL").fetchone()[0]
print(f"\nRows with pre-2000 start date: {bad}")
conn.close()
