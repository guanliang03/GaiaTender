# test_firebase.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    print("[SUCCESS] firebase-admin is installed and imported successfully.")
except ImportError as e:
    print(f"[ERROR] Failed to import firebase-admin: {e}")
    sys.exit(1)

from config import FIREBASE_SERVICE_ACCOUNT_KEY
import os

print(f"Service account key path config: {FIREBASE_SERVICE_ACCOUNT_KEY}")
if os.path.exists(FIREBASE_SERVICE_ACCOUNT_KEY):
    print("[INFO] serviceAccountKey.json exists!")
else:
    print("[WARNING] serviceAccountKey.json does not exist in the root directory. Firestore connection might fail if no application default credentials exist.")

try:
    import database
    print("[INFO] Attempting to initialize database and count documents...")
    database.init_db()
    tenders_count = len(database.db.collection("tenders").limit(5).get())
    print(f"[SUCCESS] Connected to Firestore! Found {tenders_count} (up to 5 sampled) tenders.")
except Exception as e:
    print(f"[ERROR] Failed to connect/query Firestore: {e}")
