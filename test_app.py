# test_app.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit AppTest suite for Smart Tender System.
# Run with: pytest test_app.py
# ─────────────────────────────────────────────────────────────────────────────

import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest


class TestSmartTenderSystemApp(unittest.TestCase):

    @patch("database.db_is_empty")
    @patch("database.get_all_staff")
    @patch("database.load_tenders")
    def test_onboarding_screen_when_empty(self, mock_load, mock_get_staff, mock_is_empty):
        """Verify that the onboarding page is shown when the database is completely empty."""
        # Mock database layer returning empty state
        mock_is_empty.return_value = True
        mock_get_staff.return_value = []
        mock_load.return_value = pd.DataFrame()

        # Load the Streamlit app
        at = AppTest.from_file("app.py")
        at.run()

        # 1. Assert onboarding page title or subheader is visible
        self.assertTrue(any("Smart Tender System" in title.value for title in at.title))
        self.assertTrue(any("Welcome — No data found yet" in sub.value for sub in at.subheader))
        
        # 2. Check sidebar header
        self.assertTrue(any("Sales Support Team" in sidebar_title.value for sidebar_title in at.sidebar.title))

    @patch("database.db_is_empty")
    @patch("database.get_all_staff")
    @patch("database.load_tenders")
    def test_main_dashboard_when_populated(self, mock_load, mock_get_staff, mock_is_empty):
        """Verify that the main tabs and sidebar are displayed when the database has data."""
        # Mock database layer returning populated state
        mock_is_empty.return_value = False
        mock_get_staff.return_value = ["Alice Smith", "Bob Jones"]
        mock_load.return_value = pd.DataFrame([
            {
                "id": "doc_id_123",
                "project_name": "Standard Lab Equipments",
                "client_name": "National Hospital",
                "value": 150000.0,
                "win_prob": 75.0,
                "status": "Drafting Proposal",
                "primary_factor": "Price",
                "assignee": "Alice Smith",
                "starting_date": None,
                "deadline": "2026-08-01",
                "submission_method": "Online Submission",
                "product_brand": "Thermo",
                "product_model": "Centrifuge X",
                "pdf_path": ""
            }
        ])

        # Load the Streamlit app
        at = AppTest.from_file("app.py")
        at.run()

        # 1. Verify main system title
        self.assertTrue(any("Smart Tender System" in title.value for title in at.title))

        # 2. Check sidebar displays current staff members
        sidebar_captions = [caption.value for caption in at.sidebar.caption]
        self.assertTrue(any("Alice Smith" in cap for cap in sidebar_captions))
        self.assertTrue(any("Bob Jones" in cap for cap in sidebar_captions))

        # 3. Check tabs exist
        # AppTest tabs are accessible. We verify the tabs container can load without crashing.
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
