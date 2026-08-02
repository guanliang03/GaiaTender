# Smart Tender Prediction and Groundwork Tracking System

An AI-powered dashboard built with Streamlit and Firebase Firestore to track sales tenders, analyze workload distribution, and predict win probabilities using a rule-based weighted scoring engine.

---

## Quick Start (Local Development)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Setup credentials:**
   - Place your Firebase service account key JSON file in the root directory and name it `serviceAccountKey.json`.
3. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

On the first launch, if your database has no records, you will see the onboarding screen where you can add your first staff member or download a CSV template to seed your data.

---

## File Structure

```
GaiaTender/
├── app.py                      # Main entry point (sidebar, onboarding gate, and tabs)
├── config.py                   # Central configuration (weights, thresholds, stages)
├── database.py                 # Firebase Firestore schema and CRUD operations
├── ai_engine.py                # 6-rule AI scoring engine + prediction logic
├── charts.py                   # Matplotlib chart builders + ReportLab PDF exporter
├── ui_onboarding.py            # Empty-state screen with CSV template generator
├── ui_score_breakdown.py       # Reusable UI component for live score visualization
├── ui_tab_new_entry.py         # Tab 1: Manual form + CSV/Excel bulk upload
├── ui_tab_pipeline.py          # Tab 2: Interactive editable data grid & workload charts
├── ui_tab_report.py            # Tab 3: Detailed AI score breakdowns & PDF report exporter
├── ui_tab_performance.py       # Tab 4: Win/Loss performance analytics & relationships table
└── ui_tab_gaia_analysis.py     # Tab 5: Plotly-powered advanced data analytics
```

---

## AI Scoring System (6 Components, 100 pts max)

The prediction engine evaluates 6 independent dimensions to calculate a win probability:

| # | Rule | Max Points | Logic / Measures |
|---|---|---|---|
| 1 | **Strategy** | 25 | Matches the chosen bidding attribute (Price, Technical Capability, etc.) vs. historical client buying behavior. |
| 2 | **Assignee** | 22 | Lead's historical performance (win-rate) across all completed deals. |
| 3 | **Relationship Depth** | 20 | Combines quality (win rate) and quantity (value-weighted contract volume) of prior client dealings. |
| 4 | **Value Fit** | 13 | Evaluates if the project budget fits within ±40% of the median value of historically Won deals. |
| 5 | **Deadline Urgency** | 10 | Evaluates days remaining before submission. Tighter deadlines reduce the score. |
| 6 | **Competition Density** | 10 | Penalizes bid scores if there are multiple active bidding tracks running concurrently for the same client. |

---

## Streamlit Cloud Deployment

To host this application permanently on Streamlit Community Cloud:

1. Push your code to your GitHub repository (excluding `serviceAccountKey.json` which is ignored in `.gitignore`).
2. Go to [Streamlit Share](https://share.streamlit.io/) and deploy your repository using `app.py` as the main file path.
3. In your App settings, go to the **Secrets** tab and paste your Firebase service account JSON under a `[firebase]` header in TOML format:
   ```toml
   [firebase]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-private-key-id"
   private_key = """-----BEGIN PRIVATE KEY-----
   ...your private key goes here...
   -----END PRIVATE KEY-----
   """
   client_email = "your-client-email"
   # (Copy-paste the remaining key-value fields from serviceAccountKey.json)
   ```
4. Click **Save** and the app will automatically reboot and connect to Firestore securely.

---

## Customizing Rules & Weights

All scoring weights and threshold constants are centralized in `config.py`. Changing a weight there (e.g., modifying `WEIGHT_STRATEGY`) instantly updates the prediction scoring calculations across all tabs, UI score bars, and exported PDF reports. No database schema migrations or changes in other files are needed.
