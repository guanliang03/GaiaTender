# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration — change constants here only, never in other modules.
# ─────────────────────────────────────────────────────────────────────────────

DB_FILE = "tender_system_final.db"

ATTRIBUTES = [
    "Relationship & Reputation",
    "Price",
    "Delivery & Timeline",
    "Technical Capability",
]

PIPELINE_STAGES = [
    "Qualified Lead",
    "Drafting Proposal",
    "Pending Approval",
    "Submitted",
    "Won",
    "Lost",
    "Untracked",
]

CLOSED_STATUSES = {"Won", "Lost"}

# ── AI scoring weights (6 components, 100 pts total) ─────────────────────────
#
#  Component              Max   What it measures
#  ─────────────────────  ────  ──────────────────────────────────────────────
#  STRATEGY                25   Chosen attribute vs client buying behaviour
#  ASSIGNEE                22   Lead's historical win-rate
#  RELATIONSHIP            20   Depth + quality of prior dealings with client
#  VALUE_FIT               13   Project size vs our proven deal-size range
#  DEADLINE_URGENCY        10   Days remaining before submission deadline
#  COMPETITION_DENSITY     10   Competing active bids for the same client
#
WEIGHT_STRATEGY           = 25
WEIGHT_ASSIGNEE           = 22
WEIGHT_RELATIONSHIP       = 20
WEIGHT_VALUE_FIT          = 13
WEIGHT_DEADLINE_URGENCY   = 10
WEIGHT_COMPETITION_DENSITY = 10

# ── Assignee win-rate bands ───────────────────────────────────────────────────
ASSIGNEE_RATE_HIGH   = 0.60   # ≥ 60 % win-rate → high performer
ASSIGNEE_RATE_MEDIUM = 0.35   # ≥ 35 % win-rate → average performer

# ── Relationship depth thresholds ─────────────────────────────────────────────
# Depth is now value-weighted: 60% from total contract value transacted with the
# client, 40% from raw deal count.  This prevents a small-ticket client with 3
# deals scoring the same as a large-ticket client worth RM 5M across 2 projects.
RELATIONSHIP_STRONG         = 3      # deal-count threshold for "Strong" tier
RELATIONSHIP_SOME           = 1      # deal-count threshold for "Developing" tier
RELATIONSHIP_WIN_RATE_MIN   = 0.50   # win-rate threshold for a "positive" relationship
RELATIONSHIP_STRONG_VALUE_X = 3.0    # total value ≥ 3× median won deal → strong value score
RELATIONSHIP_VALUE_WEIGHT   = 0.60   # 60 % of depth score based on value, 40 % on count

# ── Value-fit tolerance (±% around median of Won deal values) ─────────────────
VALUE_FIT_TOLERANCE = 0.40   # ±40 % → comfortable zone

# ── Deadline urgency thresholds (days remaining) ──────────────────────────────
DEADLINE_COMFORTABLE_DAYS = 21   # ≥ 21 days → full score (comfortable)
DEADLINE_TIGHT_DAYS       =  7   # 7–20 days → partial score (tight)
# < 7 days                        → low score  (critical)

# ── Competition density thresholds (other active bids, same client) ───────────
COMPETITION_EXCLUSIVE_MAX  = 0   # 0 other active bids → exclusive focus
COMPETITION_CROWDED_MIN    = 3   # ≥ 3 other active bids → spread-too-thin warning

# ── Confidence indicator (minimum closed-deal data points) ───────────────────
CONFIDENCE_HIGH_MIN        = 10   # ≥ 10 closed deals in evidence base → High
CONFIDENCE_MODERATE_MIN   =  4   #  4–9 closed deals → Moderate
CONFIDENCE_LOW_MIN        =  1   #  1–3 closed deals → Low
#                                  0 closed deals     → Insufficient

# ── Risk bands ────────────────────────────────────────────────────────────────
RISK_HIGH_MAX   = 40
RISK_MEDIUM_MAX = 70

# ── CSV import/export column order ───────────────────────────────────────────
CSV_TEMPLATE_COLUMNS = [
    "project_name",
    "client_name",
    "value",
    "primary_factor",
    "assignee",
    "starting_date",
    "deadline",
    "submission_method",
    "product_brand",
    "product_model",
    "status",
]
