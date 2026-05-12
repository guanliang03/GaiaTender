# ai_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Rule-based weighted scoring engine — 6 independent components.
# Pure Python / pandas. Zero Streamlit / DB dependency.
#
# SCORING RULES
# ─────────────
#  1. STRATEGY (25 pts)   — attribute vs client buying behaviour
#  2. ASSIGNEE (22 pts)   — lead's historical win-rate
#  3. RELATIONSHIP (20 pts) — depth + quality of prior client dealings
#  4. VALUE FIT (13 pts)  — project size vs proven deal-size range
#  5. DEADLINE URGENCY (10 pts) — days remaining before submission
#  6. COMPETITION DENSITY (10 pts) — concurrent active bids, same client
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from config import (
    ATTRIBUTES,
    ASSIGNEE_RATE_HIGH,
    ASSIGNEE_RATE_MEDIUM,
    COMPETITION_CROWDED_MIN,
    COMPETITION_EXCLUSIVE_MAX,
    CONFIDENCE_HIGH_MIN,
    CONFIDENCE_LOW_MIN,
    CONFIDENCE_MODERATE_MIN,
    DEADLINE_COMFORTABLE_DAYS,
    DEADLINE_TIGHT_DAYS,
    RELATIONSHIP_SOME,
    RELATIONSHIP_STRONG,
    RELATIONSHIP_STRONG_VALUE_X,
    RELATIONSHIP_VALUE_WEIGHT,
    RELATIONSHIP_WIN_RATE_MIN,
    VALUE_FIT_TOLERANCE,
    WEIGHT_ASSIGNEE,
    WEIGHT_COMPETITION_DENSITY,
    WEIGHT_DEADLINE_URGENCY,
    WEIGHT_RELATIONSHIP,
    WEIGHT_STRATEGY,
    WEIGHT_VALUE_FIT,
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    """Result for one scoring rule."""
    name: str           # e.g. "Strategy"
    score: int          # points awarded
    max_score: int      # maximum possible points
    verdict: str        # short emoji + label, e.g. "✅ Matches Client Preference"
    rationale: str      # 1-2 sentence plain-English explanation
    data_source: str    # what data was used

    @property
    def pct(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score else 0.0


@dataclass
class PredictionResult:
    """Full output of the scoring engine."""
    probability: int                      # 1–99
    rules: list[RuleResult] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)   # catastrophic single-rule scores
    confidence_level: str = "Insufficient"               # High / Moderate / Low / Insufficient
    action_summary: str = ""                             # natural-language recommendation

    @property
    def total_max(self) -> int:
        return sum(r.max_score for r in self.rules)

    @property
    def risk_level(self) -> str:
        if self.probability < 40:
            return "HIGH RISK (DEPRIORITISE)"
        if self.probability <= 70:
            return "MEDIUM RISK"
        return "LOW RISK (FOCUS)"

    @property
    def risk_colour(self) -> str:
        return {
            "HIGH RISK (DEPRIORITISE)": "red",
            "MEDIUM RISK":              "orange",
            "LOW RISK (FOCUS)":         "green",
        }[self.risk_level]

    @property
    def insight_str(self) -> str:
        return "  |  ".join(f"{r.verdict} ({r.score}/{r.max_score})" for r in self.rules)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sigmoid_stretch(raw: float, total_max: float) -> int:
    """
    Map a raw score (0–total_max) to a probability (1–99) using a sigmoid
    curve so scores are better distributed across the range instead of
    clustering in the 60-80 band.
    """
    if total_max <= 0:
        return 50
    pct = raw / total_max          # 0.0 – 1.0
    # Centre sigmoid at 0.55 (slight optimism bias for mid-range scores)
    x = 10 * (pct - 0.55)
    stretched = 1 / (1 + math.exp(-x))   # 0.0 – 1.0
    return int(max(1, min(99, round(stretched * 99))))


def _detect_red_flags(rules: list[RuleResult]) -> list[str]:
    """Return descriptions for any rule scoring ≤ 20% of its max."""
    flags = []
    for r in rules:
        if r.max_score > 0 and r.pct <= 20:
            flags.append(f"**{r.name}** is critically weak ({r.score}/{r.max_score} pts) — {r.verdict}")
    return flags


def _calc_confidence(history: pd.DataFrame) -> str:
    """Rate confidence based on closed-deal evidence volume."""
    closed = len(history[history["status"].isin(["Won", "Lost"])])
    if closed >= CONFIDENCE_HIGH_MIN:
        return "High"
    if closed >= CONFIDENCE_MODERATE_MIN:
        return "Moderate"
    if closed >= CONFIDENCE_LOW_MIN:
        return "Low"
    return "Insufficient"


# ── Public API ────────────────────────────────────────────────────────────────

def predict(
    project_value: float,
    client_name: str,
    primary_factor: str,
    assignee: str,
    history: pd.DataFrame,
    deadline: date | None = None,
    tender_id: int | None = None,
) -> PredictionResult:
    """
    Run all six scoring rules and return a PredictionResult.

    Parameters
    ----------
    project_value  : Estimated budget / contract value.
    client_name    : Name of the client organisation.
    primary_factor : Chosen competitive attribute (must be in ATTRIBUTES).
    assignee       : Name of the lead staff member.
    history        : Full tenders DataFrame (used as the evidence base).
    deadline       : Submission deadline date (optional, enables Rule 5).
    tender_id      : ID of the current tender to exclude from density count.
    """
    rules: list[RuleResult] = []

    rules.append(_rule_strategy(primary_factor, client_name, history))
    rules.append(_rule_assignee(assignee, history))
    rules.append(_rule_relationship(client_name, history))
    rules.append(_rule_value_fit(project_value, history))
    rules.append(_rule_deadline_urgency(deadline))
    rules.append(_rule_competition_density(client_name, history, tender_id))

    raw = sum(r.score for r in rules)
    total_max = sum(r.max_score for r in rules)
    probability = _sigmoid_stretch(raw, total_max)

    red_flags = _detect_red_flags(rules)
    confidence = _calc_confidence(history)
    summary = generate_action_summary(rules, probability, red_flags, confidence)

    return PredictionResult(
        probability=probability,
        rules=rules,
        red_flags=red_flags,
        confidence_level=confidence,
        action_summary=summary,
    )


def pick_best_attribute(
    project_value: float,
    client_name: str,
    assignee: str,
    history: pd.DataFrame,
    attributes: list[str],
    deadline: date | None = None,
    tender_id: int | None = None,
) -> tuple[str, PredictionResult]:
    """
    Select the best competitive attribute for a new tender, free of bias.

    Algorithm
    ---------
    1. Score all attributes via predict() (including deadline & competition rules).
    2. Identify all attributes that share the maximum probability (tied group).
    3. Among tied attributes, choose the one **least represented** in existing
       tenders' primary_factor column — this ensures diverse training data and
       prevents a single attribute from snowballing just because it was
       arbitrarily chosen first.
    4. If still tied after usage check, pick at random.
    """
    import random

    scored: dict[str, PredictionResult] = {
        attr: predict(project_value, client_name, attr, assignee, history,
                      deadline=deadline, tender_id=tender_id)
        for attr in attributes
    }

    max_prob = max(r.probability for r in scored.values())
    tied = [attr for attr, r in scored.items() if r.probability == max_prob]

    if len(tied) == 1:
        chosen = tied[0]
    else:
        if not history.empty and "primary_factor" in history.columns:
            usage = history["primary_factor"].value_counts()
            min_count = min(usage.get(attr, 0) for attr in tied)
            least_used = [attr for attr in tied if usage.get(attr, 0) == min_count]
        else:
            least_used = tied

        chosen = random.choice(least_used)

    return chosen, scored[chosen]


# ── Rule 1 — Strategy alignment ───────────────────────────────────────────────

def _rule_strategy(
    primary_factor: str,
    client_name: str,
    history: pd.DataFrame,
) -> RuleResult:

    W = WEIGHT_STRATEGY
    won = history[history["status"] == "Won"]

    client_wins = (
        won[won["client_name"].str.lower() == client_name.lower()]
        if client_name else pd.DataFrame()
    )

    # ── Path A: we have client-specific win data ──────────────────────────────
    if not client_wins.empty:
        factor_counts = client_wins["primary_factor"].value_counts()
        preferred = factor_counts.index[0]
        preferred_count = factor_counts.iloc[0]
        data_source = (
            f"{len(client_wins)} win(s) with this client; "
            f"'{preferred}' won {preferred_count}× "
            f"({preferred_count/len(client_wins)*100:.0f}%)"
        )

        if primary_factor == preferred:
            score = W
            verdict = "✅ Matches Client Preference"
            rationale = (
                f"This client has historically favoured '{preferred}' as the winning "
                f"factor in {preferred_count} of {len(client_wins)} won deals. "
                "Your strategy is perfectly aligned."
            )
        else:
            our_wins = len(client_wins[client_wins["primary_factor"] == primary_factor])
            if our_wins > 0:
                score = int(W * 0.55)
                verdict = "🟡 Partial Alignment"
                rationale = (
                    f"'{primary_factor}' has won {our_wins}× with this client, "
                    f"but '{preferred}' wins more often ({preferred_count}×). "
                    "Consider leading with the client's preferred driver."
                )
            else:
                score = int(W * 0.25)
                verdict = "⚠️ Strategy Misaligned"
                rationale = (
                    f"'{primary_factor}' has never won a deal with this client. "
                    f"Their historical preference is '{preferred}'. "
                    "Strongly consider realigning your strategy."
                )
        return RuleResult("Strategy", score, W, verdict, rationale, data_source)

    # ── Path B: no client history — use market-wide data ─────────────────────
    if not won.empty:
        market_counts = won["primary_factor"].value_counts()
        attr_wins = market_counts.get(primary_factor, 0)
        total_wins = len(won)
        data_source = f"No client history — using {total_wins} market win(s)"

        if attr_wins > 0:
            rate = attr_wins / total_wins
            score = int(W * (0.40 + 0.40 * rate))
            verdict = "✅ Proven Market Strategy"
            rationale = (
                f"'{primary_factor}' has won {attr_wins} of {total_wins} market deals "
                f"({rate * 100:.0f}%). No specific client data exists, but this is "
                "a validated approach."
            )
        else:
            score = int(W * 0.35)
            verdict = "ℹ️ Untested Strategy"
            rationale = (
                f"'{primary_factor}' has not won any recorded market deals yet. "
                "This does not mean it cannot work, but there is no evidence base."
            )
    else:
        score = int(W * 0.40)
        verdict = "ℹ️ No History Available"
        rationale = "No won deals exist in the system to benchmark strategy against."
        data_source = "No data"

    return RuleResult("Strategy", score, W, verdict, rationale, data_source)


# ── Rule 2 — Assignee performance ─────────────────────────────────────────────

def _rule_assignee(assignee: str, history: pd.DataFrame) -> RuleResult:
    W = WEIGHT_ASSIGNEE
    records = history[history["assignee"] == assignee]

    if records.empty:
        score = int(W * 0.50)
        return RuleResult(
            "Assignee", score, W,
            "ℹ️ No History",
            f"'{assignee}' has no recorded tenders in the system. "
            "A neutral score is applied until a track record is established.",
            "No data for this assignee",
        )

    closed = records[records["status"].isin(["Won", "Lost"])]
    total = len(closed)

    if total == 0:
        score = int(W * 0.50)
        return RuleResult(
            "Assignee", score, W,
            "ℹ️ No Closed Deals",
            f"'{assignee}' has tenders in the pipeline but none are closed yet. "
            "Neutral score applied.",
            f"{len(records)} open tender(s), 0 closed",
        )

    wins = len(closed[closed["status"] == "Won"])
    rate = wins / total
    data_source = f"{wins} win(s) / {total} closed deal(s) = {rate*100:.0f}% win rate"

    if rate >= ASSIGNEE_RATE_HIGH:
        score = W
        verdict = "⭐ High Performer"
        rationale = (
            f"'{assignee}' has won {wins} of {total} closed deals ({rate*100:.0f}%). "
            f"Win rate ≥ {ASSIGNEE_RATE_HIGH*100:.0f}% — a strong track record."
        )
    elif rate >= ASSIGNEE_RATE_MEDIUM:
        score = int(W * 0.60)
        verdict = "✅ Average Performer"
        rationale = (
            f"'{assignee}' has a {rate*100:.0f}% win rate ({wins}/{total} deals). "
            "Solid but room to improve. Consider additional support on this bid."
        )
    else:
        score = int(W * 0.20)
        verdict = "⚠️ Below Average"
        rationale = (
            f"'{assignee}' has only won {wins} of {total} closed deals ({rate*100:.0f}%). "
            "Consider pairing with a senior lead or reassigning."
        )

    return RuleResult("Assignee", score, W, verdict, rationale, data_source)


# ── Rule 3 — Relationship depth (value-weighted) ──────────────────────────────

def _rule_relationship(client_name: str, history: pd.DataFrame) -> RuleResult:
    W = WEIGHT_RELATIONSHIP

    if not client_name:
        score = 0
        return RuleResult(
            "Relationship", score, W,
            "❓ No Client Name",
            "Client name is missing — relationship depth cannot be assessed.",
            "N/A",
        )

    client_history = history[
        history["client_name"].str.lower() == client_name.lower()
    ]
    closed = client_history[client_history["status"].isin(["Won", "Lost"])]
    total_closed = len(closed)

    if total_closed == 0:
        score = int(W * 0.20)
        return RuleResult(
            "Relationship", score, W,
            "🆕 New Client",
            f"No prior closed deals found with '{client_name}'. "
            "Relationship risk is high — invest in pre-tender engagement.",
            "0 closed deals on record",
        )

    wins           = len(closed[closed["status"] == "Won"])
    win_rate       = wins / total_closed
    total_value    = closed["value"].sum()

    all_won = history[history["status"] == "Won"]
    median_deal_value = (
        all_won["value"].median() if not all_won.empty
        else (total_value / total_closed)
    )

    count_score = min(1.0, total_closed / RELATIONSHIP_STRONG)
    strong_value_threshold = RELATIONSHIP_STRONG_VALUE_X * median_deal_value
    value_score = (
        min(1.0, total_value / strong_value_threshold)
        if strong_value_threshold > 0 else 0.0
    )

    v_wt  = RELATIONSHIP_VALUE_WEIGHT
    depth_score = v_wt * value_score + (1 - v_wt) * count_score

    if depth_score >= 0.80:
        depth_label      = "Strong"
        depth_multiplier = 1.0
    elif depth_score >= 0.40:
        depth_label      = "Developing"
        depth_multiplier = 0.65
    else:
        depth_label = "Thin"
        depth_multiplier = 0.35

    if win_rate >= RELATIONSHIP_WIN_RATE_MIN:
        quality_label = "positive"
        quality_multiplier = 1.0
    elif win_rate > 0:
        quality_label = "mixed"
        quality_multiplier = 0.65
    else:
        quality_label = "negative (0 wins)"
        quality_multiplier = 0.25

    score = max(1, int(W * depth_multiplier * quality_multiplier))

    if score >= int(W * 0.80):
        verdict = "✅ Strong Relationship"
    elif score >= int(W * 0.50):
        verdict = "🟡 Developing Relationship"
    else:
        verdict = "⚠️ Weak / Negative Relationship"

    rationale = (
        f"Relationship with '{client_name}': {depth_label} depth "
        f"({total_closed} deal(s), RM {total_value:,.0f} total value). "
        f"Value score: {value_score:.0%} · Count score: {count_score:.0%} · "
        f"Composite depth: {depth_score:.0%}. "
        f"Win quality: {quality_label} ({wins}/{total_closed} won)."
    )
    data_source = (
        f"{total_closed} closed deal(s) · RM {total_value:,.0f} total value · "
        f"{wins} won ({win_rate*100:.0f}% win rate) · "
        f"Strong threshold: RM {strong_value_threshold:,.0f} (3× median)"
    )

    return RuleResult("Relationship", score, W, verdict, rationale, data_source)


# ── Rule 4 — Value fit ────────────────────────────────────────────────────────

def _rule_value_fit(project_value: float, history: pd.DataFrame) -> RuleResult:
    W = WEIGHT_VALUE_FIT
    won = history[history["status"] == "Won"]

    if won.empty or project_value <= 0:
        score = int(W * 0.50)
        return RuleResult(
            "Value Fit", score, W,
            "ℹ️ No Benchmark",
            "No won deals exist to define a comfortable deal-size range. "
            "Neutral score applied.",
            "No data",
        )

    won_values = won["value"].dropna()

    median_val = won_values.median()
    core_min = min(won_values.quantile(0.20), median_val * 0.80)
    core_max = max(won_values.quantile(0.80), median_val * 1.20)

    abs_min = min(won_values.min(), core_min * 0.50)
    abs_max = max(won_values.max(), core_max * 1.50)

    data_source = (
        f"Core sweet spot: RM {core_min:,.0f} – RM {core_max:,.0f}  "
        f"(Absolute bounds: RM {abs_min:,.0f} – RM {abs_max:,.0f})"
    )

    if core_min <= project_value <= core_max:
        score = W
        verdict = "✅ Ideal Deal Size"
        rationale = (
            f"This project (RM {project_value:,.0f}) falls squarely inside your core competency "
            "sweet spot (the most frequent volume of historically won deals)."
        )
    elif abs_min <= project_value <= abs_max:
        score = int(W * 0.65)
        verdict = "🟡 Stretching Capability"
        rationale = (
            f"This project (RM {project_value:,.0f}) sits outside your most common deal size, "
            "but is still within the bounds of your previously achieved historical extremes."
        )
    else:
        score = int(W * 0.25)
        direction = "above" if project_value > abs_max else "below"
        verdict = "⚠️ Unprecedented Scale"
        rationale = (
            f"This project (RM {project_value:,.0f}) is drastically {direction} your historically "
            "proven capability limit. Attempting to deliver at this unproven scale carries very high risk."
        )

    return RuleResult("Value Fit", score, W, verdict, rationale, data_source)


# ── Rule 5 — Deadline urgency ─────────────────────────────────────────────────

def _rule_deadline_urgency(deadline: date | None) -> RuleResult:
    W = WEIGHT_DEADLINE_URGENCY

    if deadline is None:
        score = int(W * 0.50)
        return RuleResult(
            "Deadline", score, W,
            "ℹ️ No Deadline Set",
            "No deadline was provided. A neutral score is applied.",
            "No deadline data",
        )

    today = date.today()
    days_left = (deadline - today).days

    if days_left < 0:
        score = 0
        verdict = "🔴 Deadline Passed"
        rationale = (
            f"The deadline was {abs(days_left)} day(s) ago ({deadline}). "
            "This tender can no longer be submitted."
        )
        data_source = f"Deadline: {deadline} · Days remaining: {days_left}"
    elif days_left >= DEADLINE_COMFORTABLE_DAYS:
        score = W
        verdict = "✅ Comfortable Timeline"
        rationale = (
            f"{days_left} day(s) remain until the deadline ({deadline}). "
            "There is sufficient time to prepare a high-quality proposal."
        )
        data_source = f"Deadline: {deadline} · Days remaining: {days_left}"
    elif days_left >= DEADLINE_TIGHT_DAYS:
        score = int(W * 0.60)
        verdict = "🟡 Tight Deadline"
        rationale = (
            f"Only {days_left} day(s) remain until the deadline ({deadline}). "
            "Proposal quality may be compromised under time pressure. "
            "Prioritise this tender immediately."
        )
        data_source = f"Deadline: {deadline} · Days remaining: {days_left}"
    else:
        score = int(W * 0.20)
        verdict = "⚠️ Critical — Deadline Imminent"
        rationale = (
            f"Only {days_left} day(s) until the deadline ({deadline}). "
            "Rushing a submission significantly reduces win probability. "
            "Consider whether the quality trade-off is acceptable."
        )
        data_source = f"Deadline: {deadline} · Days remaining: {days_left}"

    return RuleResult("Deadline", score, W, verdict, rationale, data_source)


# ── Rule 6 — Competition density ──────────────────────────────────────────────

def _rule_competition_density(
    client_name: str,
    history: pd.DataFrame,
    tender_id: int | None = None,
) -> RuleResult:
    W = WEIGHT_COMPETITION_DENSITY

    if not client_name or history.empty:
        score = int(W * 0.50)
        return RuleResult(
            "Competition", score, W,
            "ℹ️ No Data",
            "Unable to assess competition density — no client name or history.",
            "No data",
        )

    ACTIVE_STAGES = {"Qualified Lead", "Drafting Proposal", "Pending Approval", "Submitted"}
    active = history[
        (history["client_name"].str.lower() == client_name.lower()) &
        (history["status"].isin(ACTIVE_STAGES))
    ]

    # Exclude the current tender if editing an existing one
    if tender_id is not None and "id" in active.columns:
        active = active[active["id"] != tender_id]

    count = len(active)

    if count <= COMPETITION_EXCLUSIVE_MAX:
        score = W
        verdict = "✅ Exclusive Focus"
        rationale = (
            f"No other active bids exist for '{client_name}'. "
            "Your team can dedicate full attention to this tender."
        )
    elif count < COMPETITION_CROWDED_MIN:
        score = int(W * 0.60)
        verdict = "🟡 Shared Attention"
        rationale = (
            f"{count} other active bid(s) are running for '{client_name}'. "
            "Ensure adequate resources are allocated to each proposal."
        )
    else:
        score = int(W * 0.25)
        verdict = "⚠️ Spread Too Thin"
        rationale = (
            f"{count} simultaneous active bids for '{client_name}' risk diluting "
            "proposal quality. Consider deprioritising weaker opportunities."
        )

    data_source = f"{count} other active bid(s) for this client"
    return RuleResult("Competition", score, W, verdict, rationale, data_source)


# ── Action summary generator ──────────────────────────────────────────────────

def generate_action_summary(
    rules: list[RuleResult],
    probability: int,
    red_flags: list[str],
    confidence: str,
) -> str:
    """
    Synthesise a single natural-language recommendation from all rule results.
    """
    rule_map = {r.name: r for r in rules}

    strengths = [r.name for r in rules if r.pct >= 75]
    weaknesses = [r.name for r in rules if r.pct < 40]

    parts: list[str] = []

    # Opening line based on overall probability
    if probability >= 70:
        parts.append(f"🟢 **Strong opportunity ({probability}% win probability).**")
    elif probability >= 50:
        parts.append(f"🟡 **Moderate opportunity ({probability}% win probability).**")
    else:
        parts.append(f"🔴 **High-risk tender ({probability}% win probability).**")

    # Strengths
    if strengths:
        parts.append(f"Key strengths: {', '.join(strengths)}.")

    # Red flags / critical weaknesses
    if red_flags:
        parts.append(f"⚠️ Critical issues detected: {'; '.join(w for w in weaknesses)}.")

    # Specific actionable advice per weak rule
    advice = []
    deadline_rule = rule_map.get("Deadline")
    if deadline_rule and deadline_rule.pct < 40:
        advice.append("allocate extra resources now to meet the imminent deadline")

    rel_rule = rule_map.get("Relationship")
    if rel_rule and rel_rule.pct < 40:
        advice.append("invest in urgent pre-tender client engagement to strengthen the relationship")

    strategy_rule = rule_map.get("Strategy")
    if strategy_rule and strategy_rule.pct < 40:
        advice.append("reconsider your competitive strategy — it does not match this client's buying history")

    assignee_rule = rule_map.get("Assignee")
    if assignee_rule and assignee_rule.pct < 40:
        advice.append("consider pairing the assigned lead with a senior performer")

    comp_rule = rule_map.get("Competition")
    if comp_rule and comp_rule.pct < 40:
        advice.append("reduce concurrent bids for this client to improve proposal quality")

    if advice:
        parts.append("**Recommended actions:** " + "; ".join(advice).capitalize() + ".")

    # Confidence caveat
    if confidence in ("Low", "Insufficient"):
        parts.append(
            f"*Note: Confidence is **{confidence}** — limited historical data means "
            "this score should be treated as indicative only.*"
        )

    return "  \n".join(parts) if parts else "No specific recommendations at this time."


# ── Analytics helpers (used by Performance tab) ───────────────────────────────

def attribute_win_rates(
    completed: pd.DataFrame,
) -> list[tuple[str, float, int, int]]:
    """(attribute, win_rate_pct, wins, total) sorted by win-rate desc."""
    stats: list[tuple[str, float, int, int]] = []
    clean = completed.copy()
    clean["_fac"] = clean["primary_factor"].astype(str).str.strip()
    for attr in ATTRIBUTES:
        sub = clean[clean["_fac"] == attr]
        total = len(sub)
        if total == 0:
            continue
        wins = len(sub[sub["status"] == "Won"])
        stats.append((attr, wins / total * 100, wins, total))
    stats.sort(key=lambda x: x[1], reverse=True)
    return stats


def client_relationship_table(history: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame with one row per client, using the same
    value-weighted depth logic as _rule_relationship():
    client | Deals | Won | Lost | Win Rate % | Total Value (RM) | Depth
    """
    if history.empty:
        return pd.DataFrame()

    closed = history[history["status"].isin(["Won", "Lost"])].copy()
    if closed.empty:
        return pd.DataFrame()

    all_won = history[history["status"] == "Won"]
    median_deal_value = all_won["value"].median() if not all_won.empty else 0.0
    strong_value_threshold = RELATIONSHIP_STRONG_VALUE_X * median_deal_value

    rows = []
    for client, grp in closed.groupby("client_name"):
        total       = len(grp)
        wins        = len(grp[grp["status"] == "Won"])
        rate        = wins / total
        total_value = grp["value"].sum()

        count_score  = min(1.0, total / RELATIONSHIP_STRONG)
        value_score  = (
            min(1.0, total_value / strong_value_threshold)
            if strong_value_threshold > 0 else 0.0
        )
        depth_score  = RELATIONSHIP_VALUE_WEIGHT * value_score + (1 - RELATIONSHIP_VALUE_WEIGHT) * count_score

        if depth_score >= 0.80:
            depth = "Strong"
        elif depth_score >= 0.40:
            depth = "Developing"
        else:
            depth = "Thin"

        rows.append({
            "Client":           client,
            "Deals":            total,
            "Won":              wins,
            "Lost":             total - wins,
            "Win Rate %":       round(rate * 100, 1),
            "Total Value (RM)": round(total_value),
            "Depth":            depth,
        })

    df = pd.DataFrame(rows).sort_values("Win Rate %", ascending=False)
    return df.reset_index(drop=True)


def strategic_recommendation(best_driver: str) -> str:
    messages = {
        "Price": (
            "Your pricing strategy is your biggest strength. "
            "Continue bidding aggressively where price is the decisive factor."
        ),
        "Relationship & Reputation": (
            "Your network is your most valuable asset. "
            "Prioritise clients where you have established trust and prior wins."
        ),
        "Technical Capability": (
            "Technical expertise is consistently winning bids. "
            "Lead every proposal with certifications, case studies, and team credentials."
        ),
        "Delivery & Timeline": (
            "Speed and reliability are your competitive edge. "
            "Target urgent tenders and prominently showcase your logistics track record."
        ),
    }
    return messages.get(best_driver, "Invest in the strengths that differentiate you most.")
