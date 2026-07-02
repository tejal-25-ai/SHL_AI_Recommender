"""
Fixed constants — values that are correct by definition (SHL's own
taxonomy) or contractually required by the assignment schema, so they
must NOT be tunable via environment variables like app/core/config.py's
settings are.
"""

# SHL test type letter codes -> human-readable label.
# Source: assignment doc + SHL catalog convention.
TEST_TYPE_LABELS: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}
VALID_TEST_TYPES: frozenset[str] = frozenset(TEST_TYPE_LABELS.keys())

# Individual Test Solutions vs Job Solutions — the scraper must only
# ever ingest the former, per the assignment's explicit scope limit.
CATALOG_TYPE_INDIVIDUAL = 1
CATALOG_TYPE_JOB_SOLUTION = 2

# Recommendation count bounds — hard-eval requirement, not tunable.
MIN_RECOMMENDATIONS = 1
MAX_RECOMMENDATIONS = 10

# Decision controller intents (mirrors app.models.conversation.Intent —
# kept here too as plain strings for places that need string comparison
# without importing the enum, e.g. logging/metrics).
INTENT_CLARIFY = "clarify"
INTENT_RECOMMEND = "recommend"
INTENT_REFINE = "refine"
INTENT_COMPARE = "compare"
INTENT_REFUSE = "refuse"

# Refusal message categories — used by app/decision/guardrails.py so
# refusal copy is centralized and consistent, not scattered across
# the codebase.
REFUSAL_OFF_TOPIC = (
    "I can only help with SHL assessment selection. I'm not able to "
    "assist with that request."
)
REFUSAL_LEGAL_HIRING_ADVICE = (
    "I can help you find the right SHL assessments, but I can't provide "
    "general hiring or legal advice. Would you like assessment "
    "recommendations instead?"
)
REFUSAL_GENERAL_HIRING_ADVICE = (
    "I'm focused on SHL assessment selection rather than broader hiring "
    "process advice. I'd be glad to help you find assessments for this "
    "role instead."
)
REFUSAL_INJECTION = (
    "I can only help with SHL assessment selection and can't follow "
    "instructions that change that scope."
)
