"""
Rule-based conversation slot extractor.

Scans the FULL conversation history (not just the latest message) to
build a structured requirement profile — required by the doc's "Rich
Conversation Parsing" expectation and necessary because the simulated
user "may volunteer information out of order."

This is the primary, fast path. A separate LLM-fallback path (wired in
app/decision/controller.py) is only triggered when `has_stalled()`
reports no new slot was resolved after the last exchange — keeping the
common case cheap and reserving the LLM call for genuinely ambiguous
phrasing.
"""

import re

from app.models.request import Message
from app.models.conversation import Slots, SlotValue
from app.core.utils import get_user_messages, is_no_preference_response

# --- Seniority detection ---
_SENIORITY_KEYWORDS = {
    "entry": ["entry level", "entry-level", "graduate", "fresher", "intern", "trainee"],
    "junior": ["junior"],
    "mid": ["mid level", "mid-level", "intermediate", "mid senior"],
    "senior": ["senior", "lead", "principal", "staff"],
    "manager": ["manager", "management", "supervisor"],
    "director": ["director", "vp", "vice president", "head of", "executive", "c-level", "chief"],
}
_SENIORITY_RE = {
    label: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for label, kws in _SENIORITY_KEYWORDS.items()
}
_YEARS_EXPERIENCE_RE = re.compile(r"\b(\d{1,2})\s*(?:\+)?\s*years?\b", re.IGNORECASE)


def _seniority_from_years(years: int) -> str:
    if years <= 2:
        return "entry"
    if years <= 6:
        return "mid"
    return "senior"


def _extract_seniority(text: str) -> str | None:
    for label, pattern in _SENIORITY_RE.items():
        if pattern.search(text):
            return label
    years_match = _YEARS_EXPERIENCE_RE.search(text)
    if years_match:
        return _seniority_from_years(int(years_match.group(1)))
    return None


# --- Role detection ---
# Captures the noun phrase following a hiring-intent verb, e.g.
# "hiring a Java developer" -> "Java developer".
_ROLE_INTENT_RE = re.compile(
    r"\b(?:hiring|looking for|need|need to hire|recruiting)\s+(?:an?\s+)?([a-zA-Z][\w\s\-]{2,40}?)"
    r"(?=[.,!?]|\s+(?:who|with|for|that)\b|$)",
    re.IGNORECASE,
)


def _extract_role(text: str) -> str | None:
    match = _ROLE_INTENT_RE.search(text)
    if match:
        role = match.group(1).strip()
        # guard against over-capturing filler like "someone"
        if role.lower() not in {"someone", "somebody", "a person", "people"}:
            return role
    return None


# --- Skills detection ---
# Fixed vocabulary rather than free-form NLP extraction — deliberately
# conservative so we never invent a skill the user didn't state
# (avoids the query-expansion hallucination risk we designed against).
_SKILL_VOCAB = [
    "java", "python", "javascript", "typescript", "sql", "c++", "c#", ".net",
    "react", "angular", "node", "spring", "microservices", "rest api", "api",
    "aws", "azure", "gcp", "cloud", "devops", "kubernetes", "docker",
    "stakeholder management", "stakeholder", "communication", "leadership",
    "negotiation", "sales", "customer service", "project management",
    "data analysis", "excel", "sql server", "problem solving", "teamwork",
    "collaboration", "coaching", "mentoring", "presentation",
]


def _extract_skills(text: str) -> list[str]:
    text_lower = text.lower()
    found = [skill for skill in _SKILL_VOCAB if skill in text_lower]
    # dedup while preserving order, e.g. "stakeholder" inside "stakeholder management"
    deduped: list[str] = []
    for s in found:
        if not any(s != other and s in other for other in found):
            if s not in deduped:
                deduped.append(s)
    return deduped


# --- Test type preference detection ---
_TEST_TYPE_KEYWORDS = {
    "P": ["personality", "behavioral", "behaviour"],
    "K": ["knowledge test", "skills test", "technical test", "coding test"],
    "A": ["aptitude", "cognitive", "ability test", "reasoning"],
    "S": ["simulation", "situational judgement", "sjt"],
    "C": ["competency", "competencies"],
    "B": ["biodata"],
}
_TEST_TYPE_RE = {
    code: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.IGNORECASE)
    for code, kws in _TEST_TYPE_KEYWORDS.items()
}


def _extract_test_type_pref(text: str) -> str | None:
    for code, pattern in _TEST_TYPE_RE.items():
        if pattern.search(text):
            return code
    return None


# --- Named assessment detection (for compare intent) ---
# Coarse candidate extractor only — catches product-code-shaped tokens
# (e.g. "OPQ32r", "GSA"). Real validation against the actual catalog
# happens downstream in app/validation/catalog_validator.py; this just
# flags candidates worth checking.
_ASSESSMENT_CODE_RE = re.compile(r"\b[A-Z]{2,6}\d{0,3}[a-zA-Z]{0,2}\b")
_COMMON_ACRONYM_FALSE_POSITIVES = {"SHL", "USA", "API", "SQL", "AWS", "GCP", "HR"}


def _extract_named_assessments(text: str) -> list[str]:
    candidates = _ASSESSMENT_CODE_RE.findall(text)
    return [c for c in candidates if c.upper() not in _COMMON_ACRONYM_FALSE_POSITIVES]


# --- Assistant-question -> slot inference (for no-preference handling) ---
_QUESTION_SLOT_HINTS = {
    "seniority": ["seniority", "experience level", "years of experience", "how senior", "level"],
    "role": ["role", "position", "job title", "which role", "what role"],
    "test_type_pref": ["test type", "type of assessment", "kind of assessment", "which type"],
}


def _infer_asked_slot(assistant_text: str) -> str | None:
    text_lower = assistant_text.lower()
    for slot_name, hints in _QUESTION_SLOT_HINTS.items():
        if any(hint in text_lower for hint in hints):
            return slot_name
    return None


def extract_slots(messages: list[Message]) -> Slots:
    """
    Build a Slots object from the full conversation history. Re-run
    from scratch on every call (stateless service — no incremental
    state carried between requests).
    """
    user_text = " ".join(m.content for m in get_user_messages(messages))

    slots = Slots()

    role = _extract_role(user_text)
    if role:
        slots.role = SlotValue(value=role)

    seniority = _extract_seniority(user_text)
    if seniority:
        slots.seniority = SlotValue(value=seniority)

    slots.skills = _extract_skills(user_text)

    test_type = _extract_test_type_pref(user_text)
    if test_type:
        slots.test_type_pref = SlotValue(value=test_type)

    slots.named_assessments = _extract_named_assessments(user_text)

    # --- "no preference" closing ---
    # If the latest user message is a no-preference response and we can
    # infer which slot the preceding assistant question targeted, close
    # that slot rather than leaving it open (prevents infinite re-asking
    # against the simulated evaluator user).
    if len(messages) >= 2 and messages[-1].role == "user":
        last_user_msg = messages[-1]
        prev_assistant_msg = messages[-2] if messages[-2].role == "assistant" else None

        if is_no_preference_response(last_user_msg.content) and prev_assistant_msg:
            asked_slot = _infer_asked_slot(prev_assistant_msg.content)
            if asked_slot == "role" and not slots.role.is_resolved:
                slots.role = SlotValue(no_preference=True)
            elif asked_slot == "seniority" and not slots.seniority.is_resolved:
                slots.seniority = SlotValue(no_preference=True)
            elif asked_slot == "test_type_pref" and not slots.test_type_pref.is_resolved:
                slots.test_type_pref = SlotValue(no_preference=True)

    return slots


def _resolved_count(slots: Slots) -> int:
    count = 0
    if slots.role.is_resolved:
        count += 1
    if slots.seniority.is_resolved:
        count += 1
    if slots.test_type_pref.is_resolved:
        count += 1
    if slots.skills:
        count += 1
    return count


def has_stalled(messages: list[Message]) -> bool:
    """
    True if the most recent exchange (last assistant question + last
    user answer) resolved no additional slot compared to before it.
    Used by the decision controller to trigger the LLM-based extraction
    fallback only when the cheap rule-based pass is genuinely stuck —
    not on every turn.
    """
    if len(messages) < 4:
        # Need at least 2 full exchanges to compare "before" vs "after".
        return False

    current = extract_slots(messages)
    prior = extract_slots(messages[:-2])

    return _resolved_count(current) <= _resolved_count(prior)
