"""
Guardrail gate — the FIRST thing that runs on every /chat call, before
slot extraction or retrieval (see locked architecture). Pure code/regex,
no LLM call, so it's cheap and can't itself be talked out of refusing
by a prompt-injection attempt embedded in the user message.

This is deliberately high-precision, not high-recall: it only fires on
strong, unambiguous signals, and off-topic detection is suppressed
whenever an in-scope anchor keyword (assessment, hire, role, etc.) is
also present. Genuinely ambiguous cases that slip past this gate are a
second line of defense handled by explicit scope instructions in the
LLM prompt (app/llm/prompts.py) — a code-level catch-all that's too
aggressive would risk false-positive refusals on legitimate assessment
queries, which is a worse failure mode for Recall@10 than occasionally
letting a borderline case reach the LLM layer.
"""

import re
from dataclasses import dataclass
from enum import Enum

from app.core.constants import (
    REFUSAL_OFF_TOPIC,
    REFUSAL_LEGAL_HIRING_ADVICE,
    REFUSAL_GENERAL_HIRING_ADVICE,
    REFUSAL_INJECTION,
)


class GuardrailCategory(str, Enum):
    INJECTION = "injection"
    LEGAL_ADVICE = "legal_advice"
    GENERAL_HIRING_ADVICE = "general_hiring_advice"
    OFF_TOPIC = "off_topic"


@dataclass
class GuardrailResult:
    blocked: bool
    category: GuardrailCategory | None = None
    message: str | None = None


# --- Prompt injection patterns ---
# Targets attempts to override system behavior, not legitimate
# conversation content. Kept intentionally strict/explicit rather than
# broad, since false positives here block a real user's real turn.
_INJECTION_PATTERNS = [
    r"\bignore (?:the )?(?:previous|above|prior|all) instructions?\b",
    r"\bdisregard (?:the )?(?:previous|above|prior|all) instructions?\b",
    r"\bforget (?:your|the|all) (?:instructions|rules|prompt)\b",
    r"\byou are now\b",
    r"\bact as (?:if you are |a )?(?!an? .*assessment)",  # "act as X" but allow "act as an assessment advisor"-type phrasing
    r"\bpretend (?:you are|to be)\b",
    r"\bnew instructions?:",
    r"\breveal (?:your|the) (?:system )?prompt\b",
    r"\bwhat (?:is|are) your (?:system )?(?:instructions|prompt)\b",
    r"\boverride (?:your|the) (?:instructions|rules|behavior)\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# --- Legal-advice patterns ---
# Narrow: explicit legal-compliance questions about hiring practices,
# not general assessment-selection questions that happen to mention a
# protected-characteristic-adjacent word.
_LEGAL_ADVICE_PATTERNS = [
    r"\bis it legal (?:to|for)\b",
    r"\blegal(?:ly)? (?:allowed|permitted|risk)\b",
    r"\bcan i (?:legally )?ask (?:about|for)\b.*\b(?:age|disability|pregnan|religion|marital|citizenship)\b",
    r"\bdiscriminat(?:e|ion|ory)\b",
    r"\beeoc\b",
    r"\bgdpr\b",
    r"\blawsuit\b",
    r"\bsue (?:my|the|our) (?:company|employer)\b",
    r"\bemployment law\b",
]
_LEGAL_ADVICE_RE = re.compile("|".join(_LEGAL_ADVICE_PATTERNS), re.IGNORECASE)

# --- General hiring-process advice patterns ---
# Explicitly NOT about assessment selection — job descriptions, comp,
# interview structure, onboarding, etc. Kept narrow to avoid catching
# "what assessment should I use for a Java interview" style queries.
_GENERAL_HIRING_PATTERNS = [
    r"\bwrite (?:me )?a job description\b",
    r"\bhow (?:much|do i) (?:should i )?pay\b",
    r"\bsalary (?:negotiation|range|benchmark)\b",
    r"\bhow (?:do i|to) structure an? interview\b",
    r"\binterview questions?\b",
    r"\bwhat questions (?:should i|to) ask\b",
    r"\bonboarding (?:process|plan|checklist)\b",
    r"\bperformance review template\b",
    r"\bhow (?:do i|to) fire\b",
    r"\btermination (?:letter|process)\b",
]
_GENERAL_HIRING_RE = re.compile("|".join(_GENERAL_HIRING_PATTERNS), re.IGNORECASE)

# --- Off-topic patterns ---
# Explicit signals for clearly unrelated categories (weather, recipes,
# trivia, creative writing, unrelated coding help, entertainment,
# personal advice). Deliberately narrow, high-precision phrases — this
# is NOT a general "is this about hiring" classifier.
_OFF_TOPIC_PATTERNS = [
    r"\bweather\b",
    r"\bforecast\b",
    r"\brecipe\b",
    r"\bhow (?:do i|to) cook\b",
    r"\bcapital of\b",
    r"\bwho (?:is|was) the (?:president|prime minister|king|queen)\b",
    r"\bwrite (?:me |a )?(?:a )?(?:poem|song|story|lyrics)\b",
    r"\bsolve (?:this|the) equation\b",
    r"\brelationship advice\b",
    r"\bshould i break up\b",
    r"\bmovie recommendations?\b",
    r"\bwhat should i watch\b",
    r"\btell me a joke\b",
    r"\bwrite (?:me )?(?:a |some )?(?:python|javascript|java|c\+\+) (?:script|code|function)\b",
    r"\bdebug my (?:\w+ )?code\b",
    r"\btranslate this\b",
]
_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)

# Anchor keywords that indicate the message IS about SHL assessment
# selection, even if it happens to also match an off-topic pattern
# above (e.g. "any of these tests could work for a Java role" — "any"
# is not itself an off-topic trigger, but this anchor list exists as
# a safety net for edge phrasing). If present, the off-topic check is
# suppressed — false-positive refusals on in-scope queries are worse
# for Recall@10 than occasionally letting a borderline case through
# to the LLM layer, which also carries scope instructions.
_IN_SCOPE_ANCHOR_PATTERNS = [
    r"\bassessment\b", r"\btest\b", r"\bshl\b", r"\bhir(?:e|ing)\b",
    r"\bcandidate\b", r"\brole\b", r"\bjob\b", r"\bposition\b",
    r"\brecruit\b", r"\bskill\b", r"\bseniority\b", r"\bexperience\b",
    r"\bpersonality\b", r"\baptitude\b", r"\bcognitive\b", r"\bcompetenc\b",
    r"\bopq\b", r"\bgsa\b", r"\bshortlist\b",
]
_IN_SCOPE_ANCHOR_RE = re.compile("|".join(_IN_SCOPE_ANCHOR_PATTERNS), re.IGNORECASE)


def _normalize(text: str) -> str:
    """
    Collapse all whitespace (including embedded newlines used to try to
    break up a regex match, e.g. "ignore\\n\\nprevious\\n\\ninstructions")
    into single spaces before running any pattern checks.
    """
    return " ".join(text.split())


def check_guardrails(text: str) -> GuardrailResult:
    """
    Run all guardrail checks against a single message (typically the
    latest user message). Returns the FIRST category matched, checked
    in order: injection > legal advice > general hiring advice > off-topic.
    """
    text = _normalize(text)

    if _INJECTION_RE.search(text):
        return GuardrailResult(
            blocked=True, category=GuardrailCategory.INJECTION, message=REFUSAL_INJECTION
        )

    if _LEGAL_ADVICE_RE.search(text):
        return GuardrailResult(
            blocked=True,
            category=GuardrailCategory.LEGAL_ADVICE,
            message=REFUSAL_LEGAL_HIRING_ADVICE,
        )

    if _GENERAL_HIRING_RE.search(text):
        return GuardrailResult(
            blocked=True,
            category=GuardrailCategory.GENERAL_HIRING_ADVICE,
            message=REFUSAL_GENERAL_HIRING_ADVICE,
        )

    if _OFF_TOPIC_RE.search(text) and not _IN_SCOPE_ANCHOR_RE.search(text):
        return GuardrailResult(
            blocked=True, category=GuardrailCategory.OFF_TOPIC, message=REFUSAL_OFF_TOPIC
        )

    return GuardrailResult(blocked=False)
