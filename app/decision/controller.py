"""
Decision controller — the single entry point app/api/chat.py calls.
Ties together guardrails + slot extraction and decides which of the
five conversational behaviors this turn should follow, per the
assignment doc: Clarify, Recommend, Refine, Compare, Refuse.

Self-contained by design: chat.py calls decide(messages) once and gets
back everything it needs (intent, slots, refusal text if any, whether
the LLM slot-extraction fallback should run) rather than orchestrating
guardrails/slot_extractor separately. Keeps the API layer thin.
"""

import re
from dataclasses import dataclass

from app.models.request import Message
from app.models.conversation import Intent, Slots
from app.core.utils import turn_count, get_last_user_message, get_user_messages
from app.core.config import get_settings
from app.decision.guardrails import check_guardrails
from app.decision.slot_extractor import extract_slots, has_stalled


@dataclass
class DecisionResult:
    intent: Intent
    slots: Slots
    turn_count: int
    refusal_message: str | None = None
    force_recommend: bool = False
    needs_llm_slot_fallback: bool = False


# --- Compare intent detection ---
_COMPARE_MARKERS_RE = re.compile(
    r"\b(?:difference between|compare|versus|\bvs\b|which is better|how (?:do|does) .+ differ)\b",
    re.IGNORECASE,
)


def _is_compare_request(text: str) -> bool:
    return bool(_COMPARE_MARKERS_RE.search(text))


# --- Refine intent detection ---
_REFINEMENT_MARKERS_RE = re.compile(
    r"\b(?:actually|instead|also add|can you add|add\b|change|update|remove|"
    r"one more thing|make it|swap|replace|drop\b|instead of)\b",
    re.IGNORECASE,
)


def _is_refinement_request(text: str) -> bool:
    return bool(_REFINEMENT_MARKERS_RE.search(text))


def _prior_recommendation_likely(messages: list[Message]) -> bool:
    """
    Heuristic: the API only gives us plain {role, content} history, not
    structured records of past ChatResponse objects, so we can't know
    for certain a shortlist was already returned. Approximate it: if at
    least 2 user turns already happened before this one, it's likely a
    recommend/clarify round already completed.
    """
    prior_user_turns = len(get_user_messages(messages)) - 1
    return prior_user_turns >= 2


def _slots_sufficient(slots: Slots) -> bool:
    """
    Minimum bar to move from clarify -> recommend: role and seniority
    both resolved (value given OR explicitly no-preference). Doc's own
    example reaches this same bar (role + seniority) before recommending.
    """
    return slots.role.is_resolved and slots.seniority.is_resolved


def decide(messages: list[Message]) -> DecisionResult:
    settings = get_settings()
    t_count = turn_count(messages)
    last_user = get_last_user_message(messages)

    # --- 1. Guardrail gate (highest priority, code-level, no LLM) ---
    if last_user is not None:
        guard = check_guardrails(last_user.content)
        if guard.blocked:
            return DecisionResult(
                intent=Intent.REFUSE,
                slots=Slots(),
                turn_count=t_count,
                refusal_message=guard.message,
            )

    # --- 2. Slot extraction ---
    slots = extract_slots(messages)
    needs_fallback = has_stalled(messages)

    # --- 3. Turn-cap forcing (hard eval: 8-turn max, must not clarify forever) ---
    force_recommend = t_count >= settings.force_recommend_after_turn

    # --- 4. Compare intent ---
    if last_user is not None and slots.named_assessments and _is_compare_request(last_user.content):
        return DecisionResult(
            intent=Intent.COMPARE,
            slots=slots,
            turn_count=t_count,
            needs_llm_slot_fallback=needs_fallback,
        )

    # --- 5. Refine intent ---
    if (
        last_user is not None
        and _is_refinement_request(last_user.content)
        and _prior_recommendation_likely(messages)
    ):
        return DecisionResult(
            intent=Intent.REFINE,
            slots=slots,
            turn_count=t_count,
            needs_llm_slot_fallback=needs_fallback,
        )

    # --- 6. Recommend (either slots are sufficient, or turn cap forces it) ---
    if _slots_sufficient(slots) or force_recommend:
        return DecisionResult(
            intent=Intent.RECOMMEND,
            slots=slots,
            turn_count=t_count,
            force_recommend=force_recommend and not _slots_sufficient(slots),
            needs_llm_slot_fallback=needs_fallback,
        )

    # --- 7. Default: clarify ---
    return DecisionResult(
        intent=Intent.CLARIFY,
        slots=slots,
        turn_count=t_count,
        needs_llm_slot_fallback=needs_fallback,
    )
