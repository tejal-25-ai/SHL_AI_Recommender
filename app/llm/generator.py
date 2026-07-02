"""
Generator — the orchestration layer app/api/chat.py calls after the
decision controller has decided an intent. Builds the appropriate
prompt, calls the LLM (with the client's built-in retry-once), parses
the output against the exact candidates offered, and produces a fully
validated ChatResponse via app/validation/.

Every LLM call path funnels through _call_llm_safe so a total LLM
failure (both attempts inside the client failed) degrades to a safe
clarify-style response instead of a 500 error.
"""

from app.models.request import Message
from app.models.response import ChatResponse
from app.models.conversation import Intent
from app.models.catalog import CatalogItem
from app.decision.controller import DecisionResult
from app.retrieval.retrieval_service import RetrievalService
from app.llm.client import LLMClient
from app.llm.parser import parse_llm_output, ParsedLLMOutput
from app.llm.prompts import build_clarify_prompt, build_recommend_prompt, build_compare_prompt
from app.validation.response_validator import finalize_response
from app.core.utils import full_conversation_text
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_FALLBACK_CLARIFY_REPLY = (
    "Could you tell me more about the role — for example the job title and "
    "seniority level — so I can recommend the right assessments?"
)
_FALLBACK_RECOMMEND_REPLY = "Here are some assessments that may fit your requirements."


def _call_llm_safe(
    llm_client: LLMClient, system_prompt: str, user_prompt: str, candidates: list[CatalogItem]
) -> ParsedLLMOutput:
    try:
        raw = llm_client.generate_structured(system_prompt, user_prompt)
        return parse_llm_output(raw, candidates)
    except Exception as e:  # noqa: BLE001 — final safety net, must never crash the request
        logger.error(f"LLM generation failed entirely, falling back: {e}")
        return ParsedLLMOutput(reply="", selected_items=[], end_of_conversation=False)


def generate_response(
    messages: list[Message],
    decision: DecisionResult,
    retrieval_service: RetrievalService,
    llm_client: LLMClient,
) -> ChatResponse:
    settings = get_settings()
    slots = decision.slots
    full_catalog_by_id = retrieval_service.catalog_by_id

    # --- REFUSE: no LLM call needed, guardrail already produced the message ---
    if decision.intent == Intent.REFUSE:
        return finalize_response(
            reply=decision.refusal_message or "I can't help with that request.",
            selected_items=[],
            end_of_conversation=False,
            full_catalog_by_id=full_catalog_by_id,
        )

    # --- CLARIFY ---
    if decision.intent == Intent.CLARIFY:
        system, user = build_clarify_prompt(messages, slots)
        parsed = _call_llm_safe(llm_client, system, user, candidates=[])
        return finalize_response(
            reply=parsed.reply or _FALLBACK_CLARIFY_REPLY,
            selected_items=[],
            end_of_conversation=False,
            full_catalog_by_id=full_catalog_by_id,
        )

    # --- RECOMMEND / REFINE ---
    if decision.intent in (Intent.RECOMMEND, Intent.REFINE):
        query = full_conversation_text(messages)
        candidates = retrieval_service.retrieve(
            query,
            slots,
            top_k=settings.final_top_k,
            retrieval_top_k=settings.retrieval_top_k,
        )

        if not candidates:
            # Nothing retrievable at all — degrade to clarify rather than
            # returning an empty/broken "recommendation".
            system, user = build_clarify_prompt(messages, slots)
            parsed = _call_llm_safe(llm_client, system, user, candidates=[])
            return finalize_response(
                reply=parsed.reply or _FALLBACK_CLARIFY_REPLY,
                selected_items=[],
                end_of_conversation=False,
                full_catalog_by_id=full_catalog_by_id,
            )

        system, user = build_recommend_prompt(
            messages, slots, candidates, is_refine=(decision.intent == Intent.REFINE)
        )
        parsed = _call_llm_safe(llm_client, system, user, candidates=candidates)

        selected = parsed.selected_items
        if not selected:
            # LLM produced nothing valid — fall back to top retrieved
            # candidates directly. Still fully catalog-grounded since
            # `candidates` came straight from retrieval, not the LLM.
            logger.warning("LLM selected no valid items — falling back to top retrieval results")
            selected = candidates[: settings.final_top_k]

        return finalize_response(
            reply=parsed.reply or _FALLBACK_RECOMMEND_REPLY,
            selected_items=selected,
            end_of_conversation=True,
            full_catalog_by_id=full_catalog_by_id,
        )

    # --- COMPARE ---
    if decision.intent == Intent.COMPARE:
        named_candidates = retrieval_service.get_by_names_or_ids(slots.named_assessments)
        system, user = build_compare_prompt(messages, named_candidates)
        parsed = _call_llm_safe(llm_client, system, user, candidates=named_candidates)
        return finalize_response(
            reply=parsed.reply or "I couldn't find enough information to compare those.",
            selected_items=[],  # compare responses don't populate a shortlist
            end_of_conversation=False,
            full_catalog_by_id=full_catalog_by_id,
        )

    # Should be unreachable — every Intent value is handled above.
    logger.error(f"Unhandled intent {decision.intent} — falling back to clarify")
    return finalize_response(
        reply=_FALLBACK_CLARIFY_REPLY,
        selected_items=[],
        end_of_conversation=False,
        full_catalog_by_id=full_catalog_by_id,
    )
