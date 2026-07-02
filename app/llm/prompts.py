"""
Prompt templates, one per intent. All prompts share the same core
grounding rule: the LLM may only select from candidates explicitly
provided in the prompt — it never invents assessment names or URLs
(those are attached by code afterward via app/validation/, using the
catalog id the LLM selects, not any text it generates).
"""

from app.models.catalog import CatalogItem
from app.models.conversation import Slots
from app.models.request import Message

_BASE_SYSTEM_PROMPT = """You are an assistant that helps recruiters and hiring managers find the \
right SHL assessments for a role, through conversation.

Rules you must always follow:
- Only ever select assessments from the CANDIDATES list provided in the user message, by their id.
- Never invent, assume, or guess an assessment name, id, or URL that is not in the candidates list.
- Stay strictly focused on SHL assessment selection.
- Base any comparison strictly on the candidate data provided — never use outside knowledge about \
what these assessments might be.
- Respond with the required JSON structure only.
"""


def _format_candidates(candidates: list[CatalogItem]) -> str:
    if not candidates:
        return "(no candidates available)"
    lines = []
    for item in candidates:
        lines.append(
            f"- id: {item.id} | name: {item.name} | test_type: {item.test_type} | "
            f"description: {item.description[:200]}"
        )
    return "\n".join(lines)


def _format_history(messages: list[Message]) -> str:
    lines = [f"{m.role}: {m.content}" for m in messages]
    return "\n".join(lines)


def build_clarify_prompt(messages: list[Message], slots: Slots) -> tuple[str, str]:
    user_prompt = f"""CONVERSATION HISTORY:
{_format_history(messages)}

CURRENT KNOWN CONSTRAINTS:
- role: {slots.role.value or '(not yet known)'}
- seniority: {slots.seniority.value or '(not yet known)'}
- skills: {', '.join(slots.skills) or '(none mentioned)'}

TASK: The user's request is not yet specific enough to recommend assessments. Ask ONE clear, \
short clarifying question to learn the missing role or seniority information. Do not recommend \
anything yet. selected_ids must be empty. end_of_conversation must be false."""
    return _BASE_SYSTEM_PROMPT, user_prompt


def build_recommend_prompt(
    messages: list[Message], slots: Slots, candidates: list[CatalogItem], is_refine: bool = False
) -> tuple[str, str]:
    action = "update the shortlist based on the new constraint" if is_refine else "recommend assessments"
    user_prompt = f"""CONVERSATION HISTORY:
{_format_history(messages)}

CURRENT KNOWN CONSTRAINTS:
- role: {slots.role.value or '(not specified)'}
- seniority: {slots.seniority.value or '(not specified)'}
- skills: {', '.join(slots.skills) or '(none mentioned)'}

CANDIDATES (select ONLY from these, by id):
{_format_candidates(candidates)}

TASK: {action}. Select between 1 and 10 candidate ids that best fit the conversation. Write a \
short reply describing what you selected and why, in plain language. end_of_conversation should \
be true since a shortlist is being delivered now."""
    return _BASE_SYSTEM_PROMPT, user_prompt


def build_compare_prompt(
    messages: list[Message], candidates: list[CatalogItem]
) -> tuple[str, str]:
    user_prompt = f"""CONVERSATION HISTORY:
{_format_history(messages)}

CANDIDATES (the assessments being compared — use ONLY this data, not outside knowledge):
{_format_candidates(candidates)}

TASK: Answer the user's comparison question using ONLY the candidate data above. If a named \
assessment the user asked about is not present in the candidates list, say clearly that it is \
not in the catalog you have access to. selected_ids should include the ids being compared (for \
reference), but this is a comparison answer, not a new recommendation — keep the same shortlist \
status from before by leaving end_of_conversation as false unless the conversation is clearly done."""
    return _BASE_SYSTEM_PROMPT, user_prompt
