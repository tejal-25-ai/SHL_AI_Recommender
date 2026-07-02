"""
Shared helpers with no business logic of their own — pure utility
functions used by multiple modules (app/api/, app/decision/,
app/retrieval/). Anything that starts appearing in more than one
module should land here rather than being duplicated.
"""

import re

from app.models.request import Message

# Phrases the simulated evaluator user is documented to use when it has
# no fact to answer with ("says it has no preference when asked
# something outside its facts"). Used by the slot extractor to close a
# slot instead of re-asking indefinitely.
_NO_PREFERENCE_PATTERNS = [
    r"\bno preference\b",
    r"\bno particular preference\b",
    r"\bdon'?t (?:really )?(?:have a|mind|care)\b",
    r"\bnot sure\b",
    r"\bno idea\b",
    r"\bdoesn'?t matter\b",
    r"\bwhatever (?:works|you (?:think|recommend))\b",
    # "any(one/thing/of them) [is] fine/okay/good" — deliberately narrow:
    # requires "any" to be immediately followed by a closing phrase
    # (optionally through "of them/those/it" and/or "is"), NOT just
    # loosely near the word "fine"/"works" anywhere in the sentence.
    # This avoids false-triggering on real requirements like
    # "any Java developer who works well".
    r"\bany(?:one|thing)?\b(?:\s+of\s+(?:them|those|it))?\s+(?:is\s+)?(?:fine|okay|ok|good)\b",
]
_NO_PREFERENCE_RE = re.compile("|".join(_NO_PREFERENCE_PATTERNS), re.IGNORECASE)


def turn_count(messages: list[Message]) -> int:
    """
    Total turns in the conversation so far, per the assignment's
    definition: "each conversation at 8 turns including user & assistant".
    This is simply len(messages) since the API is stateless and the
    full history is always provided.
    """
    return len(messages)


def get_user_messages(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m.role == "user"]


def get_last_user_message(messages: list[Message]) -> Message | None:
    user_msgs = get_user_messages(messages)
    return user_msgs[-1] if user_msgs else None


def full_conversation_text(messages: list[Message]) -> str:
    """
    Flatten the whole conversation into one text blob, newest-agnostic.
    Used as the retrieval query source so context accumulated across
    turns (not just the latest message) informs retrieval — required
    by the doc's "Rich Conversation Parsing" expectation.
    """
    return " ".join(m.content.strip() for m in messages if m.content.strip())


def is_no_preference_response(text: str) -> bool:
    """
    Heuristic check for the simulated user's "no preference" answer
    pattern. Used by slot_extractor to mark a slot as resolved-but-empty
    (see ConversationState.Slots / SlotValue.is_resolved) instead of
    leaving it open and causing the agent to re-ask forever.
    """
    return bool(_NO_PREFERENCE_RE.search(text))


def normalize_for_match(text: str) -> str:
    """
    Lowercase + strip punctuation/whitespace variance, used when
    comparing an LLM-generated assessment name/url against the catalog
    for validation (app/validation/catalog_validator.py) — avoids
    false-negative rejections over trivial formatting differences.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
