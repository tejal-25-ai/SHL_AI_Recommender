"""
Schema validator — wraps ChatResponse construction so a malformed
input (e.g. an unexpected test_type slipping through, or too many
items) degrades to a safe fallback response instead of raising an
unhandled exception mid-request.
"""

from pydantic import ValidationError

from app.models.response import ChatResponse, Recommendation
from app.core.constants import MAX_RECOMMENDATIONS
from app.core.logging import get_logger

logger = get_logger(__name__)

_SAFE_FALLBACK_REPLY = (
    "I ran into an issue putting together a response. Could you tell me a bit "
    "more about the role you're hiring for?"
)


def build_validated_response(
    reply: str, recommendations: list[Recommendation], end_of_conversation: bool
) -> ChatResponse:
    """
    Attempts to construct a valid ChatResponse. If validation fails
    even after truncating recommendations to MAX_RECOMMENDATIONS, falls
    back to a safe empty-recommendations clarify-style response rather
    than propagating a 500 error — a degraded but schema-valid response
    is always better than a crashed request against the hard-eval gate.
    """
    try:
        return ChatResponse(
            reply=reply,
            recommendations=recommendations[:MAX_RECOMMENDATIONS],
            end_of_conversation=end_of_conversation,
        )
    except ValidationError as e:
        logger.error(f"ChatResponse validation failed, falling back to safe response: {e}")
        try:
            return ChatResponse(
                reply=reply or _SAFE_FALLBACK_REPLY,
                recommendations=[],
                end_of_conversation=False,
            )
        except ValidationError:
            # reply itself was somehow invalid (e.g. wrong type) — last resort
            return ChatResponse(
                reply=_SAFE_FALLBACK_REPLY, recommendations=[], end_of_conversation=False
            )
