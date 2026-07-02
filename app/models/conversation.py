"""
Internal state models. None of this is persisted — it's rebuilt from
scratch from the full `messages` history on every /chat call, since
the service is stateless.
"""

from enum import Enum
from pydantic import BaseModel, Field


class Intent(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"


class SlotValue(BaseModel):
    """
    A single extracted constraint (e.g. seniority, role, test_type_pref).

    `no_preference=True` is a distinct state from `value=None`:
    - value=None, no_preference=False  -> not yet asked / not yet answered
    - value=None, no_preference=True   -> user was asked and explicitly
      said they have no preference; this slot is CLOSED and must not be
      re-asked (prevents infinite clarify loops against the simulated
      user, who answers "no preference" for anything outside its facts).
    """
    value: str | None = None
    no_preference: bool = False

    @property
    def is_resolved(self) -> bool:
        """True if this slot no longer needs to be asked about."""
        return self.value is not None or self.no_preference


class Slots(BaseModel):
    role: SlotValue = Field(default_factory=SlotValue)
    seniority: SlotValue = Field(default_factory=SlotValue)
    skills: list[str] = Field(default_factory=list)
    test_type_pref: SlotValue = Field(default_factory=SlotValue)
    named_assessments: list[str] = Field(default_factory=list)  # for compare intent


class ConversationState(BaseModel):
    """
    Fully derived, in-memory-only state for the current turn.
    Built by app/decision/slot_extractor.py from the raw message history.
    """
    turn_count: int
    slots: Slots = Field(default_factory=Slots)
    intent: Intent | None = None
    prior_recommendation_given: bool = False
    # candidate pool from the previous turn, kept only for this request's
    # lifetime, to support REFINE without re-running retrieval from scratch
    prior_candidate_ids: list[str] = Field(default_factory=list)
