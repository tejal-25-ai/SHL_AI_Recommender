import os
os.environ.setdefault("GROQ_API_KEY", "test_key")

from app.models.request import Message
from app.decision.guardrails import check_guardrails
from app.decision.slot_extractor import extract_slots
from app.decision.controller import decide
from app.models.conversation import Intent


class TestGuardrailEdgeCases:
    def test_java_developer_who_works_not_blocked(self):
        """'works' near 'any' should not false-trigger no-preference/off-topic logic."""
        r = check_guardrails("I need any Java developer who works well under pressure")
        assert not r.blocked

    def test_debug_my_python_code_blocked(self):
        r = check_guardrails("Can you debug my Python code?")
        assert r.blocked

    def test_product_question_not_blocked(self):
        r = check_guardrails("Does the OPQ32r assessment take long to complete?")
        assert not r.blocked

    def test_interview_questions_phrasing_variants_blocked(self):
        assert check_guardrails("What interview questions should I ask a candidate?").blocked
        assert check_guardrails("Interview questions for a Java developer role").blocked


class TestSlotExtractorEdgeCases:
    def test_no_preference_variants_all_detected(self):
        from app.core.utils import is_no_preference_response
        variants = [
            "No preference, whatever works",
            "I dont really mind either way",
            "Not sure, up to you",
            "Any of them is fine",
            "Anyone is fine",
            "No idea honestly",
            "It doesn't matter to me",
        ]
        for v in variants:
            assert is_no_preference_response(v), f"Should detect no-preference in: {v!r}"

    def test_legitimate_requirement_not_flagged_as_no_preference(self):
        from app.core.utils import is_no_preference_response
        assert not is_no_preference_response("I need any Java developer who works well under pressure")
        assert not is_no_preference_response("Mid-level, around 4 years of experience")

    def test_skill_dedup_no_partial_duplicates(self):
        slots = extract_slots([
            Message(role="user", content="Need strong stakeholder management and communication skills, also Java")
        ])
        # 'stakeholder' should not appear separately from 'stakeholder management'
        assert not ("stakeholder" in slots.skills and "stakeholder management" in slots.skills)

    def test_named_assessment_extraction_excludes_common_acronyms(self):
        slots = extract_slots([Message(role="user", content="We use SHL and AWS for our hiring, comparing OPQ32r vs GSA")])
        assert "OPQ32r" in slots.named_assessments
        assert "GSA" in slots.named_assessments
        assert "SHL" not in slots.named_assessments
        assert "AWS" not in slots.named_assessments


class TestControllerEdgeCases:
    def test_empty_messages_list_raises_at_model_level(self):
        from app.models.request import ChatRequest
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])

    def test_refine_marker_on_first_turn_does_not_misfire(self):
        """'add' appearing in a first-turn message must not be
        misclassified as REFINE — there's no prior recommendation yet."""
        r = decide([Message(role="user", content="Hiring a developer, please add Java skills to the requirement")])
        assert r.intent != Intent.REFINE

    def test_compare_without_named_assessment_falls_through(self):
        r = decide([Message(role="user", content="What is the difference between personality and aptitude tests generally?")])
        assert r.intent != Intent.COMPARE  # no concrete named assessments detected
