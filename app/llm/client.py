"""
Groq API client wrapper. Uses strict structured output (json_schema
mode with strict:true) so the model's response is constrained to match
our schema at generation time, not just validated after the fact —
this is the strongest available defense for the hard-eval schema
compliance requirement.
"""

from groq import Groq

from app.core.config import get_settings
from app.core.logging import get_logger, log_duration

logger = get_logger(__name__)

_LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "selected_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "catalog ids selected from the provided candidates, empty if none",
        },
        "end_of_conversation": {"type": "boolean"},
    },
    "required": ["reply", "selected_ids", "end_of_conversation"],
    "additionalProperties": False,
}


class LLMClient:
    def __init__(self):
        settings = get_settings()
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.llm_model

    def generate_structured(
        self, system_prompt: str, user_prompt: str, max_retries: int = 1
    ) -> dict:
        """
        Calls the LLM with strict structured output. Returns the parsed
        dict matching _LLM_RESPONSE_SCHEMA. Retries once on any failure
        (API error, malformed JSON) with a stricter reminder appended,
        before letting the exception propagate to the caller, which
        should fall back to a safe clarify response.
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            prompt = user_prompt
            if attempt > 0:
                prompt += (
                    "\n\nIMPORTANT: Your previous response did not match the "
                    "required JSON schema. Respond with ONLY valid JSON matching "
                    "the schema — no extra text, no markdown formatting."
                )

            try:
                with log_duration(logger, f"llm_call_attempt_{attempt}"):
                    completion = self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "chat_decision",
                                "strict": True,
                                "schema": _LLM_RESPONSE_SCHEMA,
                            },
                        },
                        temperature=0.2,
                    )
                import json

                content = completion.choices[0].message.content
                return json.loads(content)

            except Exception as e:  # noqa: BLE001 — intentionally broad, see retry logic above
                last_error = e
                logger.warning(f"LLM call attempt {attempt} failed: {e}")

        raise RuntimeError(f"LLM generation failed after {max_retries + 1} attempts") from last_error
