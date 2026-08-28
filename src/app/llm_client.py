import hashlib
import json
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from app.config import settings


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Provider-independent LLM client.

    The application interacts with this class through generate_json().
    The underlying model is accessed through OpenRouter's
    OpenAI-compatible API.

    Responsibilities:
    - authenticate with the LLM provider
    - send prompts
    - request JSON output
    - validate output with Pydantic
    - cache successful results
    """

    def __init__(self):
        if not settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not configured.")

        if not settings.llm_model:
            raise RuntimeError("LLM_MODEL is not configured.")

        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

        self.model = settings.llm_model

        self.cache_dir = Path(
            getattr(settings, "llm_cache_dir", ".cache/llm")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> str:
        """
        Create a content-addressed cache key.

        The result depends on:
        - model
        - system prompt
        - user prompt
        - response schema
        """

        payload = {
            "model": self.model,
            "system": system_prompt,
            "user": user_prompt,
            "schema": response_model.model_json_schema(),
        }

        raw = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0,
    ) -> T:
        """
        Generate a structured response and validate it against
        the supplied Pydantic model.
        """

        cache_path = (
            self.cache_dir
            / f"{self._cache_key(system_prompt, user_prompt, response_model)}.json"
        )

        # ---------------------------------------------------------
        # 1. CACHE LOOKUP
        # ---------------------------------------------------------

        if cache_path.exists():
            cached_content = cache_path.read_text(
                encoding="utf-8"
            )

            return response_model.model_validate_json(
                cached_content
            )

        # ---------------------------------------------------------
        # 2. BUILD JSON SCHEMA
        # ---------------------------------------------------------

        schema = response_model.model_json_schema()

        # Pydantic omits fields with defaults from `required`, which
        # can allow the LLM to omit list fields entirely.
        #
        # We explicitly add every array field to `required` so the
        # structured-output contract requires those fields to appear.
        required = schema.get("required", [])

        for field_name, field_info in schema.get(
            "properties",
            {},
        ).items():
            if (
                field_info.get("type") == "array"
                and field_name not in required
            ):
                required.append(field_name)

        schema["required"] = required

        # ---------------------------------------------------------
        # 3. CALL OPENROUTER
        # ---------------------------------------------------------
        #
        # Reasoning models (e.g. glm-5.3-flash) spend part of
        # max_tokens on an internal reasoning trace before writing
        # the structured JSON. If reasoning runs long, it can
        # consume the entire budget and leave zero tokens for the
        # actual output (finish_reason="length", content=None).
        #
        # We cap reasoning tokens explicitly so content always has
        # room, and retry once with a larger budget if truncation
        # still happens (e.g. on data-heavy accounts).

        REASONING_TOKEN_CAP = 2000
        BASE_MAX_TOKENS = 8192
        RETRY_MAX_TOKENS = 16384

        def _call(max_tokens: int):
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "reasoning": {
                        "max_tokens": REASONING_TOKEN_CAP,
                    },
                },
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                    },
                },
            )

        response = _call(BASE_MAX_TOKENS)
        message = response.choices[0].message
        content = message.content
        finish_reason = response.choices[0].finish_reason

        if not content and finish_reason == "length":
            # Truncated even with the reasoning cap — retry once
            # with a larger total budget before giving up.
            response = _call(RETRY_MAX_TOKENS)
            message = response.choices[0].message
            content = message.content
            finish_reason = response.choices[0].finish_reason

        # ---------------------------------------------------------
        # 4. EXTRACT MODEL OUTPUT
        # ---------------------------------------------------------

        if not content:
            print("\n===== EMPTY LLM RESPONSE DEBUG =====")
            print(f"Model: {self.model}")
            print(f"Finish reason: {finish_reason}")
            print(f"Message: {message}")
            print(f"Full response: {response}")
            print("====================================\n")
            if finish_reason == "length":
                raise RuntimeError(
                    "LLM response was truncated before structured output "
                    "was produced, even after retrying with a larger "
                    f"token budget ({RETRY_MAX_TOKENS}). Reduce prompt "
                    "size or increase RETRY_MAX_TOKENS further."
                )
            raise RuntimeError(
                "LLM returned an empty response."
            )

        # ---------------------------------------------------------
        # 5. VALIDATE WITH PYDANTIC
        # ---------------------------------------------------------

        try:
            result = response_model.model_validate_json(
                content
            )
        except Exception as exc:
            raise RuntimeError(
                "LLM returned invalid structured output."
            ) from exc

        # ---------------------------------------------------------
        # 6. CACHE ONLY SUCCESSFUL VALIDATED RESULTS
        # ---------------------------------------------------------
        #
        # Cache writing is intentionally deferred to the caller
        # through cache_result().
        #
        # This allows application-level guardrails to reject an
        # otherwise valid Pydantic response before persistence.

        return result

    def cache_result(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        result: T,
    ) -> None:
        """
        Persist a validated result to the on-disk cache.

        Call this only after all application-level guardrails have
        passed so that invalid results are never permanently cached.
        """

        cache_path = (
            self.cache_dir
            / f"{self._cache_key(system_prompt, user_prompt, response_model)}.json"
        )

        cache_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )