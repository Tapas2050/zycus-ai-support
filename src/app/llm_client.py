import hashlib
import json
import time
from pathlib import Path
from typing import TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from app.config import settings


T = TypeVar("T", bound=BaseModel)


def _normalize_json_schema(schema: dict) -> dict:
    """Ensure required fields are explicit for array items and nested objects.

    OpenAI's structured-output JSON schema is stricter about nested required
    fields for arrays of objects. A raw Pydantic schema often leaves array-item
    required lists inside referenced $defs instead of the concrete `items`
    schema, which can lead the model to omit keys like `severity` or
    `ticket_id` from nested objects.
    """
    schema = json.loads(json.dumps(schema))
    defs = schema.get("$defs", {})

    def resolve_ref(ref: str) -> dict:
        if not ref.startswith("#/$defs/"):
            return {}
        name = ref.split("/")[-1]
        return defs.get(name, {})

    def walk(node: dict):
        if not isinstance(node, dict):
            return

        if "$ref" in node:
            ref_schema = resolve_ref(node["$ref"])
            if ref_schema:
                node = {**ref_schema, **{k: v for k, v in node.items() if k != "$ref"}}
                # update the caller since we merged the referenced schema in place
                if "properties" in node:
                    pass

        properties = node.get("properties", {})
        required = list(node.get("required", []))

        for name, prop in properties.items():
            if isinstance(prop, dict):
                if prop.get("type") == "array":
                    if name not in required:
                        required.append(name)

                    items = prop.get("items")
                    if isinstance(items, dict) and "$ref" in items:
                        ref_schema = resolve_ref(items["$ref"])
                        if ref_schema:
                            items = {**ref_schema, **{k: v for k, v in items.items() if k != "$ref"}}
                            prop["items"] = items

                    if isinstance(prop.get("items"), dict):
                        walk(prop["items"])

                if prop.get("type") == "object":
                    walk(prop)

                if "$ref" in prop:
                    ref_schema = resolve_ref(prop["$ref"])
                    if ref_schema:
                        prop.clear()
                        prop.update(ref_schema)
                        walk(prop)

        if node.get("type") == "object" or "properties" in node:
            all_props = list(node.get("properties", {}).keys())
            node["required"] = list(dict.fromkeys(required + all_props))
        elif "required" in node:
            del node["required"]

        if isinstance(node.get("items"), dict):
            walk(node["items"])

        for child_key in ("allOf", "anyOf", "oneOf"):
            for child in node.get(child_key, []):
                if isinstance(child, dict):
                    walk(child)

    walk(schema)
    for def_schema in defs.values():
        walk(def_schema)

    return schema


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

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Return True for transient provider/network failures."""
        retryable_status_codes = {408, 409, 429, 500, 502, 503, 504}

        if isinstance(
            exc,
            (APIConnectionError, APITimeoutError, RateLimitError),
        ):
            return True

        if isinstance(exc, APIStatusError):
            return exc.status_code in retryable_status_codes

        status_code = getattr(exc, "status_code", None)
        if status_code in retryable_status_codes:
            return True

        return False

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

        schema = _normalize_json_schema(response_model.model_json_schema())

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

        REASONING_TOKEN_CAP = 150
        BASE_MAX_TOKENS = 2500
        RETRY_MAX_TOKENS = 3000
        MAX_TRANSIENT_RETRIES = 3

        def _call_provider(max_tokens: int, response_format: dict):
            last_err = None
            current_max_tokens = max_tokens
            for attempt in range(MAX_TRANSIENT_RETRIES + 1):
                try:
                    kwargs = {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        "temperature": temperature,
                        "max_tokens": current_max_tokens,
                        "response_format": response_format,
                        "extra_body": {"reasoning": {"max_tokens": REASONING_TOKEN_CAP}},
                    }
                    response = self.client.chat.completions.create(**kwargs)
                    if response is None:
                        raise RuntimeError("LLM provider returned a null response object.")
                    if not getattr(response, "choices", None):
                        raise RuntimeError("LLM provider returned an empty choices list.")
                    return response
                except Exception as exc:  # pragma: no cover - exercised via live API
                    last_err = exc
                    # Handle OpenRouter 402 credit limit by clamping to affordable tokens
                    err_str = str(exc)
                    if getattr(exc, "status_code", None) == 402 or "402" in err_str or "fewer max_tokens" in err_str:
                        import re
                        afford_match = re.search(r"can only afford (\d+)", err_str)
                        if afford_match:
                            current_max_tokens = max(500, int(afford_match.group(1)) - 50)
                        else:
                            current_max_tokens = min(current_max_tokens, 1500)
                        time.sleep(1)
                        continue
                    if not self._is_retryable_error(exc) or attempt >= MAX_TRANSIENT_RETRIES:
                        raise
                    time.sleep((attempt + 1) * 1)
            raise last_err

        # ---------------------------------------------------------
        # 4. EXECUTE MODEL CALL (WITH ROBUST RETRY & FORMAT FALLBACK)
        # ---------------------------------------------------------

        last_content = ""
        last_finish_reason = None
        last_validation_exc = None

        for attempt in range(3):
            max_tokens = BASE_MAX_TOKENS if attempt == 0 else RETRY_MAX_TOKENS
            # Attempt 0 uses strict json_schema; Attempts 1-2 fall back to json_object
            if attempt == 0:
                resp_fmt = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                    },
                }
            else:
                resp_fmt = {"type": "json_object"}

            try:
                response = _call_provider(max_tokens, resp_fmt)
            except Exception as exc:
                if attempt == 2:
                    raise
                time.sleep(1)
                continue

            if not getattr(response, "choices", None):
                raise RuntimeError("LLM provider returned an empty choices list.")

            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                raise RuntimeError("LLM provider returned a choice without a message payload.")

            content = message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            last_content = content
            last_finish_reason = finish_reason

            if not content:
                # If provider returned reasoning in place of content, check for embedded JSON
                reasoning = getattr(message, "reasoning", None) or ""
                if "{" in reasoning and "}" in reasoning:
                    import re
                    match = re.search(r"(\{.*\})", reasoning, re.DOTALL)
                    if match:
                        content = match.group(1)
                        last_content = content

            if not content:
                time.sleep(1)
                continue

            # Strip markdown code blocks or surrounding prose if present
            clean_content = content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            elif clean_content.startswith("```"):
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()
            if not (clean_content.startswith("{") and clean_content.endswith("}")):
                import re
                match = re.search(r"(\{.*\})", clean_content, re.DOTALL)
                if match:
                    clean_content = match.group(1).strip()

            try:
                result = response_model.model_validate_json(clean_content)
                cache_path.write_text(clean_content, encoding="utf-8")
                return result
            except Exception as exc:
                last_validation_exc = exc
                time.sleep(1)
                continue

        # ---------------------------------------------------------
        # 5. DIAGNOSTICS IF ALL ATTEMPTS FAILED
        # ---------------------------------------------------------

        if not last_content:
            print("\n===== EMPTY LLM RESPONSE DEBUG =====")
            print(f"Model: {self.model}")
            print(f"Finish reason: {last_finish_reason}")
            print("====================================\n")
            raise RuntimeError("LLM returned an empty response.")

        print("\n===== INVALID JSON DEBUG =====")
        print(f"Model: {self.model}")
        print(f"Finish reason: {last_finish_reason}")
        print(f"Content: {repr(last_content)}")
        print("==============================\n")
        raise RuntimeError("LLM returned invalid structured output.") from last_validation_exc

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