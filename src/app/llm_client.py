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
            timeout=getattr(settings, "llm_timeout_seconds", 60.0),
        )

        self.model = settings.llm_model
        self.fallback_model = getattr(settings, "llm_fallback_model", None)

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
        # 3. CALL PROVIDER
        # ---------------------------------------------------------
        #
        # Nemotron 3 Ultra (and similar frontier reasoning models)
        # handle their own internal reasoning traces without needing
        # an explicit cap parameter.  We give a generous max_tokens
        # budget so large health-agent prompts (90-day ticket
        # history) have enough room for both reasoning + output.

        BASE_MAX_TOKENS = 4000
        MAX_ATTEMPTS = 3
        RETRY_DELAYS = (1, 2)

        def _call_provider(max_tokens: int, response_format: dict, model: str):
            kwargs = {
                "model": model,
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
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            response = self.client.chat.completions.create(**kwargs)
            if response is None:
                raise RuntimeError("LLM provider returned a null response object.")
            if not getattr(response, "choices", None):
                raise RuntimeError("LLM provider returned an empty choices list.")
            return response

        # ---------------------------------------------------------
        # 4. EXECUTE ONE BOUNDED RETRY LOOP
        # ---------------------------------------------------------

        last_content = ""
        last_finish_reason = None
        last_validation_exc = None

        resp_fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
            },
        }

        for attempt in range(MAX_ATTEMPTS):
            max_tokens = BASE_MAX_TOKENS
            model = (
                self.fallback_model
                if attempt > 0 and self.fallback_model
                else self.model
            )

            try:
                response = _call_provider(max_tokens, resp_fmt, model)
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                error_text = str(exc)
                is_credit_error = (
                    status_code == 402
                    or "402" in error_text
                    or "fewer max_tokens" in error_text
                )
                if is_credit_error:
                    raise RuntimeError(
                        "LLM provider rejected the request because the account "
                        "does not have enough credits. Add credits or lower the "
                        "configured token budget."
                    ) from exc
                model_unavailable = (
                    status_code == 404
                    and (
                        self.fallback_model
                        or "provider" in error_text.lower()
                        or "model" in error_text.lower()
                        or "unavailable" in error_text.lower()
                    )
                )
                if (
                    not self._is_retryable_error(exc)
                    and not isinstance(exc, RuntimeError)
                    and not model_unavailable
                ):
                    raise
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                time.sleep(RETRY_DELAYS[attempt])
                continue

            if not getattr(response, "choices", None):
                if attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError("LLM provider returned an empty choices list.")
                time.sleep(RETRY_DELAYS[attempt])
                continue

            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is None:
                if attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError("LLM provider returned a choice without a message payload.")
                time.sleep(RETRY_DELAYS[attempt])
                continue

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
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[attempt])
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
                return result
            except Exception as exc:
                last_validation_exc = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAYS[attempt])
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
        print(f"Content: {last_content.encode('ascii', errors='backslashreplace').decode('ascii')}")
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