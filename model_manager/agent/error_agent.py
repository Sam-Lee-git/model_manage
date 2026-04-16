"""ErrorDiagnosisAgent — diagnoses installation failures via any configured LLM."""

from __future__ import annotations

import json
import re
from typing import Optional

from model_manager.agent.base import LLMClient
from model_manager.core.exceptions import DiagnosisFailedError
from model_manager.core.events import DiagnosisStartedEvent, DiagnosisReadyEvent, bus
from model_manager.recovery.branch import DiagnosisResult, UserInputRequest
from model_manager.recovery.context import ErrorContext


import dataclasses


def _context_to_message(ctx: ErrorContext) -> str:
    d = dataclasses.asdict(ctx)
    # Trim pip_packages to ML-relevant subset to avoid token bloat
    pkgs = d.get("environment", {}).get("pip_packages", {})
    ml_keywords = ("torch", "transformers", "cuda", "nvidia", "pip", "setuptools",
                   "wheel", "accelerate", "bitsandbytes", "peft", "triton",
                   "llama", "gguf", "ctransformers", "huggingface")
    d["environment"]["pip_packages"] = {
        k: v for k, v in pkgs.items()
        if any(kw in k.lower() for kw in ml_keywords)
    }
    return json.dumps(d, default=str, indent=2)


def _extract_json(text: str) -> Optional[str]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1)
    return None


class ErrorDiagnosisAgent:
    def __init__(self, client: Optional[LLMClient] = None) -> None:
        if client is None:
            from model_manager.agent.factory import create_client_from_settings
            client = create_client_from_settings()
        self._client = client

    async def diagnose(self, ctx: ErrorContext) -> DiagnosisResult:
        await bus.emit(DiagnosisStartedEvent(step_name=ctx.step_name))

        system   = self._client.get_error_diagnosis_system()
        messages = [
            {
                "role": "user",
                "content": (
                    "Here is the error context from a failed model installation step. "
                    "Please diagnose and return a fix plan as JSON.\n\n"
                    f"```json\n{_context_to_message(ctx)}\n```"
                ),
            }
        ]

        raw = await self._client.chat(messages=messages, system=system, max_tokens=4096)

        json_str = _extract_json(raw)
        if not json_str:
            raise DiagnosisFailedError(
                f"LLM response did not contain valid JSON:\n{raw[:500]}"
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise DiagnosisFailedError(f"JSON parse error: {e}\n{json_str[:500]}")

        raw_inputs = data.get("user_inputs_needed", [])
        user_inputs = [
            UserInputRequest(
                env_var=r.get("env_var", r.get("key", "")),
                prompt=r.get("prompt", f"Enter value for {r.get('env_var', r.get('key', ''))}"),
                sensitive=bool(r.get("sensitive", True)),
            )
            for r in raw_inputs
            if r.get("env_var") or r.get("key")
        ]

        result = DiagnosisResult(
            error_category=data.get("error_category", "unknown"),
            root_cause=data.get("root_cause", ""),
            confidence=float(data.get("confidence", 0.5)),
            fix_plan=data.get("fix_plan", []),
            alternative_plans=data.get("alternative_plans", []),
            user_explanation=data.get("user_explanation", ""),
            requires_user_decision=data.get("requires_user_decision", False),
            decision_options=data.get("decision_options", []),
            user_inputs_needed=user_inputs,
        )

        await bus.emit(DiagnosisReadyEvent(
            root_cause=result.root_cause,
            confidence=result.confidence,
            requires_user_decision=result.requires_user_decision,
            decision_options=result.decision_options,
        ))

        return result
