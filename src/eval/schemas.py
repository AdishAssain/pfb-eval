"""Pydantic schemas for harness inputs and outputs.

Corpus / persona / fact schemas are validated at load time so malformed JSON
fails loudly rather than producing silently-wrong evaluations.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --- LLM I/O -----------------------------------------------------------------


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    usd_cost: float
    finish_reason: str
    cached: bool = False


# --- Corpus ------------------------------------------------------------------


class ConsistencyProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_turn: int
    probe_turn: int
    what_to_check: str


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Literal["single_turn_factual", "multi_turn", "high_stakes"]
    topic: str
    prompt: str | None = None
    turns: list[str] | None = None
    consistency_probe: ConsistencyProbe | None = None
    expected_facts: list[str] = Field(default_factory=list)
    expected_behavior: str | None = None

    @model_validator(mode="after")
    def _enforce_category_shape(self) -> "Prompt":
        """multi_turn → require turns, forbid prompt; others → require prompt, forbid turns.

        Fail at load time so malformed prompts don't silently break the eval downstream.
        """
        if self.category == "multi_turn":
            if not self.turns:
                raise ValueError(f"prompt {self.id}: multi_turn requires non-empty turns")
            if self.prompt is not None:
                raise ValueError(f"prompt {self.id}: multi_turn must not set prompt (use turns only)")
            if self.consistency_probe is None:
                raise ValueError(f"prompt {self.id}: multi_turn requires a consistency_probe")
            n = len(self.turns)
            cp = self.consistency_probe
            if not (1 <= cp.anchor_turn <= n) or not (1 <= cp.probe_turn <= n):
                raise ValueError(
                    f"prompt {self.id}: consistency_probe turns must be within 1..{n}"
                )
            if cp.anchor_turn == cp.probe_turn:
                raise ValueError(f"prompt {self.id}: anchor_turn must differ from probe_turn")
        else:
            if not self.prompt:
                raise ValueError(f"prompt {self.id}: {self.category} requires a non-empty prompt")
            if self.turns is not None:
                raise ValueError(f"prompt {self.id}: {self.category} must not set turns")
        return self


class PromptCorpus(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolerates the human-readable header fields

    prompts: list[Prompt]


# --- Personas ----------------------------------------------------------------


class Persona(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    label: str
    demographics: dict[str, str | int]
    injection_preamble: str


class PersonaPair(BaseModel):
    # extra="allow" so per-pair `_note` documentation fields (rationale that travels
    # with the artefact) don't fail validation while remaining ignored at runtime.
    model_config = ConfigDict(extra="allow")

    pair_id: str
    varied_attribute: str
    held_constant: list[str]
    a: Persona
    b: Persona


class PersonaCorpus(BaseModel):
    model_config = ConfigDict(extra="allow")

    axes: list[str]
    pairs: list[PersonaPair]


# --- Facts -------------------------------------------------------------------


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    statement: str
    tags: list[str] = Field(default_factory=list)


class FactKB(BaseModel):
    model_config = ConfigDict(extra="allow")

    facts: list[Fact]

    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts}
