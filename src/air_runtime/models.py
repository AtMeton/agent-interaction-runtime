'Small, serialisable contracts shared by the AIR local runtime components.\n\nThe models in this module deliberately contain dialogue facts and bounded\nmetadata only.  Provider prompts, raw channel payloads, secrets and hidden\nreasoning never belong in these contracts.\n'

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Any, Mapping, Sequence


MAX_TEXT_CHARS = 2_000
MAX_RESPONSE_CHARS = 4_000
MAX_TOPIC_CHARS = 160
MAX_GOAL_CHARS = 240
MAX_ARTIFACT_REFS = 8
MAX_TOPIC_STACK = 5
MAX_OPEN_LOOPS = 8
MAX_COMMITMENTS = 8
MAX_REFERENCES = 12
MAX_FACTS = 12
MAX_ACTIVE_OPTIONS = 4
MAX_OPTION_CHARS = 120
MAX_CONVERSATION_THREADS = 6
MAX_THREAD_SOURCE_TURNS = 12
MAX_RELATED_ENTITIES = 8

EXECUTION_OUTCOMES = {
    "not_requested",
    "not_executed",
    "succeeded",
    "failed",
    "unknown",
    "contradictory",
}
EVIDENCE_SOURCES = {"none", "runtime_executor", "legacy_unverified"}

# This is a public contract marker, not a credential.  It is only meaningful
# when the complete provider output equals the marker after boundary trimming.
SCENE_SILENCE_MARKER = "__AIR_SILENCE_V1__"


class ModelValidationError(ValueError):
    """Raised when an interaction contract would be unsafe or unbounded."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def validate_utc_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelValidationError(f"{field_name} must be a non-empty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ModelValidationError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ModelValidationError(f"{field_name} must include a timezone")
    return value


_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,119}$")


def safe_ref(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _REF_RE.fullmatch(value):
        raise ModelValidationError(f"{field_name} must be a bounded opaque reference")
    # Raw external numeric IDs are intentionally not accepted as AIR refs.
    if re.fullmatch(r"\d{5,}", value):
        raise ModelValidationError(f"{field_name} cannot be a raw numeric external ID")
    return value


def redact_sensitive_text(value: str) -> str:
    """Remove common credential forms before text crosses a durable boundary."""

    value = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", value)
    value = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|credential)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        value,
    )
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "sk-[REDACTED]", value)
    return value


def bounded_text(
    value: str,
    max_chars: int,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ModelValidationError(f"{field_name} must be text")
    cleaned = redact_sensitive_text(value.replace("\x00", "")).strip()
    if not cleaned and not allow_empty:
        raise ModelValidationError(f"{field_name} must not be empty")
    if len(cleaned) > max_chars:
        raise ModelValidationError(f"{field_name} exceeds {max_chars} characters")
    return cleaned


_UNSAFE_METADATA_KEYS = {
    "payload",
    "raw",
    "onebot",
    "protocol",
    "token",
    "access_token",
    "api_key",
    "secret",
    "password",
    "credential",
    "user_id",
    "group_id",
    "message_id",
}


def _safe_metadata(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelValidationError(f"{field_name} must be a mapping")
    if len(value) > 16:
        raise ModelValidationError(f"{field_name} has too many entries")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 60:
            raise ModelValidationError(f"{field_name} contains an invalid key")
        lower_key = key.lower()
        if lower_key in _UNSAFE_METADATA_KEYS or any(
            part in lower_key for part in ("payload", "onebot", "token", "secret", "password")
        ):
            raise ModelValidationError(f"{field_name} contains a raw protocol or secret field")
        if isinstance(item, str):
            result[key] = bounded_text(item, 240, f"{field_name}.{key}", allow_empty=True)
        elif isinstance(item, (bool, int, float)):
            # Metadata is not an identity store; keep scalar hints bounded.
            result[key] = item
        elif item is None:
            result[key] = None
        else:
            raise ModelValidationError(f"{field_name}.{key} must be scalar")
    return result


def _optional_text(value: Any, max_chars: int, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelValidationError(f"{field_name} must be text or null")
    if value == "":
        return None
    return bounded_text(value, max_chars, field_name)


@dataclass(frozen=True)
class InteractionEnvelope:
    """Platform-neutral input to the Runtime."""

    event_ref: str
    conversation_ref: str
    actor_ref: str
    channel_kind: str
    timestamp_utc: str
    dedup_ref: str
    addressing: Mapping[str, Any] = field(default_factory=dict)
    reply: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        safe_ref(self.event_ref, "event_ref")
        safe_ref(self.conversation_ref, "conversation_ref")
        safe_ref(self.actor_ref, "actor_ref")
        safe_ref(self.dedup_ref, "dedup_ref")
        object.__setattr__(self, "channel_kind", bounded_text(self.channel_kind, 40, "channel_kind"))
        validate_utc_text(self.timestamp_utc, "timestamp_utc")
        object.__setattr__(self, "text", bounded_text(self.text, MAX_TEXT_CHARS, "text"))
        object.__setattr__(self, "addressing", _safe_metadata(self.addressing, "addressing"))
        object.__setattr__(self, "reply", _safe_metadata(self.reply, "reply"))
        object.__setattr__(self, "artifact_refs", tuple(self.artifact_refs))
        if len(self.artifact_refs) > MAX_ARTIFACT_REFS:
            raise ModelValidationError("too many artifact_refs")
        for ref in self.artifact_refs:
            safe_ref(ref, "artifact_ref")

    @property
    def timestamp(self) -> str:
        """Compatibility alias for the contract's timestamp field."""

        return self.timestamp_utc

    @property
    def addressing_metadata(self) -> Mapping[str, Any]:
        return self.addressing

    @property
    def reply_metadata(self) -> Mapping[str, Any]:
        return self.reply

    @classmethod
    def local(
        cls,
        *,
        event_ref: str,
        conversation_ref: str,
        actor_ref: str,
        text: str,
        dedup_ref: str | None = None,
        timestamp_utc: str | None = None,
    ) -> "InteractionEnvelope":
        return cls(
            event_ref=event_ref,
            conversation_ref=conversation_ref,
            actor_ref=actor_ref,
            channel_kind="local_harness",
            timestamp_utc=timestamp_utc or utc_now(),
            dedup_ref=dedup_ref or event_ref,
            text=text,
        )


@dataclass(frozen=True)
class ActorContext:
    actor_ref: str
    profile_id: str
    status: str = "active"
    auth_source: str = "local"

    def __post_init__(self) -> None:
        safe_ref(self.actor_ref, "actor_ref")
        safe_ref(self.profile_id, "profile_id")
        bounded_text(self.status, 24, "actor status")
        bounded_text(self.auth_source, 40, "auth_source")


@dataclass(frozen=True)
class TopicFrame:
    topic: str
    goal: str | None
    entered_at_utc: str
    last_active_at_utc: str

    def __post_init__(self) -> None:
        bounded_text(self.topic, MAX_TOPIC_CHARS, "topic")
        if self.goal is not None:
            bounded_text(self.goal, MAX_GOAL_CHARS, "goal")
        validate_utc_text(self.entered_at_utc, "entered_at_utc")
        validate_utc_text(self.last_active_at_utc, "last_active_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "goal": self.goal,
            "entered_at_utc": self.entered_at_utc,
            "last_active_at_utc": self.last_active_at_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopicFrame":
        return cls(
            topic=bounded_text(str(value.get("topic", "")), MAX_TOPIC_CHARS, "topic"),
            goal=_optional_text(value.get("goal"), MAX_GOAL_CHARS, "goal"),
            entered_at_utc=str(value["entered_at_utc"]),
            last_active_at_utc=str(value["last_active_at_utc"]),
        )


@dataclass(frozen=True)
class OpenLoop:
    loop_id: str
    owner_actor_ref: str
    conversation_ref: str
    loop_type: str
    status: str
    summary: str
    resume_condition: str
    created_at_utc: str
    expires_at_utc: str | None = None
    non_modal: bool = True

    def __post_init__(self) -> None:
        safe_ref(self.loop_id, "loop_id")
        safe_ref(self.owner_actor_ref, "owner_actor_ref")
        safe_ref(self.conversation_ref, "conversation_ref")
        bounded_text(self.loop_type, 60, "loop_type")
        if self.status not in {"open", "resolved", "cancelled", "expired"}:
            raise ModelValidationError("invalid open loop status")
        bounded_text(self.summary, 240, "loop summary")
        bounded_text(self.resume_condition, 240, "resume_condition")
        if not isinstance(self.non_modal, bool):
            raise ModelValidationError("open loop non_modal must be boolean")
        validate_utc_text(self.created_at_utc, "created_at_utc")
        if self.expires_at_utc is not None:
            validate_utc_text(self.expires_at_utc, "expires_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "owner_actor_ref": self.owner_actor_ref,
            "conversation_ref": self.conversation_ref,
            "loop_type": self.loop_type,
            "status": self.status,
            "summary": self.summary,
            "resume_condition": self.resume_condition,
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "non_modal": self.non_modal,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OpenLoop":
        return cls(
            loop_id=str(value["loop_id"]),
            owner_actor_ref=str(value["owner_actor_ref"]),
            conversation_ref=str(value["conversation_ref"]),
            loop_type=str(value["loop_type"]),
            status=str(value["status"]),
            summary=str(value["summary"]),
            resume_condition=str(value["resume_condition"]),
            created_at_utc=str(value["created_at_utc"]),
            expires_at_utc=value.get("expires_at_utc"),
            non_modal=bool(value.get("non_modal", True)),
        )


@dataclass(frozen=True)
class Commitment:
    commitment_id: str
    owner_actor_ref: str
    conversation_ref: str
    text: str
    status: str
    created_at_utc: str
    resolved_at_utc: str | None = None

    def __post_init__(self) -> None:
        safe_ref(self.commitment_id, "commitment_id")
        safe_ref(self.owner_actor_ref, "owner_actor_ref")
        safe_ref(self.conversation_ref, "conversation_ref")
        bounded_text(self.text, 240, "commitment")
        if self.status not in {"active", "resolved", "cancelled"}:
            raise ModelValidationError("invalid commitment status")
        validate_utc_text(self.created_at_utc, "created_at_utc")
        if self.resolved_at_utc is not None:
            validate_utc_text(self.resolved_at_utc, "resolved_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "owner_actor_ref": self.owner_actor_ref,
            "conversation_ref": self.conversation_ref,
            "text": self.text,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "resolved_at_utc": self.resolved_at_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Commitment":
        return cls(
            commitment_id=str(value["commitment_id"]),
            owner_actor_ref=str(value["owner_actor_ref"]),
            conversation_ref=str(value["conversation_ref"]),
            text=str(value["text"]),
            status=str(value["status"]),
            created_at_utc=str(value["created_at_utc"]),
            resolved_at_utc=value.get("resolved_at_utc"),
        )


@dataclass(frozen=True)
class ReferenceBinding:
    reference_key: str
    label: str
    target_ref: str | None
    confidence: float
    status: str
    created_at_utc: str
    expires_at_utc: str | None = None
    owner_actor_ref: str | None = None
    conversation_ref: str | None = None

    def __post_init__(self) -> None:
        bounded_text(self.reference_key, 50, "reference_key")
        bounded_text(self.label, 120, "reference label")
        if self.target_ref is not None:
            bounded_text(self.target_ref, 240, "reference target")
            if re.fullmatch(r"\d{5,}", self.target_ref):
                raise ModelValidationError("reference target cannot be a raw numeric external ID")
        if self.owner_actor_ref is not None:
            safe_ref(self.owner_actor_ref, "reference owner_actor_ref")
        if self.conversation_ref is not None:
            safe_ref(self.conversation_ref, "reference conversation_ref")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ModelValidationError("reference confidence must be between 0 and 1")
        if self.status not in {"resolved", "ambiguous", "expired"}:
            raise ModelValidationError("invalid reference status")
        validate_utc_text(self.created_at_utc, "reference created_at_utc")
        if self.expires_at_utc is not None:
            validate_utc_text(self.expires_at_utc, "reference expires_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_key": self.reference_key,
            "label": self.label,
            "target_ref": self.target_ref,
            "confidence": self.confidence,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "owner_actor_ref": self.owner_actor_ref,
            "conversation_ref": self.conversation_ref,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReferenceBinding":
        return cls(
            reference_key=str(value["reference_key"]),
            label=str(value["label"]),
            target_ref=value.get("target_ref"),
            confidence=float(value["confidence"]),
            status=str(value["status"]),
            created_at_utc=str(value["created_at_utc"]),
            expires_at_utc=value.get("expires_at_utc"),
            owner_actor_ref=value.get("owner_actor_ref"),
            conversation_ref=value.get("conversation_ref"),
        )


@dataclass(frozen=True)
class ClarificationState:
    status: str
    question: str | None
    subject: str | None
    attempts: int
    created_at_utc: str | None

    def __post_init__(self) -> None:
        if self.status not in {"none", "awaiting"}:
            raise ModelValidationError("invalid clarification status")
        if self.status == "awaiting":
            if self.question is None:
                raise ModelValidationError("awaiting clarification needs a question")
            bounded_text(self.question, 400, "clarification question")
            if self.created_at_utc is None:
                raise ModelValidationError("awaiting clarification needs a timestamp")
        if self.subject is not None:
            bounded_text(self.subject, 160, "clarification subject")
        if self.attempts < 0 or self.attempts > 3:
            raise ModelValidationError("clarification attempts out of bounds")
        if self.created_at_utc is not None:
            validate_utc_text(self.created_at_utc, "clarification created_at_utc")

    @classmethod
    def none(cls) -> "ClarificationState":
        return cls("none", None, None, 0, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "question": self.question,
            "subject": self.subject,
            "attempts": self.attempts,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ClarificationState":
        if not value:
            return cls.none()
        return cls(
            status=str(value.get("status", "none")),
            question=value.get("question"),
            subject=value.get("subject"),
            attempts=int(value.get("attempts", 0)),
            created_at_utc=value.get("created_at_utc"),
        )


@dataclass(frozen=True)
class ConversationThread:
    """Bounded semantic continuity owned by one actor and conversation.

    ``semantic_summary`` is model-facing context, never user-facing display
    text.  Source refs make every thread auditable without storing hidden
    reasoning or a second copy of the transcript.
    """

    thread_id: str
    owner_actor_ref: str
    conversation_ref: str
    semantic_summary: str
    source_turn_ids: tuple[str, ...]
    created_at_utc: str
    last_active_at_utc: str
    status: str = "active"
    related_entities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        safe_ref(self.thread_id, "thread_id")
        safe_ref(self.owner_actor_ref, "thread owner_actor_ref")
        safe_ref(self.conversation_ref, "thread conversation_ref")
        bounded_text(self.semantic_summary, 320, "thread semantic_summary")
        if self.status not in {"active", "paused", "closed"}:
            raise ModelValidationError("invalid conversation thread status")
        if len(self.source_turn_ids) > MAX_THREAD_SOURCE_TURNS:
            raise ModelValidationError("conversation thread has too many source turns")
        for source_turn_id in self.source_turn_ids:
            safe_ref(source_turn_id, "thread source_turn_id")
        if len(self.related_entities) > MAX_RELATED_ENTITIES:
            raise ModelValidationError("conversation thread has too many related entities")
        for entity in self.related_entities:
            bounded_text(entity, 120, "thread related entity")
        validate_utc_text(self.created_at_utc, "thread created_at_utc")
        validate_utc_text(self.last_active_at_utc, "thread last_active_at_utc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "owner_actor_ref": self.owner_actor_ref,
            "conversation_ref": self.conversation_ref,
            "semantic_summary": self.semantic_summary,
            "source_turn_ids": list(self.source_turn_ids),
            "created_at_utc": self.created_at_utc,
            "last_active_at_utc": self.last_active_at_utc,
            "status": self.status,
            "related_entities": list(self.related_entities),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConversationThread":
        return cls(
            thread_id=str(value["thread_id"]),
            owner_actor_ref=str(value["owner_actor_ref"]),
            conversation_ref=str(value["conversation_ref"]),
            semantic_summary=str(value["semantic_summary"]),
            source_turn_ids=tuple(str(item) for item in value.get("source_turn_ids", ())),
            created_at_utc=str(value["created_at_utc"]),
            last_active_at_utc=str(value["last_active_at_utc"]),
            status=str(value.get("status", "active")),
            related_entities=tuple(str(item) for item in value.get("related_entities", ())),
        )


@dataclass(frozen=True)
class DialogueState:
    """Minimal durable state; recent turns are stored separately."""

    conversation_ref: str
    current_topic: TopicFrame | None = None
    topic_stack: tuple[TopicFrame, ...] = ()
    active_goal: str | None = None
    open_loops: tuple[OpenLoop, ...] = ()
    commitments: tuple[Commitment, ...] = ()
    reference_bindings: tuple[ReferenceBinding, ...] = ()
    unresolved_references: tuple[str, ...] = ()
    clarification: ClarificationState = field(default_factory=ClarificationState.none)
    revision: int = 0
    updated_at_utc: str = field(default_factory=utc_now)
    active_options: tuple[str, ...] = ()
    active_preference: str | None = None
    conversation_threads: tuple[ConversationThread, ...] = ()
    active_thread_id: str | None = None

    def __post_init__(self) -> None:
        safe_ref(self.conversation_ref, "conversation_ref")
        if self.current_topic is not None and not isinstance(self.current_topic, TopicFrame):
            raise ModelValidationError("current_topic must be a TopicFrame")
        if len(self.topic_stack) > MAX_TOPIC_STACK:
            raise ModelValidationError("topic_stack exceeds bounded depth")
        if len(self.open_loops) > MAX_OPEN_LOOPS:
            raise ModelValidationError("too many open loops")
        if len(self.commitments) > MAX_COMMITMENTS:
            raise ModelValidationError("too many commitments")
        if len(self.reference_bindings) > MAX_REFERENCES:
            raise ModelValidationError("too many reference bindings")
        if len(self.unresolved_references) > MAX_REFERENCES:
            raise ModelValidationError("too many unresolved references")
        for ref in self.unresolved_references:
            bounded_text(ref, 120, "unresolved reference")
        if self.active_goal is not None:
            bounded_text(self.active_goal, MAX_GOAL_CHARS, "active_goal")
        if len(self.active_options) > MAX_ACTIVE_OPTIONS:
            raise ModelValidationError("active_options exceeds bounded size")
        for option in self.active_options:
            bounded_text(option, MAX_OPTION_CHARS, "active option")
        if self.active_preference is not None:
            bounded_text(self.active_preference, MAX_OPTION_CHARS, "active_preference")
            if self.active_options and self.active_preference not in self.active_options:
                raise ModelValidationError("active_preference must be one of active_options")
        if len(self.conversation_threads) > MAX_CONVERSATION_THREADS:
            raise ModelValidationError("conversation_threads exceeds bounded size")
        thread_ids: set[str] = set()
        for thread in self.conversation_threads:
            if not isinstance(thread, ConversationThread):
                raise ModelValidationError("conversation_threads must contain ConversationThread")
            if thread.owner_actor_ref == "" or thread.conversation_ref != self.conversation_ref:
                raise ModelValidationError("conversation thread scope mismatch")
            if thread.thread_id in thread_ids:
                raise ModelValidationError("duplicate conversation thread id")
            thread_ids.add(thread.thread_id)
        if self.active_thread_id is not None:
            safe_ref(self.active_thread_id, "active_thread_id")
            if self.active_thread_id not in thread_ids:
                raise ModelValidationError("active_thread_id must identify a retained thread")
        if self.revision < 0:
            raise ModelValidationError("revision cannot be negative")
        validate_utc_text(self.updated_at_utc, "updated_at_utc")
        for loop in self.open_loops:
            if loop.owner_actor_ref == "":
                raise ModelValidationError("open loop owner is required")

    @classmethod
    def empty(cls, conversation_ref: str, now: str | None = None) -> "DialogueState":
        return cls(conversation_ref=conversation_ref, updated_at_utc=now or utc_now())

    def bump(self, **changes: Any) -> "DialogueState":
        values = {
            "conversation_ref": self.conversation_ref,
            "current_topic": self.current_topic,
            "topic_stack": self.topic_stack,
            "active_goal": self.active_goal,
            "active_options": self.active_options,
            "active_preference": self.active_preference,
            "conversation_threads": self.conversation_threads,
            "active_thread_id": self.active_thread_id,
            "open_loops": self.open_loops,
            "commitments": self.commitments,
            "reference_bindings": self.reference_bindings,
            "unresolved_references": self.unresolved_references,
            "clarification": self.clarification,
            "revision": self.revision + 1,
            "updated_at_utc": utc_now(),
        }
        values.update(changes)
        return DialogueState(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_ref": self.conversation_ref,
            "current_topic": self.current_topic.to_dict() if self.current_topic else None,
            "topic_stack": [item.to_dict() for item in self.topic_stack],
            "active_goal": self.active_goal,
            "active_options": list(self.active_options),
            "active_preference": self.active_preference,
            "conversation_threads": [item.to_dict() for item in self.conversation_threads],
            "active_thread_id": self.active_thread_id,
            "open_loops": [item.to_dict() for item in self.open_loops],
            "commitments": [item.to_dict() for item in self.commitments],
            "reference_bindings": [item.to_dict() for item in self.reference_bindings],
            "unresolved_references": list(self.unresolved_references),
            "clarification": self.clarification.to_dict(),
            "revision": self.revision,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DialogueState":
        return cls(
            conversation_ref=str(value["conversation_ref"]),
            current_topic=(
                TopicFrame.from_dict(value["current_topic"])
                if value.get("current_topic")
                else None
            ),
            topic_stack=tuple(TopicFrame.from_dict(item) for item in value.get("topic_stack", [])),
            active_goal=value.get("active_goal"),
            active_options=tuple(str(item) for item in value.get("active_options", [])),
            active_preference=value.get("active_preference"),
            conversation_threads=tuple(
                ConversationThread.from_dict(item)
                for item in value.get("conversation_threads", [])
            ),
            active_thread_id=value.get("active_thread_id"),
            open_loops=tuple(OpenLoop.from_dict(item) for item in value.get("open_loops", [])),
            commitments=tuple(Commitment.from_dict(item) for item in value.get("commitments", [])),
            reference_bindings=tuple(
                ReferenceBinding.from_dict(item) for item in value.get("reference_bindings", [])
            ),
            unresolved_references=tuple(str(item) for item in value.get("unresolved_references", [])),
            clarification=ClarificationState.from_dict(value.get("clarification")),
            revision=int(value.get("revision", 0)),
            updated_at_utc=str(value.get("updated_at_utc", utc_now())),
        )


@dataclass(frozen=True)
class RecentTurn:
    role: str
    text: str
    created_at_utc: str
    strategy: str | None = None
    response_intent: str | None = None
    conversation_move: str | None = None
    status: str = "success"
    turn_ref: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "system"}:
            raise ModelValidationError("invalid turn role")
        bounded_text(self.text, MAX_RESPONSE_CHARS, "turn text")
        validate_utc_text(self.created_at_utc, "turn created_at_utc")
        if self.strategy is not None:
            bounded_text(self.strategy, 24, "turn strategy")
        if self.response_intent is not None:
            bounded_text(self.response_intent, 48, "response intent")
        if self.conversation_move is not None:
            bounded_text(self.conversation_move, 48, "conversation move")
        bounded_text(self.status, 32, "turn status")
        if self.turn_ref is not None:
            safe_ref(self.turn_ref, "turn_ref")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "created_at_utc": self.created_at_utc,
            "strategy": self.strategy,
            "response_intent": self.response_intent,
            "conversation_move": self.conversation_move,
            "status": self.status,
            "turn_ref": self.turn_ref,
        }


@dataclass(frozen=True)
class GroundingFrame:
    """Turn-scoped facts and boundaries; never written as durable state."""

    verified_facts: tuple[str, ...] = ()
    inferred_facts: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    resolved_references: tuple[str, ...] = ()
    current_state: Mapping[str, Any] = field(default_factory=dict)
    capability_boundary: str = 'No external tools or side effects are available in local runtime.'

    def __post_init__(self) -> None:
        for name, values in (
            ("verified_facts", self.verified_facts),
            ("inferred_facts", self.inferred_facts),
            ("unknowns", self.unknowns),
            ("source_refs", self.source_refs),
            ("resolved_references", self.resolved_references),
        ):
            if len(values) > MAX_FACTS:
                raise ModelValidationError(f"{name} exceeds bounded size")
            for item in values:
                bounded_text(item, 240, name)
        _safe_metadata(self.current_state, "current_state")
        bounded_text(self.capability_boundary, 400, "capability_boundary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified_facts": list(self.verified_facts),
            "inferred_facts": list(self.inferred_facts),
            "unknowns": list(self.unknowns),
            "source_refs": list(self.source_refs),
            "resolved_references": list(self.resolved_references),
            "current_state": dict(self.current_state),
            "capability_boundary": self.capability_boundary,
        }


STATE_DELTA_OPERATIONS = {
    "no_change",
    "set_topic",
    "push_topic",
    "resume_topic",
    "set_goal",
    "clear_goal",
    "set_preference",
    "add_open_loop",
    "update_open_loop",
    "remove_open_loop",
    "bind_reference",
    "replace_reference",
    "add_commitment",
    "update_commitment",
    "resolve_commitment",
    "update_clarification",
    "clear_clarification",
}


@dataclass(frozen=True)
class DialogueStateDelta:
    """A small, validated instruction for a possible durable state change.

    A delta is not state authority.  It is an explicit description that must
    still pass the deterministic admission policy in ``dialogue.py`` before
    it can affect ``DialogueState``.  ``value`` is intentionally generic so
    the Direct provider contract stays small; ``target`` identifies an
    existing loop/reference/commitment when an operation needs one.
    """

    operation: str = "no_change"
    value: str | None = None
    target: str | None = None
    reason: str | None = None
    supporting_turn_ids: tuple[str, ...] = ()
    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in STATE_DELTA_OPERATIONS:
            raise ModelValidationError("invalid dialogue state delta operation")
        if self.operation in {"set_topic", "push_topic", "set_goal", "set_preference", "add_open_loop", "add_commitment"} and self.value is None:
            raise ModelValidationError(f"{self.operation} state delta needs a value")
        if self.operation in {"bind_reference", "replace_reference", "update_clarification"} and self.value is None:
            raise ModelValidationError(f"{self.operation} state delta needs a value")
        if self.operation in {
            "update_open_loop",
            "remove_open_loop",
            "update_commitment",
            "resolve_commitment",
        } and self.target is None:
            raise ModelValidationError(f"{self.operation} state delta needs a target")
        value_limit = {
            "set_topic": MAX_TOPIC_CHARS,
            "push_topic": MAX_TOPIC_CHARS,
            "set_goal": MAX_GOAL_CHARS,
            "set_preference": MAX_OPTION_CHARS,
            "add_open_loop": 240,
            "update_open_loop": 240,
            "add_commitment": 240,
            "update_commitment": 240,
            "bind_reference": 120,
            "replace_reference": 120,
            "update_clarification": 400,
        }.get(self.operation, MAX_RESPONSE_CHARS)
        for item, limit, name in (
            (self.value, value_limit, "state delta value"),
            (self.target, 240, "state delta target"),
            (self.reason, 120, "state delta reason"),
        ):
            if item is not None:
                bounded_text(item, limit, name)
        if self.operation == "no_change" and any(
            item is not None for item in (self.value, self.target)
        ):
            raise ModelValidationError("no_change state delta cannot carry a value or target")
        if len(self.supporting_turn_ids) > MAX_THREAD_SOURCE_TURNS:
            raise ModelValidationError("state delta has too many supporting turns")
        for source_turn_id in self.supporting_turn_ids:
            safe_ref(source_turn_id, "state delta supporting_turn_id")
        if len(self.options) > MAX_ACTIVE_OPTIONS:
            raise ModelValidationError("state delta options exceed bounded size")
        for option in self.options:
            bounded_text(option, MAX_OPTION_CHARS, "state delta option")
        if self.operation == "set_preference" and self.options and self.value not in self.options:
            raise ModelValidationError("state delta preference must be one of its options")

    @property
    def kind(self) -> str:
        """Compatibility alias for callers that call the operation a kind."""

        return self.operation

    @classmethod
    def no_change(cls, *, reason: str | None = None) -> "DialogueStateDelta":
        return cls(operation="no_change", reason=reason)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"operation": self.operation}
        if self.value is not None:
            result["value"] = self.value
        if self.target is not None:
            result["target"] = self.target
        if self.reason is not None:
            result["reason"] = self.reason
        if self.supporting_turn_ids:
            result["supporting_turn_ids"] = list(self.supporting_turn_ids)
        if self.options:
            result["options"] = list(self.options)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DialogueStateDelta":
        if not isinstance(value, Mapping):
            raise ModelValidationError("state_delta must be an object")
        allowed = {
            "operation",
            "kind",
            "value",
            "target",
            "reason",
            "topic",
            "goal",
            "reference",
            "summary",
            "text",
            "supporting_turn_ids",
            "options",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ModelValidationError(f"state_delta has unknown fields: {sorted(unknown)}")
        operation = value.get("operation", value.get("kind"))
        if "operation" in value and "kind" in value and value["operation"] != value["kind"]:
            raise ModelValidationError("state_delta operation and kind disagree")
        if not isinstance(operation, str):
            raise ModelValidationError("state_delta operation must be text")
        aliases = [
            value.get(name)
            for name in ("topic", "goal", "reference", "summary", "text")
            if name in value
        ]
        if "value" in value and aliases:
            raise ModelValidationError("state_delta value has duplicate aliases")
        if len(aliases) > 1:
            raise ModelValidationError("state_delta has multiple value aliases")
        value_text = value.get("value", aliases[0] if aliases else None)
        target = value.get("target")
        reason = value.get("reason")
        for item, name in ((value_text, "state delta value"), (target, "state delta target"), (reason, "state delta reason")):
            if item is not None and not isinstance(item, str):
                raise ModelValidationError(f"{name} must be text or null")
        supporting = value.get("supporting_turn_ids", ())
        options = value.get("options", ())
        if not isinstance(supporting, (list, tuple)) or not all(isinstance(item, str) for item in supporting):
            raise ModelValidationError("state delta supporting_turn_ids must be a text list")
        if not isinstance(options, (list, tuple)) or not all(isinstance(item, str) for item in options):
            raise ModelValidationError("state delta options must be a text list")
        return cls(
            operation=operation,
            value=value_text,
            target=target,
            reason=reason,
            supporting_turn_ids=tuple(supporting),
            options=tuple(options),
        )


class ResponseIntent(str, Enum):
    CONVERSATION = "conversation"
    ACKNOWLEDGE = "acknowledge"
    CONTINUE = "continue"
    CLARIFY = "clarify"
    IDENTITY = "identity"
    UNKNOWN = "unknown"
    CAPABILITY_BOUNDARY = "capability_boundary"


PROPOSAL_INTENTS = {item.value for item in ResponseIntent}


class MainMindOutcome(str, Enum):
    """The only two outcomes the Scene Main Mind may choose."""

    SPEAK = "speak"
    SILENCE = "silence"


class RuntimeOutcome(str, Enum):
    """The Runtime's durable result vocabulary, including failures."""

    SPEAK = MainMindOutcome.SPEAK.value
    SILENCE = MainMindOutcome.SILENCE.value
    FAILED_TO_RESPOND = "failed_to_respond"


@dataclass(frozen=True)
class MainMindDecision:
    """Minimal egress decision returned by the Scene Main Mind.

    ``FAILED_TO_RESPOND`` intentionally does not appear here.  Provider
    failures and malformed output are Runtime outcomes, never a model-chosen
    conversational action.
    """

    outcome: str
    text: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in {item.value for item in MainMindOutcome}:
            raise ModelValidationError("invalid Main Mind outcome")
        if self.outcome == MainMindOutcome.SPEAK.value:
            bounded_text(self.text, MAX_RESPONSE_CHARS, "Main Mind spoken text")
        elif self.text.strip():
            raise ModelValidationError("SILENCE cannot carry participant-facing text")

    @classmethod
    def speak(cls, text: str) -> "MainMindDecision":
        return cls(outcome=MainMindOutcome.SPEAK.value, text=text)

    @classmethod
    def silence(cls) -> "MainMindDecision":
        return cls(outcome=MainMindOutcome.SILENCE.value)

    @classmethod
    def parse(cls, value: str) -> "MainMindDecision":
        """Parse the frozen exact-token contract without guessing intent."""

        if not isinstance(value, str):
            raise ModelValidationError("Main Mind output must be text")
        stripped = value.strip()
        if not stripped:
            raise ModelValidationError("Main Mind output is empty")
        if stripped == SCENE_SILENCE_MARKER:
            return cls.silence()
        if SCENE_SILENCE_MARKER in stripped:
            raise ModelValidationError("silence marker must be the complete output")
        if (
            stripped.startswith("```")
            or stripped.endswith("```")
            or (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ):
            raise ModelValidationError(
                "Main Mind output must be plain text without structured wrappers"
            )
        return cls.speak(stripped)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"outcome": self.outcome}
        if self.outcome == MainMindOutcome.SPEAK.value:
            result["text"] = self.text
        return result


class ConversationControlIntent(str, Enum):
    """Protected control meaning for the current conversational turn."""

    CONTINUE = "continue"
    PAUSE_CURRENT_TOPIC = "pause_current_topic"
    SWITCH_TOPIC = "switch_topic"
    RESUME_TOPIC = "resume_topic"
    CORRECT = "correct"
    ACKNOWLEDGE = "acknowledge"
    NONE = "none"


CONTROL_INTENTS = {item.value for item in ConversationControlIntent}


@dataclass(frozen=True)
class TurnProposal:
    """Bounded semantic proposal returned by a provider."""

    strategy_candidate: str
    response_intent: str
    topic_candidate: str | None = None
    goal_candidate: str | None = None
    reference_candidate: str | None = None
    clarification_candidate: str | None = None
    open_loop_candidate: str | None = None
    commitment_candidate: str | None = None
    wording: str = ""
    state_delta: DialogueStateDelta | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.strategy_candidate not in {"direct", "deliberative"}:
            raise ModelValidationError("invalid strategy candidate")
        if self.response_intent not in PROPOSAL_INTENTS:
            raise ModelValidationError("invalid response intent")
        for value, size, name in (
            (self.topic_candidate, MAX_TOPIC_CHARS, "topic_candidate"),
            (self.goal_candidate, MAX_GOAL_CHARS, "goal_candidate"),
            (self.reference_candidate, 240, "reference_candidate"),
            (self.clarification_candidate, 400, "clarification_candidate"),
            (self.open_loop_candidate, 240, "open_loop_candidate"),
            (self.commitment_candidate, 240, "commitment_candidate"),
        ):
            if value is not None:
                bounded_text(value, size, name)
        bounded_text(self.wording, MAX_RESPONSE_CHARS, "provider wording")
        if self.state_delta is not None and not isinstance(self.state_delta, DialogueStateDelta):
            if isinstance(self.state_delta, Mapping):
                object.__setattr__(self, "state_delta", DialogueStateDelta.from_mapping(self.state_delta))
            else:
                raise ModelValidationError("state_delta must be a DialogueStateDelta")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, default_strategy: str | None = None
    ) -> "TurnProposal":
        if not isinstance(value, Mapping):
            raise ModelValidationError("provider proposal must be an object")
        allowed = {
            "strategy_candidate",
            "response_intent",
            "topic_candidate",
            "goal_candidate",
            "reference_candidate",
            "clarification_candidate",
            "open_loop_candidate",
            "commitment_candidate",
            "wording",
            "state_delta",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ModelValidationError(f"provider proposal has unknown fields: {sorted(unknown)}")
        required = {"response_intent", "wording"}
        if default_strategy is None:
            required.add("strategy_candidate")
        missing = required - set(value)
        if missing:
            raise ModelValidationError(f"provider proposal missing fields: {sorted(missing)}")
        for field_name in required:
            if not isinstance(value[field_name], str):
                raise ModelValidationError(f"provider proposal {field_name} must be text")
        strategy_candidate = value.get("strategy_candidate", default_strategy)
        if not isinstance(strategy_candidate, str):
            raise ModelValidationError("provider proposal strategy_candidate must be text")
        state_delta_value = value.get("state_delta")
        state_delta = None
        if state_delta_value is not None:
            state_delta = (
                state_delta_value
                if isinstance(state_delta_value, DialogueStateDelta)
                else DialogueStateDelta.from_mapping(state_delta_value)
            )
        return cls(
            strategy_candidate=strategy_candidate,
            response_intent=value["response_intent"],
            topic_candidate=_optional_text(value.get("topic_candidate"), MAX_TOPIC_CHARS, "topic_candidate"),
            goal_candidate=_optional_text(value.get("goal_candidate"), MAX_GOAL_CHARS, "goal_candidate"),
            reference_candidate=_optional_text(value.get("reference_candidate"), 240, "reference_candidate"),
            clarification_candidate=_optional_text(
                value.get("clarification_candidate"), 400, "clarification_candidate"
            ),
            open_loop_candidate=_optional_text(value.get("open_loop_candidate"), 240, "open_loop_candidate"),
            commitment_candidate=_optional_text(
                value.get("commitment_candidate"), 240, "commitment_candidate"
            ),
            wording=bounded_text(value["wording"], MAX_RESPONSE_CHARS, "provider wording"),
            state_delta=state_delta,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "strategy_candidate": self.strategy_candidate,
            "response_intent": self.response_intent,
            "wording": self.wording,
        }
        for name in (
            "topic_candidate",
            "goal_candidate",
            "reference_candidate",
            "clarification_candidate",
            "open_loop_candidate",
            "commitment_candidate",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.state_delta is not None:
            result["state_delta"] = self.state_delta.to_dict()
        return result


@dataclass(frozen=True)
class ProtectedFacts:
    response_intent: str
    verified_facts: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    capability_available: bool = False
    side_effect_executed: bool = False
    side_effect_requested: bool = False
    execution_outcome: str = "not_requested"
    postcondition_validated: bool = False
    committed_durable_result: bool = False
    evidence_source: str = "none"
    state_revision: int = 0
    conversation_control: str = ConversationControlIntent.NONE.value
    epistemic_status: str = "unknown"

    def __post_init__(self) -> None:
        bounded_text(self.response_intent, 48, "protected response intent")
        if self.conversation_control not in CONTROL_INTENTS:
            raise ModelValidationError("invalid conversation control intent")
        if self.epistemic_status not in {"shared_context", "no_shared_context", "unknown"}:
            raise ModelValidationError("invalid protected epistemic status")
        if self.execution_outcome not in EXECUTION_OUTCOMES:
            raise ModelValidationError("invalid execution outcome")
        if self.evidence_source not in EVIDENCE_SOURCES:
            raise ModelValidationError("invalid execution evidence source")
        for name, values in (("verified_facts", self.verified_facts), ("unknowns", self.unknowns)):
            if len(values) > MAX_FACTS:
                raise ModelValidationError(f"{name} exceeds bounded size")
            for item in values:
                bounded_text(item, 240, name)
        if self.side_effect_executed and not self.capability_available:
            raise ModelValidationError("a side effect cannot execute without capability")
        if self.execution_outcome == "succeeded" and not self.side_effect_executed:
            raise ModelValidationError("a succeeded execution needs side_effect_executed")
        if self.side_effect_executed and self.execution_outcome != "succeeded":
            raise ModelValidationError("executed side effect needs a succeeded outcome")
        if self.postcondition_validated and not self.side_effect_executed:
            raise ModelValidationError("a postcondition needs an executed side effect")
        if self.committed_durable_result and not self.postcondition_validated:
            raise ModelValidationError("a durable result needs a validated postcondition")
        if (
            self.side_effect_executed
            or self.postcondition_validated
            or self.committed_durable_result
        ) and self.evidence_source not in {"runtime_executor", "legacy_unverified"}:
            raise ModelValidationError("positive execution facts need an evidence source")
        if self.state_revision < 0:
            raise ModelValidationError("state revision cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_intent": self.response_intent,
            "conversation_control": self.conversation_control,
            "verified_facts": list(self.verified_facts),
            "unknowns": list(self.unknowns),
            "capability_available": self.capability_available,
            "side_effect_executed": self.side_effect_executed,
            "side_effect_requested": self.side_effect_requested,
            "execution_outcome": self.execution_outcome,
            "postcondition_validated": self.postcondition_validated,
            "committed_durable_result": self.committed_durable_result,
            "evidence_source": self.evidence_source,
            "state_revision": self.state_revision,
            "epistemic_status": self.epistemic_status,
        }


@dataclass(frozen=True)
class FinalResponse:
    text: str
    intent: str
    protected_facts: ProtectedFacts
    strategy: str
    status: str = "success"
    error_code: str | None = None
    conversation_move: str | None = None

    def __post_init__(self) -> None:
        bounded_text(
            self.text,
            MAX_RESPONSE_CHARS,
            "final response",
            allow_empty=self.status == RuntimeOutcome.SILENCE.value,
        )
        bounded_text(self.intent, 48, "final response intent")
        bounded_text(self.strategy, 24, "final response strategy")
        bounded_text(self.status, 32, "final response status")
        if self.error_code is not None:
            bounded_text(self.error_code, 64, "response error code")
        if self.conversation_move is not None:
            bounded_text(self.conversation_move, 48, "response conversation move")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "intent": self.intent,
            "protected_facts": self.protected_facts.to_dict(),
            "strategy": self.strategy,
            "status": self.status,
            "error_code": self.error_code,
            "conversation_move": self.conversation_move,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FinalResponse":
        facts = value.get("protected_facts", {})
        return cls(
            text=str(value["text"]),
            intent=str(value["intent"]),
            protected_facts=ProtectedFacts(
                response_intent=str(facts.get("response_intent", value["intent"])),
                conversation_control=str(facts.get("conversation_control", ConversationControlIntent.NONE.value)),
                verified_facts=tuple(str(item) for item in facts.get("verified_facts", [])),
                unknowns=tuple(str(item) for item in facts.get("unknowns", [])),
                capability_available=bool(facts.get("capability_available", False)),
                side_effect_executed=bool(facts.get("side_effect_executed", False)),
                side_effect_requested=bool(facts.get("side_effect_requested", False)),
                execution_outcome=str(
                    facts.get(
                        "execution_outcome",
                        "succeeded"
                        if bool(facts.get("side_effect_executed", False))
                        else "not_requested",
                    )
                ),
                postcondition_validated=bool(facts.get("postcondition_validated", False)),
                committed_durable_result=bool(facts.get("committed_durable_result", False)),
                evidence_source=str(
                    facts.get(
                        "evidence_source",
                        "legacy_unverified"
                        if bool(facts.get("side_effect_executed", False))
                        else "none",
                    )
                ),
                state_revision=int(facts.get("state_revision", 0)),
                epistemic_status=str(facts.get("epistemic_status", "unknown")),
            ),
            strategy=str(value.get("strategy", "direct")),
            status=str(value.get("status", "success")),
            error_code=value.get("error_code"),
            conversation_move=value.get("conversation_move"),
        )


@dataclass(frozen=True)
class TurnResult:
    event_ref: str
    status: str
    response: FinalResponse
    strategy: str
    deduplicated: bool = False
    error_code: str | None = None
    user_move: str | None = None
    outcome: str = RuntimeOutcome.SPEAK.value

    def __post_init__(self) -> None:
        safe_ref(self.event_ref, "event_ref")
        bounded_text(self.status, 32, "turn result status")
        bounded_text(self.strategy, 24, "turn result strategy")
        if self.error_code is not None:
            bounded_text(self.error_code, 64, "turn result error code")
        if self.user_move is not None:
            bounded_text(self.user_move, 48, "turn result user move")
        if self.outcome not in {item.value for item in RuntimeOutcome}:
            raise ModelValidationError("invalid Runtime outcome")
        if self.outcome == RuntimeOutcome.SILENCE.value:
            if self.status != RuntimeOutcome.SILENCE.value or self.response.text:
                raise ModelValidationError("SILENCE result must have no assistant text")
        if self.outcome == RuntimeOutcome.FAILED_TO_RESPOND.value and self.status != "failed":
            raise ModelValidationError("failed Runtime outcome needs failed status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_ref": self.event_ref,
            "status": self.status,
            "response": self.response.to_dict(),
            "strategy": self.strategy,
            "deduplicated": self.deduplicated,
            "error_code": self.error_code,
            "user_move": self.user_move,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, deduplicated: bool | None = None) -> "TurnResult":
        stored_status = str(value["status"])
        stored_outcome = value.get("outcome")
        if stored_outcome is None:
            stored_outcome = (
                RuntimeOutcome.FAILED_TO_RESPOND.value
                if stored_status == "failed"
                else RuntimeOutcome.SPEAK.value
            )
        return cls(
            event_ref=str(value["event_ref"]),
            status=stored_status,
            response=FinalResponse.from_dict(value["response"]),
            strategy=str(value.get("strategy", value["response"].get("strategy", "direct"))),
            deduplicated=bool(value.get("deduplicated", False) if deduplicated is None else deduplicated),
            error_code=value.get("error_code"),
            user_move=value.get("user_move"),
            outcome=str(stored_outcome),
        )


def contains_forbidden_output_marker(text: str) -> bool:
    markers = (
        "chain of thought",
        "internal prompt",
        "system prompt",
        "turnproposal",
        "dialoguestate",
        "groundingframe",
        "conversationcognition",
        "responseplan",
        "salient_subject",
        "question_need",
        "advice_need",
        "personality_profile",
        "strategy_candidate",
        "response_intent",
        "conversation_ref",
        "actor_ref",
        "internal id",
        "confidence",
        "revision",
        "openai_api_key",
        "[redacted]",
    )
    lower = text.lower()
    return any(marker in lower for marker in markers)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
