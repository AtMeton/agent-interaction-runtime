from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence
from .models import ActorContext, DialogueState, InteractionEnvelope, MAX_RESPONSE_CHARS, ModelValidationError, ProtectedFacts, RecentTurn, bounded_text, safe_ref, utc_now, validate_utc_text
from .dialogue import analyze_turn
from .consequence import ConsequenceTruthGate
ACTOR_ACTIONS = frozenset({'NO_ACTION', 'SET_DRAFT', 'CLEAR_DRAFT', 'REQUEST_SEND'})

@dataclass(frozen=True)
class ActorAction:
    action: str
    content: str | None = None
    expected_world_revision: int | None = None
    expected_draft_revision: int | None = None
    reconsider_at: str | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTOR_ACTIONS:
            raise ModelValidationError('invalid Actor action')
        if self.content is not None:
            bounded_text(self.content, MAX_RESPONSE_CHARS, 'Actor action content')
        if self.action == 'SET_DRAFT' and (not self.content):
            raise ModelValidationError('SET_DRAFT requires content')
        if self.action == 'CLEAR_DRAFT' and self.content is not None:
            raise ModelValidationError('CLEAR_DRAFT cannot carry content')
        for attribute, name in (('expected_world_revision', 'expected_world_revision'), ('expected_draft_revision', 'expected_draft_revision')):
            value = getattr(self, attribute)
            normalized = _optional_revision(value, name)
            if normalized != value:
                object.__setattr__(self, attribute, normalized)
        _optional_utc(self.reconsider_at, 'reconsider_at')
        if self.action == 'REQUEST_SEND' and self.reconsider_at is not None:
            raise ModelValidationError('REQUEST_SEND cannot schedule reconsideration')

def _optional_ref(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return safe_ref(value, field_name)

def _optional_revision(value: Any, field_name: str) -> int | None:
    """Normalize legacy provider revision metadata without trusting it."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ModelValidationError(f'{field_name} must be a non-negative integer or null')
    if isinstance(value, int):
        if value < 0:
            raise ModelValidationError(f'{field_name} must be a non-negative integer or null')
        return value
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip())
    raise ModelValidationError(f'{field_name} must be a non-negative integer or null')

def _optional_utc(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_utc_text(value, field_name)

@dataclass(frozen=True)
class ConversationDraft:
    draft_ref: str
    content: str
    revision: int
    source_event_ref: str
    source_world_revision: int
    created_at_utc: str
    updated_at_utc: str
    reconsider_at: str | None = None

    def __post_init__(self) -> None:
        safe_ref(self.draft_ref, 'draft_ref')
        bounded_text(self.content, MAX_RESPONSE_CHARS, 'draft content')
        if self.revision < 1:
            raise ModelValidationError('draft revision must be positive')
        safe_ref(self.source_event_ref, 'draft source_event_ref')
        if self.source_world_revision < 0:
            raise ModelValidationError('draft source_world_revision cannot be negative')
        validate_utc_text(self.created_at_utc, 'draft created_at_utc')
        validate_utc_text(self.updated_at_utc, 'draft updated_at_utc')
        _optional_utc(self.reconsider_at, 'draft reconsider_at')

    def to_dict(self) -> dict[str, Any]:
        return {'draft_ref': self.draft_ref, 'content': self.content, 'revision': self.revision, 'source_event_ref': self.source_event_ref, 'source_world_revision': self.source_world_revision, 'created_at_utc': self.created_at_utc, 'updated_at_utc': self.updated_at_utc, 'reconsider_at': self.reconsider_at}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'ConversationDraft':
        return cls(draft_ref=str(value['draft_ref']), content=str(value['content']), revision=int(value['revision']), source_event_ref=str(value['source_event_ref']), source_world_revision=int(value['source_world_revision']), created_at_utc=str(value['created_at_utc']), updated_at_utc=str(value['updated_at_utc']), reconsider_at=value.get('reconsider_at'))

@dataclass(frozen=True)
class DeliveryReceipt:
    status: str
    external_ref: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {'DELIVERED', 'DELIVERY_FAILED', 'DELIVERY_UNKNOWN'}:
            raise ModelValidationError('invalid delivery receipt status')
        _optional_ref(self.external_ref, 'delivery external_ref')
        if self.error is not None:
            bounded_text(self.error, 240, 'delivery error')

class ActorMindPort(Protocol):

    async def deliberate(self, request: ActorDeliberationRequest) -> ActorAction:
        ...

class DeliveryPort(Protocol):

    async def send(self, pending_send: PendingSend) -> DeliveryReceipt:
        ...

def _protected_facts(revision: int, *, side_effect_requested: bool) -> Any:
    from .models import ProtectedFacts
    return ProtectedFacts(response_intent='conversation', side_effect_requested=side_effect_requested, execution_outcome='not_executed' if side_effect_requested else 'not_requested', state_revision=revision)

def _event_requests_side_effect(event: ActorEvent) -> bool:
    if event.kind != 'USER_MESSAGE':
        return False
    state = DialogueState.empty(event.conversation_ref, event.timestamp_utc)
    hints = analyze_turn(event.text, state)
    return bool(getattr(hints, 'consequence_request_candidate', False) or ConsequenceTruthGate.has_outcome_inducement(event.text))

@dataclass(frozen=True)
class ActorEvent:
    kind: str
    event_ref: str
    dedup_ref: str
    conversation_ref: str
    actor_ref: str
    timestamp_utc: str
    text: str = ''

    def __post_init__(self) -> None:
        if self.kind != 'USER_MESSAGE':
            raise ModelValidationError('public Actor slice accepts USER_MESSAGE only')
        safe_ref(self.event_ref, 'event_ref')
        safe_ref(self.dedup_ref, 'dedup_ref')
        safe_ref(self.conversation_ref, 'conversation_ref')
        safe_ref(self.actor_ref, 'actor_ref')
        validate_utc_text(self.timestamp_utc, 'timestamp_utc')
        object.__setattr__(self, 'text', bounded_text(self.text, 2000, 'Actor event text'))

    @classmethod
    def from_envelope(cls, envelope: InteractionEnvelope, actor: ActorContext) -> ActorEvent:
        return cls(kind='USER_MESSAGE', event_ref=envelope.event_ref, dedup_ref=envelope.dedup_ref, conversation_ref=envelope.conversation_ref, actor_ref=actor.actor_ref, timestamp_utc=envelope.timestamp_utc, text=envelope.text)

@dataclass(frozen=True)
class PendingSend:
    send_ref: str
    content: str
    source_event_ref: str
    world_revision: int
    draft_revision: int | None
    status: str
    attempt_count: int
    created_at_utc: str
    updated_at_utc: str
    external_ref: str | None = None
    last_error: str | None = None
    conversation_ref: str | None = None

    def __post_init__(self) -> None:
        safe_ref(self.send_ref, 'send_ref')
        safe_ref(self.source_event_ref, 'source_event_ref')
        bounded_text(self.content, MAX_RESPONSE_CHARS, 'pending send content')
        if self.status not in {'COMMITTED_PENDING_DELIVERY', 'SEND_REQUESTED', 'DELIVERED', 'DELIVERY_FAILED', 'DELIVERY_UNKNOWN'}:
            raise ModelValidationError('invalid public single-send status')
        if self.world_revision < 0 or self.attempt_count < 0:
            raise ModelValidationError('pending send counters cannot be negative')
        validate_utc_text(self.created_at_utc, 'send created_at_utc')
        validate_utc_text(self.updated_at_utc, 'send updated_at_utc')

    @property
    def blocks_future_actor_work(self) -> bool:
        return self.status != 'DELIVERED'

@dataclass(frozen=True)
class ConversationWorldState:
    conversation_ref: str
    actor_ref: str
    revision: int = 0
    current_event_ref: str | None = None
    current_event_kind: str | None = None
    current_event_text: str = ''
    current_event_at_utc: str | None = None
    draft: ConversationDraft | None = None
    pending_send: PendingSend | None = None
    updated_at_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        safe_ref(self.conversation_ref, 'conversation_ref')
        safe_ref(self.actor_ref, 'actor_ref')
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ModelValidationError('world revision must be a non-negative integer')
        validate_utc_text(self.updated_at_utc, 'world updated_at_utc')

    @classmethod
    def empty(cls, conversation_ref: str, actor_ref: str, now: str) -> ConversationWorldState:
        return cls(conversation_ref=conversation_ref, actor_ref=actor_ref, updated_at_utc=now)

    def for_event(self, event: ActorEvent) -> ConversationWorldState:
        return replace(self, revision=self.revision + 1, current_event_ref=event.event_ref, current_event_kind=event.kind, current_event_text=event.text, current_event_at_utc=event.timestamp_utc, updated_at_utc=event.timestamp_utc)

@dataclass(frozen=True)
class ActorDeliberationRequest:
    event: ActorEvent
    world_state: ConversationWorldState
    recent_turns: tuple[RecentTurn, ...]
    deadline_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.world_state.conversation_ref != self.event.conversation_ref:
            raise ModelValidationError('Actor request world/event scope mismatch')
        if not 0 < self.deadline_seconds <= 60:
            raise ModelValidationError('Actor deliberation deadline must be between 0 and 60')

@dataclass(frozen=True)
class ActorTurnResult:
    event_ref: str
    event_kind: str
    status: str
    action: ActorAction
    world_state: ConversationWorldState
    delivery_status: str = 'NOT_ATTEMPTED'
    error_code: str | None = None
    deduplicated: bool = False
    provider_called: bool = False
    deliberation_count: int = 0
    base_world_revision: int | None = None
    final_world_revision: int | None = None

    @property
    def commit_status(self) -> str:
        return 'COMMITTED'
