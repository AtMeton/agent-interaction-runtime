"""Selected ordinary USER_MESSAGE composition over projected canonical Actor contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from .actor import (
    ActorAction, ActorContext, ActorDeliberationRequest, ActorEvent,
    ActorTurnResult, ConversationDraft, ConversationWorldState, DeliveryReceipt,
    PendingSend, _event_requests_side_effect, _protected_facts,
)
from .admission import LocalAdmissionPolicy
from .consequence import ConsequenceTruthGate
from .identity import LocalIdentityResolver
from .models import InteractionEnvelope, ModelValidationError
from .persistence import DuplicateEvent, PersistenceError, SQLiteActorStore


class AIRRuntime:
    def __init__(self, mind, delivery, store, identity_resolver=None, admission=None, *, deadline_seconds: float = 30.0) -> None:
        if not 0 < deadline_seconds <= 60:
            raise ValueError("deadline_seconds must be between 0 and 60")
        self.mind = mind
        self.delivery = delivery
        self.store = store
        self.identity_resolver = identity_resolver or LocalIdentityResolver()
        self.admission = admission or LocalAdmissionPolicy()
        self.deadline_seconds = deadline_seconds
        self._lanes: dict[tuple[str, str], asyncio.Lock] = {}

    async def handle(self, envelope: InteractionEnvelope) -> ActorTurnResult:
        if not isinstance(envelope, InteractionEnvelope):
            raise TypeError("AIRRuntime.handle requires an InteractionEnvelope")
        actor = self.identity_resolver.resolve(envelope)
        key = (envelope.conversation_ref, actor.actor_ref)
        lane = self._lanes.setdefault(key, asyncio.Lock())
        async with lane:
            return await self._handle_locked(envelope, actor)

    async def _handle_locked(self, envelope: InteractionEnvelope, actor: ActorContext) -> ActorTurnResult:
        self.admission.admit(actor, envelope)
        event = ActorEvent.from_envelope(envelope, actor)
        self.store.ensure_actor_state(event.conversation_ref, actor, event.timestamp_utc)
        try:
            duplicate = self.store.get_actor_event_result(event, actor)
        except DuplicateEvent:
            raise
        if duplicate is not None:
            return duplicate

        previous = self.store.load_actor_state(event.conversation_ref, actor.actor_ref)
        current = previous.for_event(event)
        recent = self.store.recent_turns(event.conversation_ref, actor.actor_ref, limit=16)
        request = ActorDeliberationRequest(
            event=event,
            world_state=current,
            recent_turns=recent,
            deadline_seconds=self.deadline_seconds,
        )
        try:
            action = await asyncio.wait_for(
                self.mind.deliberate(request), timeout=self.deadline_seconds
            )
        except asyncio.TimeoutError:
            result = self._result(event, current, ActorAction(action="NO_ACTION"), "failed", "main_mind_timeout", True)
            return self.store.commit_turn(event, actor, previous, current, result)
        except Exception:
            result = self._result(event, current, ActorAction(action="NO_ACTION"), "failed", "main_mind_failure", True)
            return self.store.commit_turn(event, actor, previous, current, result)
        if not isinstance(action, ActorAction):
            result = self._result(event, current, ActorAction(action="NO_ACTION"), "failed", "main_mind_schema_failure", True)
            return self.store.commit_turn(event, actor, previous, current, result)

        if action.expected_world_revision is not None and action.expected_world_revision != current.revision:
            result = self._result(event, current, action, "stale_action", "stale_world_revision", True)
            return self.store.commit_turn(event, actor, previous, current, result)
        draft_revision = current.draft.revision if current.draft else None
        if action.expected_draft_revision is not None and action.expected_draft_revision != draft_revision:
            result = self._result(event, current, action, "stale_action", "stale_draft_revision", True)
            return self.store.commit_turn(event, actor, previous, current, result)

        if action.action == "NO_ACTION":
            no_action_state = current
            if current.draft is not None:
                no_action_state = replace(
                    current,
                    draft=replace(current.draft, reconsider_at=action.reconsider_at, updated_at_utc=event.timestamp_utc),
                )
            result = self._result(event, no_action_state, action, "no_action", None, True)
            return self.store.commit_turn(event, actor, previous, no_action_state, result)
        if action.action == "SET_DRAFT":
            draft = ConversationDraft(
                draft_ref=f"draft-{event.event_ref}-{(current.draft.revision if current.draft else 0) + 1}",
                content=action.content,
                revision=(current.draft.revision if current.draft else 0) + 1,
                source_event_ref=event.event_ref,
                source_world_revision=current.revision,
                created_at_utc=current.draft.created_at_utc if current.draft else event.timestamp_utc,
                updated_at_utc=event.timestamp_utc,
                reconsider_at=action.reconsider_at,
            )
            new_state = replace(current, draft=draft)
            result = self._result(event, new_state, action, "draft_saved", None, True)
            return self.store.commit_turn(event, actor, previous, new_state, result)
        if action.action == "CLEAR_DRAFT":
            new_state = replace(current, draft=None)
            result = self._result(event, new_state, action, "draft_cleared", None, True)
            return self.store.commit_turn(event, actor, previous, new_state, result)

        pending_old = current.pending_send
        if pending_old is not None and pending_old.blocks_future_actor_work:
            if pending_old.status != "DELIVERY_UNKNOWN":
                result = self._result(event, current, action, "send_blocked_pending_delivery", "pending_send_exists", True)
                return self.store.commit_turn(event, actor, previous, current, result)
        content = action.content
        used_draft_revision = None
        if content is None and current.draft is not None:
            content = current.draft.content
            used_draft_revision = current.draft.revision
        if not content:
            result = self._result(event, current, action, "action_invalid", "send_content_missing", True)
            return self.store.commit_turn(event, actor, previous, current, result)

        if _event_requests_side_effect(event):
            truth = ConsequenceTruthGate.evaluate(
                text=content,
                protected_facts=_protected_facts(
                    current.revision, side_effect_requested=True
                ),
            )
            if not truth.accepted:
                result = self._result(event, current, action, "consequence_truth_blocked", "consequence_truth_violation", True)
                return self.store.commit_turn(event, actor, previous, current, result)

        send_ref = f"send-{event.event_ref}-{current.revision}"
        pending = PendingSend(
            send_ref=send_ref,
            content=content,
            source_event_ref=event.event_ref,
            world_revision=current.revision,
            draft_revision=used_draft_revision,
            status="COMMITTED_PENDING_DELIVERY",
            attempt_count=0,
            created_at_utc=event.timestamp_utc,
            updated_at_utc=event.timestamp_utc,
            conversation_ref=event.conversation_ref,
        )
        new_state = replace(current, draft=None, pending_send=pending)
        proposed = self._result(event, new_state, action, "pending_delivery", None, True)
        committed = self.store.commit_turn(event, actor, previous, new_state, proposed)
        if committed.deduplicated or committed.world_state.pending_send is None:
            return committed
        return await self._deliver_once(event, actor, committed)

    def _result(self, event, state, action, status, error_code, mind_called):
        return ActorTurnResult(
            event_ref=event.event_ref,
            event_kind=event.kind,
            status=status,
            action=action,
            world_state=state,
            delivery_status=(state.pending_send.status if state.pending_send else "NOT_ATTEMPTED"),
            error_code=error_code,
            provider_called=False,
            deliberation_count=1 if mind_called else 0,
            base_world_revision=state.revision - 1,
            final_world_revision=state.revision,
        )

    async def _deliver_once(self, event, actor, committed):
        pending = committed.world_state.pending_send
        attempting = replace(
            pending,
            status="SEND_REQUESTED",
            attempt_count=pending.attempt_count + 1,
            updated_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        )
        try:
            receipt = await asyncio.wait_for(self.delivery.send(attempting), timeout=self.deadline_seconds)
            if not isinstance(receipt, DeliveryReceipt):
                raise ModelValidationError("delivery adapter returned an invalid receipt")
            status = receipt.status
        except asyncio.TimeoutError:
            receipt = DeliveryReceipt(status="DELIVERY_UNKNOWN", error="delivery_timeout")
            status = receipt.status
        except Exception:
            receipt = DeliveryReceipt(status="DELIVERY_FAILED", error="delivery_adapter_failure")
            status = receipt.status
        updated_pending = replace(
            attempting,
            status=status,
            external_ref=receipt.external_ref,
            last_error=receipt.error,
        )
        updated_state = replace(committed.world_state, pending_send=updated_pending)
        final = replace(
            committed,
            status=status.lower(),
            delivery_status=status,
            world_state=updated_state,
        )
        return self.store.record_delivery(event, actor, final)
