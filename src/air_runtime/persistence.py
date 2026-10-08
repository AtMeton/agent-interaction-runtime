"""Minimal public Actor event/state/send adapter with SQLite transactions."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

from .actor import ActorAction, ActorContext, ActorEvent, ActorTurnResult, ConversationDraft, ConversationWorldState, PendingSend
from .models import RecentTurn


class PersistenceError(RuntimeError):
    pass


class DuplicateEvent(PersistenceError):
    pass


class StaleRevisionConflict(PersistenceError):
    pass


class SQLiteActorStore:
    """Fresh local schema; no private AIR database or default path is used."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        if path is None or not os.fspath(path).strip():
            raise ValueError("an explicit database path is required")
        self.path = os.fspath(path)
        self._db = sqlite3.connect(self.path, timeout=2.0, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS actor_state ("
            "conversation_ref TEXT NOT NULL, actor_ref TEXT NOT NULL, revision INTEGER NOT NULL, "
            "state_json TEXT NOT NULL, PRIMARY KEY(conversation_ref, actor_ref));"
            "CREATE TABLE IF NOT EXISTS actor_event_dedup ("
            "dedup_ref TEXT PRIMARY KEY, event_ref TEXT NOT NULL, actor_ref TEXT NOT NULL, "
            "conversation_ref TEXT NOT NULL, result_json TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS actor_turn ("
            "sequence INTEGER PRIMARY KEY, event_ref TEXT NOT NULL, conversation_ref TEXT NOT NULL, "
            "actor_ref TEXT NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL, "
            "created_at_utc TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS actor_send_ledger ("
            "send_ref TEXT PRIMARY KEY, event_ref TEXT NOT NULL, conversation_ref TEXT NOT NULL, "
            "actor_ref TEXT NOT NULL, content TEXT NOT NULL, send_state TEXT NOT NULL, "
            "attempt_count INTEGER NOT NULL, external_ref TEXT, error_code TEXT);"
        )
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def ensure_actor_state(self, conversation_ref: str, actor: ActorContext, now: str) -> None:
        empty = ConversationWorldState.empty(conversation_ref, actor.actor_ref, now)
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO actor_state VALUES (?, ?, 0, ?)",
                (conversation_ref, actor.actor_ref, self._state_json(empty)),
            )
            self._db.commit()

    def load_actor_state(self, conversation_ref: str, actor_ref: str) -> ConversationWorldState:
        with self._lock:
            row = self._db.execute(
                "SELECT state_json FROM actor_state WHERE conversation_ref=? AND actor_ref=?",
                (conversation_ref, actor_ref),
            ).fetchone()
        if row is None:
            raise PersistenceError("actor state is not initialized")
        return self._state_from_dict(json.loads(row["state_json"]))

    def recent_turns(self, conversation_ref: str, actor_ref: str, limit: int = 16) -> tuple[RecentTurn, ...]:
        with self._lock:
            rows = self._db.execute(
                "SELECT event_ref, role, text, status, created_at_utc FROM actor_turn "
                "WHERE conversation_ref=? AND actor_ref=? ORDER BY sequence DESC LIMIT ?",
                (conversation_ref, actor_ref, limit),
            ).fetchall()
        return tuple(
            RecentTurn(role=row["role"], text=row["text"], created_at_utc=row["created_at_utc"], status=row["status"], turn_ref=row["event_ref"])
            for row in reversed(rows)
        )

    def get_actor_event_result(self, event: ActorEvent, actor: ActorContext) -> ActorTurnResult | None:
        with self._lock:
            row = self._db.execute(
                "SELECT actor_ref, conversation_ref, result_json FROM actor_event_dedup WHERE dedup_ref=?",
                (event.dedup_ref,),
            ).fetchone()
        if row is None:
            return None
        if row["actor_ref"] != actor.actor_ref or row["conversation_ref"] != event.conversation_ref:
            raise DuplicateEvent("Actor dedup reference collision")
        return self._result_from_dict(json.loads(row["result_json"]), deduplicated=True)

    def commit_turn(self, event: ActorEvent, actor: ActorContext, previous: ConversationWorldState, new_state: ConversationWorldState, result: ActorTurnResult) -> ActorTurnResult:
        if new_state.revision != previous.revision + 1:
            raise StaleRevisionConflict("Actor world revision must advance exactly once")
        if (new_state.conversation_ref, new_state.actor_ref) != (event.conversation_ref, actor.actor_ref):
            raise PersistenceError("Actor state scope mismatch")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                duplicate = self._db.execute(
                    "SELECT actor_ref, conversation_ref, result_json FROM actor_event_dedup WHERE dedup_ref=?",
                    (event.dedup_ref,),
                ).fetchone()
                if duplicate is not None:
                    if duplicate["actor_ref"] != actor.actor_ref or duplicate["conversation_ref"] != event.conversation_ref:
                        raise DuplicateEvent("Actor dedup reference collision")
                    self._db.execute("ROLLBACK")
                    return self._result_from_dict(json.loads(duplicate["result_json"]), deduplicated=True)
                row = self._db.execute(
                    "SELECT revision FROM actor_state WHERE conversation_ref=? AND actor_ref=?",
                    (event.conversation_ref, actor.actor_ref),
                ).fetchone()
                if row is None or int(row["revision"]) != previous.revision:
                    raise StaleRevisionConflict("stale Actor world revision")
                self._db.execute(
                    "INSERT INTO actor_turn(event_ref,conversation_ref,actor_ref,role,text,status,created_at_utc) VALUES(?,?,?,?,?,?,?)",
                    (event.event_ref, event.conversation_ref, actor.actor_ref, "user", event.text, result.status, event.timestamp_utc),
                )
                pending = new_state.pending_send
                if pending is not None and (previous.pending_send is None or pending.send_ref != previous.pending_send.send_ref):
                    self._db.execute(
                        "INSERT INTO actor_send_ledger VALUES(?,?,?,?,?,?,?,?,?)",
                        (pending.send_ref, event.event_ref, event.conversation_ref, actor.actor_ref, pending.content, pending.status, pending.attempt_count, pending.external_ref, pending.last_error),
                    )
                    self._db.execute(
                        "INSERT INTO actor_turn(event_ref,conversation_ref,actor_ref,role,text,status,created_at_utc) VALUES(?,?,?,?,?,?,?)",
                        (f"assistant:{pending.send_ref}", event.conversation_ref, actor.actor_ref, "assistant", pending.content, pending.status, pending.created_at_utc),
                    )
                updated = self._db.execute(
                    "UPDATE actor_state SET revision=?, state_json=? WHERE conversation_ref=? AND actor_ref=? AND revision=?",
                    (new_state.revision, self._state_json(new_state), event.conversation_ref, actor.actor_ref, previous.revision),
                )
                if updated.rowcount != 1:
                    raise StaleRevisionConflict("Actor state compare-and-swap failed")
                self._db.execute(
                    "INSERT INTO actor_event_dedup VALUES(?,?,?,?,?)",
                    (event.dedup_ref, event.event_ref, actor.actor_ref, event.conversation_ref, self._result_json(result)),
                )
                self._db.execute("COMMIT")
                return result
            except Exception:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    def record_delivery(self, event: ActorEvent, actor: ActorContext, result: ActorTurnResult) -> ActorTurnResult:
        pending = result.world_state.pending_send
        if pending is None:
            raise PersistenceError("delivery result has no send identity")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                state_row = self._db.execute(
                    "SELECT state_json FROM actor_state WHERE conversation_ref=? AND actor_ref=?",
                    (event.conversation_ref, actor.actor_ref),
                ).fetchone()
                ledger = self._db.execute(
                    "SELECT actor_ref, conversation_ref FROM actor_send_ledger WHERE send_ref=?",
                    (pending.send_ref,),
                ).fetchone()
                if state_row is None or ledger is None or ledger["actor_ref"] != actor.actor_ref or ledger["conversation_ref"] != event.conversation_ref:
                    raise PersistenceError("delivery send identity or scope is missing")
                stored = self._state_from_dict(json.loads(state_row["state_json"]))
                if stored.pending_send is None or stored.pending_send.send_ref != pending.send_ref:
                    raise StaleRevisionConflict("delivery no longer matches current pending send")
                if stored.revision != result.world_state.revision:
                    raise StaleRevisionConflict("delivery result has a stale Actor revision")
                self._db.execute(
                    "UPDATE actor_send_ledger SET send_state=?,attempt_count=?,external_ref=?,error_code=? WHERE send_ref=?",
                    (pending.status, pending.attempt_count, pending.external_ref, pending.last_error, pending.send_ref),
                )
                self._db.execute(
                    "UPDATE actor_turn SET status=? WHERE event_ref=? AND conversation_ref=? AND actor_ref=?",
                    (pending.status, f"assistant:{pending.send_ref}", event.conversation_ref, actor.actor_ref),
                )
                updated = self._db.execute(
                    "UPDATE actor_state SET state_json=? WHERE conversation_ref=? AND actor_ref=? AND revision=?",
                    (self._state_json(result.world_state), event.conversation_ref, actor.actor_ref, stored.revision),
                )
                if updated.rowcount != 1:
                    raise StaleRevisionConflict("delivery state compare-and-swap failed")
                result_update = self._db.execute(
                    "UPDATE actor_event_dedup SET result_json=? WHERE dedup_ref=? AND actor_ref=? AND conversation_ref=?",
                    (self._result_json(result), event.dedup_ref, actor.actor_ref, event.conversation_ref),
                )
                if result_update.rowcount != 1:
                    raise PersistenceError("delivery event identity is missing")
                self._db.execute("COMMIT")
                return result
            except Exception:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    @staticmethod
    def _state_json(state: ConversationWorldState) -> str:
        return json.dumps({
            "conversation_ref": state.conversation_ref, "actor_ref": state.actor_ref,
            "revision": state.revision, "current_event_ref": state.current_event_ref,
            "current_event_kind": state.current_event_kind, "current_event_text": state.current_event_text,
            "current_event_at_utc": state.current_event_at_utc,
            "draft": state.draft.to_dict() if state.draft else None,
            "pending_send": SQLiteActorStore._pending_dict(state.pending_send),
            "updated_at_utc": state.updated_at_utc,
        }, separators=(",", ":"))

    @staticmethod
    def _pending_dict(pending: PendingSend | None) -> dict[str, Any] | None:
        if pending is None:
            return None
        return {name: getattr(pending, name) for name in (
            "send_ref", "content", "source_event_ref", "world_revision", "draft_revision", "status",
            "attempt_count", "created_at_utc", "updated_at_utc", "external_ref", "last_error", "conversation_ref",
        )}

    @staticmethod
    def _state_from_dict(value: dict[str, Any]) -> ConversationWorldState:
        draft = ConversationDraft.from_dict(value["draft"]) if value.get("draft") else None
        pending = PendingSend(**value["pending_send"]) if value.get("pending_send") else None
        return ConversationWorldState(
            conversation_ref=value["conversation_ref"], actor_ref=value["actor_ref"], revision=value["revision"],
            current_event_ref=value.get("current_event_ref"), current_event_kind=value.get("current_event_kind"),
            current_event_text=value.get("current_event_text", ""), current_event_at_utc=value.get("current_event_at_utc"),
            draft=draft, pending_send=pending, updated_at_utc=value["updated_at_utc"],
        )

    @classmethod
    def _result_json(cls, result: ActorTurnResult) -> str:
        value = {
            "event_ref": result.event_ref, "event_kind": result.event_kind, "status": result.status,
            "action": {"action": result.action.action, "content": result.action.content,
                       "expected_world_revision": result.action.expected_world_revision,
                       "expected_draft_revision": result.action.expected_draft_revision,
                       "reconsider_at": result.action.reconsider_at},
            "world_state": json.loads(cls._state_json(result.world_state)),
            "delivery_status": result.delivery_status, "error_code": result.error_code,
            "deduplicated": result.deduplicated, "provider_called": result.provider_called,
            "deliberation_count": result.deliberation_count,
            "base_world_revision": result.base_world_revision,
            "final_world_revision": result.final_world_revision,
        }
        return json.dumps(value, separators=(",", ":"))

    @classmethod
    def _result_from_dict(cls, value: dict[str, Any], *, deduplicated: bool) -> ActorTurnResult:
        action = ActorAction(**value["action"])
        return ActorTurnResult(
            event_ref=value["event_ref"], event_kind=value["event_kind"], status=value["status"], action=action,
            world_state=cls._state_from_dict(value["world_state"]), delivery_status=value["delivery_status"],
            error_code=value.get("error_code"), deduplicated=deduplicated,
            provider_called=value.get("provider_called", False), deliberation_count=value.get("deliberation_count", 0),
            base_world_revision=value.get("base_world_revision"), final_world_revision=value.get("final_world_revision"),
        )
