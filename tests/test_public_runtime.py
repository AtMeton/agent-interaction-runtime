from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from air_runtime import ActorAction, AIRRuntime, DeliveryReceipt, FakeActorMind, FakeDelivery, InteractionEnvelope, SQLiteActorStore
from air_runtime.actor import ActorContext, ActorEvent, ActorTurnResult
from air_runtime.persistence import StaleRevisionConflict


def envelope(number: int, text: str = "A synthetic message.") -> InteractionEnvelope:
    return InteractionEnvelope.local(
        event_ref=f"event-{number}", conversation_ref="conversation-test", actor_ref="actor-test",
        dedup_ref=f"dedup-{number}", text=text,
        timestamp_utc=f"2026-10-07T00:00:{number:02d}Z",
    )


def runtime(path: Path | str, actions, receipts=()):
    store = SQLiteActorStore(path)
    mind = FakeActorMind(actions)
    delivery = FakeDelivery(receipts)
    return AIRRuntime(mind, delivery, store), store, mind, delivery


class PublicRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_ingress_has_one_main_mind_call_and_no_action_is_committed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            app, store, mind, delivery = runtime(Path(folder) / "fresh.sqlite3", [ActorAction(action="NO_ACTION")])
            result = await app.handle(envelope(1))
            self.assertEqual(mind.calls, 1)
            self.assertEqual(result.deliberation_count, 1)
            self.assertEqual(result.action.action, "NO_ACTION")
            self.assertEqual(result.status, "no_action")
            self.assertEqual(result.commit_status, "COMMITTED")
            self.assertEqual(delivery.attempts, [])
            self.assertEqual(store.load_actor_state("conversation-test", "actor-test").revision, 1)
            store.close()

    async def test_request_send_commits_before_fake_delivery_and_keeps_truth_separate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteActorStore(Path(folder) / "fresh.sqlite3")
            mind = FakeActorMind([ActorAction(action="REQUEST_SEND", content="A local expression.")])

            class CommitObservingDelivery(FakeDelivery):
                async def send(inner_self, pending):
                    stored = store.get_actor_event_result(envelope(1), ActorContext("actor-test", "air-default"))
                    self.assertIsNotNone(stored)
                    self.assertEqual(stored.status, "pending_delivery")
                    return await super(CommitObservingDelivery, inner_self).send(pending)

            delivery = CommitObservingDelivery([DeliveryReceipt(status="DELIVERED", external_ref="local-sim:1")])
            app = AIRRuntime(mind, delivery, store)
            result = await app.handle(envelope(1))
            self.assertEqual(mind.calls, 1)
            self.assertEqual(len(delivery.attempts), 1)
            self.assertEqual(delivery.attempts[0].status, "SEND_REQUESTED")
            self.assertEqual(result.action.action, "REQUEST_SEND")
            self.assertEqual(result.commit_status, "COMMITTED")
            self.assertEqual(result.status, "delivered")
            self.assertEqual(result.delivery_status, "DELIVERED")
            self.assertEqual(result.world_state.pending_send.external_ref, "local-sim:1")
            store.close()

    async def test_draft_actions_remain_distinct_and_send_uses_draft_revision(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            app, store, mind, delivery = runtime(
                Path(folder) / "fresh.sqlite3",
                [ActorAction(action="SET_DRAFT", content="Saved proposal."), ActorAction(action="REQUEST_SEND", expected_world_revision=2, expected_draft_revision=1)],
                [DeliveryReceipt(status="DELIVERED", external_ref="local-sim:2")],
            )
            draft_result = await app.handle(envelope(1, "Save this synthetic draft."))
            self.assertEqual(draft_result.status, "draft_saved")
            self.assertEqual(draft_result.world_state.draft.revision, 1)
            sent = await app.handle(envelope(2, "Send the saved draft."))
            self.assertEqual(sent.action.action, "REQUEST_SEND")
            self.assertEqual(sent.world_state.pending_send.content, "Saved proposal.")
            self.assertEqual(sent.world_state.pending_send.draft_revision, 1)
            self.assertIsNone(sent.world_state.draft)
            self.assertEqual(mind.calls, 2)
            self.assertEqual(len(delivery.attempts), 1)
            store.close()

    async def test_consequence_gate_blocks_false_completion_without_rewriting_action(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            app, store, mind, delivery = runtime(
                Path(folder) / "fresh.sqlite3",
                [ActorAction(action="REQUEST_SEND", content="The upload is completed.")],
            )
            result = await app.handle(envelope(1, "Please tell me when the upload is completed."))
            self.assertEqual(mind.calls, 1)
            self.assertEqual(result.status, "consequence_truth_blocked")
            self.assertEqual(result.action.action, "REQUEST_SEND")
            self.assertIsNone(result.world_state.pending_send)
            self.assertEqual(delivery.attempts, [])
            self.assertEqual(result.commit_status, "COMMITTED")
            store.close()

    async def test_replay_is_deduplicated_without_second_cognition_or_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            app, store, mind, delivery = runtime(
                Path(folder) / "fresh.sqlite3",
                [ActorAction(action="REQUEST_SEND", content="One local expression.")],
                [DeliveryReceipt(status="DELIVERED", external_ref="local-sim:3")],
            )
            incoming = envelope(1)
            first = await app.handle(incoming)
            replay = await app.handle(incoming)
            self.assertFalse(first.deduplicated)
            self.assertTrue(replay.deduplicated)
            self.assertEqual(replay.delivery_status, "DELIVERED")
            self.assertEqual(mind.calls, 1)
            self.assertEqual(len(delivery.attempts), 1)
            store.close()

    async def test_restart_recovers_revision_and_recent_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fresh.sqlite3"
            first, store, mind, delivery = runtime(
                path, [ActorAction(action="REQUEST_SEND", content="Persisted local expression.")],
                [DeliveryReceipt(status="DELIVERED", external_ref="local-sim:4")],
            )
            await first.handle(envelope(1, "First synthetic event."))
            store.close()
            resumed, resumed_store, resumed_mind, _ = runtime(path, [ActorAction(action="NO_ACTION")])
            result = await resumed.handle(envelope(2, "Second synthetic event."))
            self.assertEqual(result.world_state.revision, 2)
            self.assertEqual(resumed_mind.calls, 1)
            self.assertEqual([turn.role for turn in resumed_mind.requests[0].recent_turns], ["user", "assistant"])
            self.assertEqual(resumed_mind.requests[0].recent_turns[0].text, "First synthetic event.")
            self.assertEqual(mind.calls, 1)
            self.assertEqual(len(delivery.attempts), 1)
            resumed_store.close()

    async def test_unknown_delivery_is_persisted_and_replay_never_retries_it(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            app, store, mind, delivery = runtime(
                Path(folder) / "fresh.sqlite3",
                [ActorAction(action="REQUEST_SEND", content="One uncertain local send.")],
                [DeliveryReceipt(status="DELIVERY_UNKNOWN", error="simulated_unknown")],
            )
            incoming = envelope(1)
            first = await app.handle(incoming)
            replay = await app.handle(incoming)
            self.assertEqual(first.delivery_status, "DELIVERY_UNKNOWN")
            self.assertEqual(replay.delivery_status, "DELIVERY_UNKNOWN")
            self.assertEqual(mind.calls, 1)
            self.assertEqual(len(delivery.attempts), 1)
            store.close()

    async def test_sqlite_revision_cas_rejects_a_stale_second_commit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteActorStore(Path(folder) / "fresh.sqlite3")
            actor = ActorContext("actor-test", "air-default")
            incoming_a, incoming_b = envelope(1), envelope(2)
            event_a, event_b = ActorEvent.from_envelope(incoming_a, actor), ActorEvent.from_envelope(incoming_b, actor)
            store.ensure_actor_state(event_a.conversation_ref, actor, event_a.timestamp_utc)
            previous = store.load_actor_state(event_a.conversation_ref, actor.actor_ref)
            next_a, next_b = previous.for_event(event_a), previous.for_event(event_b)
            result_a = ActorTurnResult(event_a.event_ref, event_a.kind, "no_action", ActorAction(action="NO_ACTION"), next_a, deliberation_count=1)
            result_b = ActorTurnResult(event_b.event_ref, event_b.kind, "no_action", ActorAction(action="NO_ACTION"), next_b, deliberation_count=1)
            store.commit_turn(event_a, actor, previous, next_a, result_a)
            with self.assertRaises(StaleRevisionConflict):
                store.commit_turn(event_b, actor, previous, next_b, result_b)
            store.close()

    async def test_fake_script_exhaustion_is_a_failure_not_valid_no_action(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            app, store, mind, delivery = runtime(Path(folder) / "fresh.sqlite3", [])
            result = await app.handle(envelope(1))
            self.assertEqual(mind.calls, 1)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.error_code, "main_mind_failure")
            self.assertEqual(result.action.action, "NO_ACTION")
            self.assertEqual(result.delivery_status, "NOT_ATTEMPTED")
            self.assertEqual(delivery.attempts, [])
            store.close()


class PublicSurfaceTests(unittest.TestCase):
    def test_interaction_envelope_is_part_of_the_supported_package_root_api(self) -> None:
        import air_runtime

        expected = {
            "ActorAction",
            "ActorDeliberationRequest",
            "ActorEvent",
            "ActorMindPort",
            "ActorTurnResult",
            "AIRRuntime",
            "ConversationWorldState",
            "DeliveryReceipt",
            "FakeActorMind",
            "FakeDelivery",
            "InteractionEnvelope",
            "SQLiteActorStore",
        }
        self.assertEqual(set(air_runtime.__all__), expected)
        self.assertIs(air_runtime.InteractionEnvelope, InteractionEnvelope)

    def test_readme_programmatic_example_runs_using_the_installed_root_api(self) -> None:
        readme = Path(__file__).resolve().parents[1] / "README.md"
        text = readme.read_text(encoding="utf-8")
        start = text.index("<!-- programmatic-example:start -->")
        end = text.index("<!-- programmatic-example:end -->", start)
        section = text[start:end]
        match = re.search(r"```python\s*(.*?)\s*```", section, flags=re.DOTALL)
        self.assertIsNotNone(match)
        assert match is not None
        exec(compile(match.group(1), str(readme), "exec"), {"__name__": "__main__"})

    def test_module_help_uses_the_importable_module_name(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "air_runtime", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertTrue(result.stdout.startswith("usage: python -m air_runtime"))


if __name__ == "__main__":
    unittest.main()
