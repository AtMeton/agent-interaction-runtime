"""Deterministic local Actor-slice demo."""

from __future__ import annotations

import argparse
import asyncio

from .actor import ActorAction, DeliveryReceipt
from .delivery import FakeDelivery
from .fake_actor import FakeActorMind
from .identity import LocalIdentityResolver
from .admission import LocalAdmissionPolicy
from .models import InteractionEnvelope
from .persistence import SQLiteActorStore
from .runtime import AIRRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m air_runtime")
    parser.add_argument("command", choices=("demo",), help="run the local Actor slice")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "demo":
        return 2
    return asyncio.run(_run_demo())


async def _run_demo() -> int:
    store = SQLiteActorStore(":memory:")
    mind = FakeActorMind([
        ActorAction(action="REQUEST_SEND", content="The upload is completed."),
        ActorAction(action="NO_ACTION"),
        ActorAction(action="REQUEST_SEND", content="A deterministic local expression."),
    ])
    delivery = FakeDelivery([DeliveryReceipt(status="DELIVERED", external_ref="local-simulation:1")])
    runtime = AIRRuntime(mind, delivery, store, LocalIdentityResolver(), LocalAdmissionPolicy())
    inputs = (
        "Please tell me when the upload is completed.",
        "No further action is needed.",
        "Thanks for helping me prepare this example.",
    )
    results = []
    for index, text in enumerate(inputs, 1):
        envelope = InteractionEnvelope.local(
            event_ref=f"demo-event-{index}", conversation_ref="demo-conversation",
            actor_ref="demo-actor", text=text, dedup_ref=f"demo-dedup-{index}",
            timestamp_utc=f"2026-10-07T00:00:0{index}Z",
        )
        result = await runtime.handle(envelope)
        results.append(result)
        if index == 1:
            if result.status != "consequence_truth_blocked" or result.action.action != "REQUEST_SEND":
                raise RuntimeError("synthetic consequence-truth demonstration did not block as expected")
            consequence = "BLOCKED"
        elif index == 2:
            if result.status != "no_action" or result.delivery_status != "NOT_ATTEMPTED":
                raise RuntimeError("synthetic NO_ACTION demonstration did not remain a no-send result")
            consequence = "NOT_REQUIRED"
        elif result.action.action == "NO_ACTION":
            raise RuntimeError("ordinary expression demonstration unexpectedly returned NO_ACTION")
        else:
            consequence = "NOT_REQUIRED"
        if index == 3 and (result.status != "delivered" or result.delivery_status != "DELIVERED"):
            raise RuntimeError("local FakeDelivery demonstration did not return its scripted receipt")
        committed = result.final_world_revision > result.base_world_revision
        print(
            f"TURN={index}; MIND_ACTION={result.action.action}; "
            f"CONSEQUENCE_DECISION={consequence}; "
            f"COMMIT_STATUS={'COMMITTED' if committed else 'NOT_COMMITTED'}; "
            f"DELIVERY_STATUS={result.delivery_status}"
        )
    state = store.load_actor_state("demo-conversation", "demo-actor")
    if state.revision != 3 or mind.calls != 3 or len(delivery.attempts) != 1:
        raise RuntimeError("synthetic demo continuity/call-count invariant failed")
    print(
        f"CONTINUITY_REVISION={state.revision}; MAIN_MIND_CALLS={mind.calls}; "
        f"FAKE_DELIVERY_ATTEMPTS={len(delivery.attempts)}; EXTERNAL_SEND=NO"
    )
    store.close()
    return 0
