# Agent Interaction Runtime

Agent Interaction Runtime (AIR) is a small Python runtime slice for
applications that need to keep a Main Mind's proposed expression distinct from consequence
validation, durable local commit, and delivery truth. It is more than an LLM
wrapper: the selected Actor path composes those boundaries around one explicit
interaction. Public v0 is the `ActorAction` vertical slice; it is **not** a
complete mirror of AIR's private default chat command or a claim of parity.

## Selected public flow

```text
InteractionEnvelope
  -> AIRRuntime + caller-supplied ActorMindPort
  -> ActorAction proposal
  -> consequence validation
  -> durable local event/state/send-intent commit
  -> separate delivery receipt
```

The included demo supplies a scripted `FakeActorMind` and local `FakeDelivery`.
Applications may provide their own Main Mind and delivery adapters; those
caller-supplied components, not this package, determine whether an application
contacts a service.

## Quickstart

Package metadata uses the distribution name `agent-interaction-runtime` at version
`0.1.0rc1`; Python code imports `air_runtime`.

This release candidate has not been published to PyPI. After an authorized
publication, install by distribution name with:

```console
python -m pip install agent-interaction-runtime
```

From a fresh source checkout, use the validated `python -m pip install .`
command below.

Python 3.12 is required. From a fresh source checkout, create and activate a
virtual environment, then install the project and run the local demo:

```console
python -m pip install .
python -m air_runtime demo
```

The source-checkout install command above is validated. To run the public test
suite from that checkout after installation:

```console
python -m unittest discover -s tests -v
```

## Programmatic example

This synthetic example uses only the supported package-root API, temporary
SQLite state, and local fakes. It makes no external send.

<!-- programmatic-example:start -->
```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from air_runtime import (
    AIRRuntime,
    ActorAction,
    DeliveryReceipt,
    FakeActorMind,
    FakeDelivery,
    InteractionEnvelope,
    SQLiteActorStore,
)


async def main() -> None:
    with TemporaryDirectory() as directory:
        store = SQLiteActorStore(Path(directory) / "example.sqlite3")
        runtime = AIRRuntime(
            FakeActorMind([
                ActorAction(action="REQUEST_SEND", content="A local example response.")
            ]),
            FakeDelivery([
                DeliveryReceipt(status="DELIVERED", external_ref="local-example-1")
            ]),
            store,
        )
        try:
            result = await runtime.handle(
                InteractionEnvelope.local(
                    event_ref="example-event-1",
                    conversation_ref="example-conversation",
                    actor_ref="example-actor",
                    dedup_ref="example-dedup-1",
                    text="Please show a synthetic example.",
                )
            )
            print(f"{result.status}: {result.delivery_status}")
        finally:
            store.close()


asyncio.run(main())
```
<!-- programmatic-example:end -->

Expected output:

```text
delivered: DELIVERED
```

## Supported API and observable contract

The exact supported v0 package-root surface is `air_runtime.__all__`:

- `ActorAction`
- `ActorDeliberationRequest`
- `ActorEvent`
- `ActorMindPort`
- `ActorTurnResult`
- `AIRRuntime`
- `ConversationWorldState`
- `DeliveryReceipt`
- `FakeActorMind`
- `FakeDelivery`
- `InteractionEnvelope`
- `SQLiteActorStore`

`InteractionEnvelope` is supported from `air_runtime`. Other submodule-level
names are implementation details or experimental unless explicitly listed in
this root API. The 0.x API may change.

- `NO_ACTION` is a valid committed result after one deliberation.
- `REQUEST_SEND` is a proposal, not proof of commit or delivery.
- `SET_DRAFT` and `CLEAR_DRAFT` remain distinct actions.
- A consequence gate may block a proposal without rewriting its action.
- SQLite commit/replay identity is separate from a later delivery receipt.
- `FakeActorMind` must be explicitly scripted; exhaustion raises instead of
  manufacturing `NO_ACTION`.

This curated projection retains bounded identity/admission, one deliberation,
action preconditions, an atomic event/state/send-intent commit, deduplication,
and post-commit delivery status. It omits private experimental controls,
workers, channels, migrations, identity maps, multi-user production policy, and
the private default chat entry point.

## Network and installation boundaries

The package has no runtime dependencies beyond Python's standard library. The
included demo and core do not contact the network by default; an application-
supplied port may do so. A first source-checkout installation is **not promised
to be offline**: pip may need to resolve the pinned `setuptools==84.0.0` build
backend from an available package source. Separately, the RC build/install gate
was verified with index access disabled and the pinned backend supplied from a
locally verified wheel. That build artifact is not bundled with this project.

## License

Apache-2.0. See `LICENSE`.

This RC does not include a real Provider, credential handling, QQ, NapCat,
OneBot, GMA, Android, Generic Capability execution, internal evals, production
readiness, world-action execution, or human-acceptance claims. A simulated
`DELIVERED` receipt means only that the fake adapter returned that status.
SQLite is opt-in and requires an explicit caller-owned path; tests use fresh
temporary files. The release-gate run was validated on CPython 3.12 on Windows;
other platforms have not been validated here.
