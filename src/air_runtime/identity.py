"""Identity resolution is explicit and separate from admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .models import ActorContext, InteractionEnvelope, ModelValidationError, safe_ref


class IdentityResolutionError(ModelValidationError):
    pass


class IdentityCollisionError(IdentityResolutionError):
    pass


class IdentityResolver(Protocol):
    def resolve(self, envelope: InteractionEnvelope) -> ActorContext:
        ...


@dataclass(frozen=True)
class StaticIdentityResolver:
    """Small local resolver used by the harness and deterministic tests."""

    principals: Mapping[str, str]
    profile_id: str = "air-default"

    def __post_init__(self) -> None:
        safe_ref(self.profile_id, "profile_id")
        seen: dict[str, str] = {}
        for principal, actor in self.principals.items():
            safe_ref(principal, "external principal")
            safe_ref(actor, "actor_ref")
            prior = seen.get(actor)
            if prior is not None and prior != principal:
                raise IdentityCollisionError(
                    f"multiple external principals cannot share actor {actor!r}"
                )
            seen[actor] = principal

    def resolve(self, envelope: InteractionEnvelope) -> ActorContext:
        actor = self.principals.get(envelope.actor_ref)
        if actor is None:
            raise IdentityResolutionError("unknown actor principal")
        return ActorContext(
            actor_ref=actor,
            profile_id=self.profile_id,
            status="active",
            auth_source="local",
        )


@dataclass(frozen=True)
class LocalIdentityResolver:
    """Explicit local mapping where the harness actor ref is already opaque."""

    profile_id: str = "air-default"

    def resolve(self, envelope: InteractionEnvelope) -> ActorContext:
        return ActorContext(
            actor_ref=envelope.actor_ref,
            profile_id=self.profile_id,
            status="active",
            auth_source="local",
        )
