"""Admission policy after identity resolution and before interaction."""

from __future__ import annotations

from .models import ActorContext, InteractionEnvelope, ModelValidationError


class AdmissionError(ModelValidationError):
    pass


class LocalAdmissionPolicy:
    'local runtime admission: active local actor and no explicit denial metadata.'

    def admit(self, actor: ActorContext, envelope: InteractionEnvelope) -> None:
        if actor.status != "active":
            raise AdmissionError("actor is not active")
        if envelope.channel_kind != "local_harness":
            raise AdmissionError('local runtime admits only the local harness')
        if envelope.addressing.get("admit") is False:
            raise AdmissionError("interaction was denied by admission policy")
