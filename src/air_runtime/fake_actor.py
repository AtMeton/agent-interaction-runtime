"""A scripted single-call Main Mind implementing the projected ActorMindPort."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from .actor import ActorAction, ActorDeliberationRequest, ActorMindPort


class FakeActorMind(ActorMindPort):
    def __init__(self, actions: Iterable[ActorAction]) -> None:
        self._actions = deque(actions)
        self.calls = 0
        self.requests: list[ActorDeliberationRequest] = []

    async def deliberate(self, request: ActorDeliberationRequest) -> ActorAction:
        self.calls += 1
        self.requests.append(request)
        if not self._actions:
            raise RuntimeError("FakeActorMind script exhausted")
        action = self._actions.popleft()
        if not isinstance(action, ActorAction):
            raise TypeError("FakeActorMind requires explicit ActorAction values")
        return action
