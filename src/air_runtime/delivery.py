"""Local-only delivery port. Receipts are simulations, not external sends."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from .actor import DeliveryPort, DeliveryReceipt, PendingSend


class FakeDelivery(DeliveryPort):
    def __init__(self, receipts: Iterable[DeliveryReceipt]) -> None:
        self._receipts = deque(receipts)
        self.attempts: list[PendingSend] = []

    async def send(self, pending_send: PendingSend) -> DeliveryReceipt:
        self.attempts.append(pending_send)
        if not self._receipts:
            raise RuntimeError("FakeDelivery script exhausted")
        receipt = self._receipts.popleft()
        if not isinstance(receipt, DeliveryReceipt):
            raise TypeError("FakeDelivery requires explicit DeliveryReceipt values")
        return receipt
