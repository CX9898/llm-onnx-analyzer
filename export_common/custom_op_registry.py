from __future__ import annotations

from collections.abc import Callable


CustomOutputSpecs = list[tuple[list[int] | None, int | None]]
CustomOutputResolver = Callable[[str, str, list[list[int] | None], list[int | None]], CustomOutputSpecs]


class CustomOpRegistry:
    """Model adapters register custom-op shape/type rules here."""

    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], Callable[[list[list[int] | None], list[int | None]], CustomOutputSpecs]] = {}

    def register(
        self,
        *,
        domain: str,
        op_type: str,
        handler: Callable[[list[list[int] | None], list[int | None]], CustomOutputSpecs],
    ) -> None:
        self._handlers[(domain, op_type)] = handler

    def resolve(
        self,
        domain: str,
        op_type: str,
        input_shapes: list[list[int] | None],
        input_elem_types: list[int | None],
    ) -> CustomOutputSpecs:
        handler = self._handlers.get((domain, op_type))
        if handler is None:
            return []
        return handler(input_shapes, input_elem_types)

