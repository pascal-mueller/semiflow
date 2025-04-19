from __future__ import annotations
from typing import Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node  # Avoid circular import issues


class GradFunction:
    def __init__(
        self,
        backward_fn: Callable,
        input_nodes: List[Node],
    ):
        self.backward: Callable = backward_fn
        self.input_nodes: List[Node] = input_nodes
