from __future__ import annotations
from typing import Callable, List, TYPE_CHECKING
import jax.numpy as jnp


if TYPE_CHECKING:
    from .node import Node  # Avoid circular import issues


class GradFunction:
    def __init__(
        self,
        backward_fn: Callable,
        input_nodes: List[Node],
    ):
        self.backward: Callable = backward_fn
        # TODO: I think this is not needed?
        self.input_nodes: List[Node] = input_nodes

    def __call__(self, *args, **kwargs):
        result = self.backward(*args, **kwargs)

        return result

    def __repr__(self):
        return f"<GradFunction: {self.backward.__name__}>"
