from __future__ import annotations
from typing import List, TYPE_CHECKING
import jax

if TYPE_CHECKING:
    from .node import Node  # Avoid circular import issues


def add_backward(grad_output: jax.Array, a, b) -> List[jax.Array]:
    # Local derivatives: df/da = 1, df/db = 1
    grad_a = grad_output * 1.0
    grad_b = grad_output * 1.0

    return [grad_a, grad_b]


def sub_backward(grad_output: jax.Array, a, b) -> List[jax.Array]:
    # Local derivatives: df/da = 1, df/db = -1
    grad_a = grad_output * 1.0
    grad_b = grad_output * -1.0

    return [grad_a, grad_b]


def mul_backward(grad_output: jax.Array, a: Node, b: Node) -> List[jax.Array]:
    # Local derivatives: df/da = b, df/db = a
    grad_a = grad_output * b.data
    grad_b = grad_output * a.data

    return [grad_a, grad_b]
