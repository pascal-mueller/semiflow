from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node  # Avoid circular import issues


def add_backward(grad_output: Node, a, b) -> List[Node]:
    # Local derivatives: df/da = 1, df/db = 1
    grad_a = grad_output * 1.0
    grad_b = grad_output * 1.0

    return [grad_a, grad_b]


def sub_backward(grad_output: Node, a, b) -> List[Node]:
    # Local derivatives: df/da = 1, df/db = -1
    grad_a = grad_output * 1.0
    grad_b = grad_output * -1.0

    return [grad_a, grad_b]


def mul_backward(grad_output: Node, a, b) -> List[Node]:
    # Local derivatives: df/da = b, df/db = a
    grad_a = grad_output * b
    grad_b = grad_output * a

    return [grad_a, grad_b]
