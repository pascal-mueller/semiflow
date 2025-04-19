from typing import Callable, List
from .node import Node


class GradFunction:
    def __init__(
        self,
        backward_fn: Callable,
        input_nodes: List[Node],
    ):
        self.backward: Callable = backward_fn
        self.input_nodes: List[Node] = input_nodes
