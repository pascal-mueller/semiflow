from __future__ import annotations
from typing import Optional, List

import torch


class Node:
    def __init__(
        self,
        data,
        operation: Optional[str] = None,
        parents: Optional[List[Node]] = None,
    ) -> None:
        self.tensor: torch.tensor = torch.tensor(data)
        self.operation: Optional[str] = operation
        self.parents: List[Node] = parents or []

    def __repr__(self) -> str:
        return str(self.tensor)

    def __add__(self, other_node) -> Node:
        if isinstance(other_node, Node):
            return Node(
                self.tensor + other_node.tensor,
                operation="add",
                parents=[self, other_node],
            )
        else:
            raise TypeError(
                f"Unsupported type for addition with Node. Can't add {type(other_node)} to Node."
            )
