from __future__ import annotations
from typing import Optional, List

import torch


class Node:
    def __init__(
        self,
        data: torch.Tensor,
        operation: Optional[str] = None,
        parents: Optional[List[Node]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        pin_memory: Optional[bool] = False,
    ) -> None:
        self.tensor: torch.tensor = torch.as_tensor(data, dtype=dtype, device=device)
        self.operation: Optional[str] = operation
        self.parents: List[Node] = parents or []
        self.grad: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return str(self.tensor)

    def __add__(self, other_node) -> Node:
        if isinstance(other_node, Node):
            result: torch.Tensor = self.tensor + other_node.tensor

            if self.parents is not None:
                pass
            return Node(
                result,
                operation="add",
                parents=[self, other_node],
                dtype=result.dtype,
                device=result.device,
            )
        else:
            raise TypeError(
                f"Unsupported type for addition with Node. Can't add {type(other_node)} to Node."
            )

    def __sub__(self, other_node) -> Node:
        if isinstance(other_node, Node):
            result: torch.Tensor = self.tensor - other_node.tensor

            return Node(
                result,
                operation="sub",
                parents=[self, other_node],
                dtype=result.dtype,
                device=result.device,
            )
        else:
            raise TypeError(
                f"Unsupported type for addition with Node. Can't subtract {type(other_node)} to Node."
            )

    def __mul__(self, other_node) -> Node:
        if isinstance(other_node, (int, torch.Tensor)):
            result: torch.Tensor = self.tensor * other_node

            return Node(
                result,
                operation="mul",
                parents=[self, other_node],
                dtype=result.dtype,
                device=result.device,
            )
        else:
            raise TypeError(
                f"Unsupported type for addition with Node. Can't multiply {type(other_node)} with Node."
            )

    def __rmul__(self, other_node) -> Node:
        if isinstance(other_node, (int, torch.Tensor)):
            result: torch.Tensor = self.tensor * other_node

            return Node(
                result,
                operation="rmul",
                parents=[self, other_node],
                dtype=result.dtype,
                device=result.device,
            )
        else:
            raise TypeError(
                f"Unsupported type for addition with Node. Can't multiply {type(other_node)} with Node."
            )

    def backward(self) -> None:
        """
        Backpropagation through the graph.
        """

        # Assume it's already topologically sorted  for now.
        pass
