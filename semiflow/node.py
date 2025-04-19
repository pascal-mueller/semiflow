from __future__ import annotations
from typing import Optional, List, Callable, TYPE_CHECKING

import torch

from .functions import add_backward, sub_backward, mul_backward

# Include guard for type checking
if TYPE_CHECKING:
    from .gradFunction import GradFunction


class Node:
    def __init__(
        self,
        data: torch.Tensor,
        operation: Optional[str] = None,  # Still needed?
        parents: Optional[List[Node]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        pin_memory: Optional[bool] = False,  # do I need this?
    ) -> None:
        self.tensor: torch.tensor = torch.as_tensor(data, dtype=dtype, device=device)
        self.operation: Optional[str] = operation
        self.parents: List[Node] = parents or []
        self.grads: Optional[torch.Tensor] = None
        self.grad_fn: Optional[GradFunction] = None
        self.requires_grad: bool = requires_grad
        self.pin_memory: bool = pin_memory  # do I need this?

    def __repr__(self) -> str:
        return str(self.tensor)

    def __add__(self, other_node) -> Node:
        if not isinstance(other_node, Node):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't add {type(other_node)} to Node."
            )

        result: torch.Tensor = self.tensor + other_node.tensor

        result = Node(
            result,
            operation="add",
            parents=[self, other_node],
            dtype=result.dtype,
            device=result.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
            pin_memory=result.pin_memory,
        )

        result.grad_fn = GradFunction(
            backward_fn=add_backward,
            input_nodes=[self, other_node],
        )

        return result

    def __sub__(self, other_node) -> Node:
        if not isinstance(other_node, Node):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't subtract {type(other_node)} to Node."
            )

        result: torch.Tensor = self.tensor - other_node.tensor

        result = Node(
            result,
            operation="sub",
            parents=[self, other_node],
            dtype=result.dtype,
            device=result.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
            pin_memory=result.pin_memory,
        )

        result.grad_fn = GradFunction(
            backward_fn=sub_backward,
            input_nodes=[self, other_node],
        )

        return result

    def __mul__(self, other_node) -> Node:
        if not isinstance(other_node, (int, torch.Tensor)):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't multiply {type(other_node)} with Node."
            )

        result: torch.Tensor = self.tensor * other_node

        result = Node(
            result,
            operation="mul",
            parents=[self, other_node],
            dtype=result.dtype,
            device=result.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
            pin_memory=result.pin_memory,
        )

        result.grad_fn = GradFunction(
            backward_fn=mul_backward,
            input_nodes=[self, other_node],
        )

        return result

    def __rmul__(self, other_node) -> Node:
        if not isinstance(other_node, (int, torch.Tensor)):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't multiply {type(other_node)} with Node."
            )
        result: torch.Tensor = self.tensor * other_node

        result = Node(
            result,
            operation="mul",
            parents=[self, other_node],
            dtype=result.dtype,
            device=result.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
            pin_memory=result.pin_memory,
        )

        result.grad_fn = GradFunction(
            backward_fn=mul_backward,
            input_nodes=[self, other_node],
        )

        return result

    def backward(self) -> None:
        """
        Backpropagation through the graph.
        """

        # Assume it's already topologically sorted  for now.
        pass
