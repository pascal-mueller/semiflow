from __future__ import annotations
from typing import Optional, List

import torch

from .functions import add_backward, sub_backward, mul_backward
from .gradFunction import GradFunction


class Node:
    def __init__(
        self,
        data: torch.Tensor,
        name: Optional[str] = None,
        operation: Optional[str] = None,  # Still needed?
        parents: Optional[List[Node]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: Optional[bool] = False,
        pin_memory: Optional[bool] = False,  # do I need this?
    ) -> None:
        self.tensor: torch.tensor = torch.as_tensor(data, dtype=dtype, device=device)
        self.operation: Optional[str] = operation
        self.name: str = name or ""
        self.parents: List[Node] = parents or []
        self.grads: Optional[torch.Tensor] = None
        self.grad_fn: Optional[GradFunction] = None
        self.requires_grad: bool = requires_grad
        self.pin_memory: bool = pin_memory  # do I need this?

    def __repr__(self) -> str:
        return f"{self.name} - {str(self.tensor)}"

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
        Backpropagate through computation graph to compute gradients.
        Uses a non‑recursive post‑order DFS to topologically sort the graph.

        Note: We call the node backward() was called on the "L" node.
        Note: We use a non-recursive approach to avoid recursion limit issues.
        """
        # STEP 1: topologically sort the graph
        graph_topo = []
        visited = set()
        stack = [self]  # start from L

        while stack:
            # Get the last node in the stack
            node = stack[-1]

            # Note not yet explored
            if node not in visited:
                # Mark nodes as explored
                visited.add(node)

                # Add parents of visited node to stack so they are marked as
                # to be visited in the future. Note that we didn't visit the
                # parents yet.
                for parent in node.parents:
                    if parent not in visited:
                        stack.append(parent)
            else:
                # We are done with the current node, so pop it from the stack.
                stack.pop()
                # Add it to the topological ordered graph
                graph_topo.append(node)

        # Step 2: Backpropagation
        # dL/dL = 1 - preserving shape
        self.grads = torch.ones_like(self.tensor)

        for node in reversed(graph_topo):
            # grad_fn is only set for nodes that are a result of an opeartion
            # and e.g. not for the input nodes.
            if node.grad_fn is None:  #  or node.grads is None: <- when use this?
                continue

            # get local gradients
            # TODO: Make this flexible for arbitrary grad_fn signatures
            out_grads = node.grad_fn(node.grads, *node.parents)

            for parent, grad in zip(node.parents, out_grads):
                # Initially parent.grads is None.
                if parent.grads is None:
                    parent.grads = grad.clone()
                # Chain rule implies accumulation of gradients.
                else:
                    parent.grads = parent.grads + grad
