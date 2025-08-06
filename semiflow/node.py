from __future__ import annotations
from typing import Any, List

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from .functions import (
    add_backward,
    sub_backward,
    mul_backward,
    T_backward,
    transpose_backward,
    div_backward,
    matmul_backward,
    reshape_backward,
    mean_backward,
    std_backward,
)
from .gradFunction import GradFunction
from semiflow.nn.activation.softmax import softmax


class Node:
    def __init__(
        self,
        data: jax.Array | list[float],
        name: str | None = None,
        parents: list[Node] | None = None,
        dtype: DTypeLike | None = None,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> None:
        self.data: jax.Array = jax.device_put(
            jnp.array(data, dtype=dtype), device=device
        )
        self.name: str = name or ""
        self.parents: List[Node] = parents or []
        self.grads: jax.Array | None = None
        self.grad_fn: GradFunction | None = None
        self.requires_grad: bool = requires_grad

    def __repr__(self) -> str:
        return f"Node: {self.name} Data: {str(self.data)} Dtype: {self.data.dtype} Device: {self.data.device} Requires Grad: {self.requires_grad}"

    def zero_grad(self) -> None:
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

        for node in graph_topo:
            node.grads = None

    @property
    def T(self) -> Node:
        """Return the transposed JAX array"""
        result = Node(
            data=self.data.T,
            name=f"{self.name}.T" if self.name else None,
            parents=[self],
            requires_grad=self.requires_grad,
            dtype=self.data.dtype,
            device=self.data.device,
        )

        # You'll need to create a transpose_backward function
        result.grad_fn = GradFunction(
            backward_fn=T_backward,  # Need to implement this
            input_nodes=[self],
        )

        return result

    def transpose(self, dim0: int, dim1: int):
        """
        Transpose two dimensions (PyTorch-style)

        Args:
            dim0: First dimension to transpose
            dim1: Second dimension to transpose
        """
        transposed_data = jnp.swapaxes(self.data, dim0, dim1)

        result = Node(
            data=transposed_data,
            parents=[self],
            dtype=self.data.dtype,
            device=self.data.device,
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:
            result.grad_fn = GradFunction(
                backward_fn=transpose_backward,
                input_nodes=[self],
            )

        return result

    # TODO: Check if this is correct
    def reshape(self, *shape):
        """Reshape the node's data while maintaining gradient tracking"""
        reshaped_data = self.data.reshape(*shape)

        result = Node(
            data=reshaped_data,
            parents=[self],
            dtype=self.data.dtype,
            device=self.data.device,
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:
            result.grad_fn = GradFunction(
                backward_fn=reshape_backward,
                input_nodes=[self],
            )

        return result

    # TODO: Do we need the alias?
    def view(self, *shape):
        """Alias for reshape() - for PyTorch compatibility"""
        return self.reshape(*shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the node's data"""
        return self.data.shape

    def __getitem__(self, key: Any) -> Node:
        """Support indexing operations like node[0] or node[1:3]"""
        result = Node(
            data=self.data[key],
            name=f"{self.name}[{key}]" if self.name else None,
            parents=[self],  # TODO: Make sure this is correct
            requires_grad=self.requires_grad,  # TODO: Not sure if this is true?
            dtype=self.data.dtype,
            device=self.data.device,
        )

        return result

    def __matmul__(self, other_node: Node) -> Node:
        if not isinstance(other_node, Node):
            raise TypeError(
                f"Unsupported type for matrix multiplication with Node. Can't multiply {type(other_node)} with Node."
            )

        data: jax.Array = self.data @ other_node.data

        result = Node(
            data=data,
            parents=[self, other_node],
            dtype=data.dtype,
            device=data.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
        )

        result.grad_fn = GradFunction(
            backward_fn=matmul_backward,  # Need to implement this
            input_nodes=[self, other_node],
        )

        return result

    def __add__(self, other_node) -> Node:
        if not isinstance(other_node, (int, float, jax.Array, Node)):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't add {type(other_node)} to Node."
            )

        if isinstance(other_node, jax.Array):
            other_node = Node(other_node)

        if isinstance(other_node, (int, float)):
            other_node = Node(jnp.array(other_node))

        data: jax.Array = self.data + other_node.data

        result = Node(
            data=data,
            parents=[self, other_node],
            dtype=data.dtype,
            device=data.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
        )

        result.grad_fn = GradFunction(
            backward_fn=add_backward,
            input_nodes=[self, other_node],
        )

        return result

    def __radd__(self, other_node) -> Node:
        return self.__add__(other_node)

    def __sub__(self, other_node) -> Node:
        if not isinstance(other_node, (int, float, jax.Array, Node)):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't subtract {type(other_node)} to Node."
            )

        if isinstance(other_node, jax.Array):
            other_node = Node(other_node)

        if isinstance(other_node, (int, float)):
            other_node = Node(jnp.array(other_node))

        data: jax.Array = self.data - other_node.data

        result = Node(
            data=data,
            parents=[self, other_node],
            dtype=data.dtype,
            device=data.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
        )

        result.grad_fn = GradFunction(
            backward_fn=sub_backward,
            input_nodes=[self, other_node],
        )

        return result

    def __truediv__(self, other_node: Node | int | float | jax.Array) -> Node:
        if not isinstance(other_node, (int, float, jax.Array, Node)):
            raise TypeError(
                f"Unsupported type for division with Node. Can't divide Node by {type(other_node)}."
            )

        if isinstance(other_node, jax.Array):
            other_node = Node(other_node)

        if isinstance(other_node, (int, float)):
            other_node = Node(jnp.array(other_node))

        data: jax.Array = self.data / other_node.data

        result = Node(
            data=data,
            parents=[self, other_node],
            dtype=data.dtype,
            device=data.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
        )

        result.grad_fn = GradFunction(
            backward_fn=div_backward,  # Division can be treated as multiplication by the inverse
            input_nodes=[self, other_node],
        )

        return result

    def __rtruediv__(self, other_node: Node | int | float | jax.Array) -> Node:
        if not isinstance(other_node, (int, float, jax.Array, Node)):
            raise TypeError(
                f"Unsupported type for division with Node. Can't divide {type(other_node)} by Node."
            )

        if isinstance(other_node, jax.Array):
            other_node = Node(other_node)

        if isinstance(other_node, (int, float)):
            other_node = Node(jnp.array(other_node))

        data: jax.Array = other_node.data / self.data

        result = Node(
            data=data,
            parents=[self, other_node],
            dtype=data.dtype,
            device=data.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
        )

        result.grad_fn = GradFunction(
            backward_fn=div_backward,  # Division can be treated as multiplication by the inverse
            input_nodes=[self, other_node],
        )

        return result

    def __mul__(self, other_node: Node | int | float | jax.Array) -> Node:
        print("__mul__ called")
        if not isinstance(other_node, (int, float, jax.Array, Node)):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't multiply {type(other_node)} with Node."
            )

        if isinstance(other_node, jax.Array):
            other_node = Node(other_node)

        if isinstance(other_node, (int, float)):
            other_node = Node(jnp.array(other_node))

        data: jax.Array = self.data * other_node.data

        result = Node(
            data=data,
            parents=[self, other_node],
            dtype=data.dtype,
            device=data.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
        )

        result.grad_fn = GradFunction(
            backward_fn=mul_backward,
            input_nodes=[self, other_node],
        )

        return result

    # TODO: I turn any other_node to a node. Make sure rmul and mul are correctly implemented.
    def __rmul__(self, other_node) -> Node:
        if not isinstance(other_node, (int, float, jax.Array, Node)):
            raise TypeError(
                f"Unsupported type for addition with Node. Can't multiply {type(other_node)} with Node."
            )

        if isinstance(other_node, jax.Array):
            other_node = Node(other_node)

        if isinstance(other_node, (int, float)):
            other_node = Node(jnp.array(other_node))

        data: jax.Array = self.data * other_node.data

        result = Node(
            data=data,
            parents=[self, other_node],
            dtype=data.dtype,
            device=data.device,
            requires_grad=self.requires_grad or other_node.requires_grad,
        )

        result.grad_fn = GradFunction(
            backward_fn=mul_backward,
            input_nodes=[self, other_node],
        )

        return result

    # TODO: Check correctness of this method
    def softmax(self, dim: int = -1) -> Node:
        """Apply softmax activation along specified dimension"""
        return softmax(self, dim)

    def mean(self, axis=None, keepdims=False) -> Node:
        """Compute mean along specified axis"""
        mean_data = jnp.mean(self.data, axis=axis, keepdims=keepdims)

        result = Node(
            data=mean_data,
            parents=[self],
            dtype=self.data.dtype,
            device=self.data.device,
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:
            result.grad_fn = GradFunction(
                backward_fn=mean_backward,
                input_nodes=[self],
            )

        return result

    def std(self, axis=None, keepdims=False) -> Node:
        """Compute standard deviation along specified axis"""
        std_data = jnp.std(self.data, axis=axis, keepdims=keepdims)

        result = Node(
            data=std_data,
            parents=[self],
            dtype=self.data.dtype,
            device=self.data.device,
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:
            from .functions import std_backward

            result.grad_fn = GradFunction(
                backward_fn=std_backward,
                input_nodes=[self],
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
        self.grads = jnp.ones_like(self.data)
        for node in reversed(graph_topo):
            # grad_fn is only set for nodes that are a result of an opeartion
            # and e.g. not for the input nodes.
            if node.grad_fn is None or not node.requires_grad:
                print(node)
                print(node.grad_fn)
                continue

            # get local gradients
            # TODO: Make this flexible for arbitrary grad_fn signatures
            out_grads: jax.Array = node.grad_fn(node.grads, *node.parents)

            for parent, grad in zip(node.parents, out_grads):
                # Initially parent.grads is None.
                if parent.grads is None:
                    parent.grads = grad.clone()
                # Chain rule implies accumulation of gradients.
                else:
                    parent.grads = parent.grads + grad
