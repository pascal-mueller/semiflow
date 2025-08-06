import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semiflow.node import Node

from semiflow.functions import relu_backward
from semiflow.gradFunction import GradFunction


def relu(x: "Node") -> "Node":
    """
    Apply ReLU activation function

    Args:
        x: Input node

    Returns:
        Node with ReLU applied
    """
    from semiflow.node import Node  # Import here to avoid circular imports

    relu_output = jnp.maximum(0, x.data)

    result = Node(
        data=relu_output,
        parents=[x],
        dtype=x.data.dtype,
        device=x.data.device,
        requires_grad=x.requires_grad,
    )

    if x.requires_grad:
        result.grad_fn = GradFunction(
            backward_fn=relu_backward,
            input_nodes=[x],
        )

    return result
