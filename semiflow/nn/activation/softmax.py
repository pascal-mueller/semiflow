import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semiflow.node import Node

from semiflow.functions import softmax_backward
from semiflow.gradFunction import GradFunction


# TODO: Check correctness of this whole file
def softmax(x: "Node", dim: int = -1) -> "Node":
    """
    Apply softmax activation along specified dimension

    Args:
        x: Input node
        dim: Dimension to apply softmax along

    Returns:
        Node with softmax applied
    """
    from semiflow.node import Node  # Import here to avoid circular imports

    # Subtract max for numerical stability
    x_max = jnp.max(x.data, axis=dim, keepdims=True)
    x_shifted = x.data - x_max

    # Compute softmax
    exp_x = jnp.exp(x_shifted)
    sum_exp = jnp.sum(exp_x, axis=dim, keepdims=True)
    softmax_output = exp_x / sum_exp

    result = Node(
        data=softmax_output,
        parents=[x],
        dtype=x.data.dtype,
        device=x.data.device,
        requires_grad=x.requires_grad,
    )

    if x.requires_grad:
        result.grad_fn = GradFunction(
            backward_fn=softmax_backward,
            input_nodes=[x],
        )

    return result
