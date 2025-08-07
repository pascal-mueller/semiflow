from __future__ import annotations
from typing import List, TYPE_CHECKING
import jax
import jax.numpy as jnp

from semiflow.nn.activation.sigmoid import sigmoid

if TYPE_CHECKING:
    from .node import Node  # Avoid circular import issues


# TODO: Make sure we handle batch dim reduction correctly.
# Notably the sum(axis=0) will reduce dims, so we might need atleast_1d or something?
def add_backward(grad_output: jax.Array, a, b) -> List[jax.Array]:
    # Local derivatives: df/da = 1, df/db = 1
    grad_a = grad_output * 1.0
    grad_b = grad_output * 1.0

    return [grad_a, jnp.atleast_1d(grad_b.sum(axis=0))]


def sub_backward(grad_output: jax.Array, a, b) -> List[jax.Array]:
    # Local derivatives: df/da = 1, df/db = -1
    grad_a = grad_output * 1.0
    grad_b = grad_output * -1.0

    # Use keepdims=True to maintain at least 1D shape
    return [grad_a, grad_b]


# TODO: Check correctness of this backward function
def div_backward(grad_output: jax.Array, a: Node, b: Node) -> List[jax.Array]:
    # Local derivatives: df/da = 1/b, df/db = -a/(b^2)
    grad_a = grad_output / b.data
    grad_b = -grad_output * a.data / (b.data**2)

    if grad_a.ndim == 0:
        return [grad_a, grad_b]

    # Use keepdims=True to maintain at least 1D shape
    return [grad_a, grad_b]


def mul_backward(grad_output: jax.Array, a: Node, b: Node) -> List[jax.Array]:
    # Local derivatives: df/da = b, df/db = a
    grad_a = grad_output * b.data
    grad_b = grad_output * a.data

    if grad_a.ndim == 0:
        return [grad_a, grad_b]

    # Use keepdims=True to maintain at least 1D shape
    return [grad_a, grad_b]


def matmul_backward(grad_output, input_A, input_B):
    """
    Backward pass for matrix multiplication Y = A @ B

    Args:
        grad_output: ∂L/∂Y - gradient flowing back from the output
        input_A: The first input matrix A
        input_B: The second input matrix B

    Returns:
        tuple: (grad_A, grad_B) - gradients with respect to inputs
    """

    # ∂L/∂A = (∂L/∂Y) @ B^T
    grad_A = grad_output @ input_B.data.T

    # ∂L/∂B = A^T @ (∂L/∂Y)
    grad_B = input_A.data.T @ grad_output

    return (grad_A, grad_B)


# TODO: Check correctness of this backward function
def T_backward(grad_output, input_node):
    """
    Backward pass for transpose operation.

    Args:
        grad_output: The gradient flowing back (dL/dY where Y = X.T)
        input_node: The original input node X

    Returns:
        List containing the gradient for the input (dL/dX)
    """
    grad_result = grad_output.T

    if grad_result.ndim == 0:
        return [grad_result]

    return [grad_result]


def transpose_backward(grad_output: jax.Array, dim0: int, dim1: int) -> List[jax.Array]:
    """Backward pass for transpose operation"""
    grad_result = jnp.swapaxes(grad_output, dim0, dim1)

    if grad_result.ndim == 0:
        return [grad_result]

    return [grad_result]


# TODO: Check correctness of this backward function
def reshape_backward(grad_output, input_node):
    grad_result = grad_output.reshape(input_node.data.shape)

    if grad_result.ndim == 0:
        return [grad_result]

    return [grad_result]


# TODO: Check correctness of this backward function
def softmax_backward(
    grad_output: jax.Array, input_node: Node, dim: int
) -> List[jax.Array]:
    """
    Backward pass for softmax operation.

    For softmax: y_i = exp(x_i) / Σ exp(x_j)
    The Jacobian is: ∂y_i/∂x_j = y_i * (δ_ij - y_j)
    where δ_ij is the Kronecker delta (1 if i==j, 0 otherwise)

    This simplifies to: grad_x = y * (grad_output - Σ(y * grad_output))

    Args:
        grad_output: Gradient flowing back
        input_node: Original input node
        dim: Dimension along which softmax was applied

    Returns:
        List containing gradient for input
    """
    # Get the softmax output (we need to recompute it)
    x_max = jnp.max(input_node.data, axis=dim, keepdims=True)
    x_shifted = input_node.data - x_max
    exp_x = jnp.exp(x_shifted)
    sum_exp = jnp.sum(exp_x, axis=dim, keepdims=True)
    softmax_output = exp_x / sum_exp

    # Compute gradient using the softmax Jacobian formula
    sum_term = jnp.sum(softmax_output * grad_output, axis=dim, keepdims=True)
    grad_input = softmax_output * (grad_output - sum_term)

    if grad_input.ndim == 0:
        return [grad_input]

    return [grad_input]


# TODO: Check correctness of this backward function
def mean_backward(
    grad_output: jax.Array, input_node: Node, axis=None, keepdims=False
) -> List[jax.Array]:
    """Backward pass for mean operation"""
    if axis is None:
        # Mean over all elements
        scale = 1.0 / input_node.data.size
        return [jnp.full_like(input_node.data, grad_output.item() * scale)]
    else:
        # Mean over specific axis
        input_shape = input_node.data.shape
        scale = 1.0 / input_shape[axis]
        # Expand grad_output back to original shape
        if not keepdims:
            expanded_grad = jnp.expand_dims(grad_output, axis)
        else:
            expanded_grad = grad_output
        return [jnp.broadcast_to(expanded_grad * scale, input_shape)]


# TODO: Check correctness of this backward function
def std_backward(
    grad_output: jax.Array, input_node: Node, axis=None, keepdims=False
) -> List[jax.Array]:
    """Backward pass for standard deviation operation"""
    # For std backward: d(std)/dx = (x - mean) / (std * (N-1))
    # Where N is the number of elements being reduced

    mean = jnp.mean(input_node.data, axis=axis, keepdims=True)
    std = jnp.std(input_node.data, axis=axis, keepdims=True)

    # Avoid division by zero
    std = jnp.where(std == 0, 1e-8, std)

    # Number of elements in reduction
    if axis is None:
        N = input_node.data.size
    else:
        N = input_node.data.shape[axis]

    # Gradient computation
    diff = input_node.data - mean
    grad_input = diff / (std * N)

    # Handle keepdims
    if not keepdims and axis is not None:
        expanded_grad = jnp.expand_dims(grad_output, axis)
    else:
        expanded_grad = grad_output

    # Broadcast to input shape
    grad_input = grad_input * jnp.broadcast_to(expanded_grad, input_node.data.shape)

    if grad_input.ndim == 0:
        return [grad_input]

    return [grad_input]


# TODO: Check correctness of this backward function
def log_backward(grad_output: jax.Array, input_node: Node) -> List[jax.Array]:
    """Backward pass for natural logarithm: d/dx(log(x)) = 1/x"""
    grad_input = grad_output / input_node.data

    if grad_input.ndim == 0:
        return [grad_input]

    return [grad_input]


# TODO: Check correctness of this backward function
def sum_backward(
    grad_output: jax.Array, input_node: Node, axis=None, keepdims=False
) -> List[jax.Array]:
    """Backward pass for sum operation"""
    if axis is None:
        # Sum over all elements - broadcast scalar gradient to input shape
        grad_input = jnp.broadcast_to(grad_output, input_node.data.shape)
    else:
        # Sum over specific axis - expand gradient back to input shape
        if not keepdims:
            expanded_grad = jnp.expand_dims(grad_output, axis)
        else:
            expanded_grad = grad_output
        grad_input = jnp.broadcast_to(expanded_grad, input_node.data.shape)

    if grad_input.ndim == 0:
        return [grad_input]

    return [grad_input]


# TODO: Check correctness of this backward function
def relu_backward(grad_output: jax.Array, input_node: Node) -> List[jax.Array]:
    """Backward pass for ReLU activation"""
    # ReLU derivative: 1 if x > 0, else 0
    mask = input_node.data > 0
    grad_result = grad_output * mask

    if grad_result.ndim == 0:
        return [grad_result]

    return [grad_result]


# TODO: Check correctness of this backward function
def sigmoid_backward(grad_output, input_node):
    """
    Backward pass for sigmoid function.

    Args:
        grad_output: Gradient flowing back from the output
        input_node: The input node containing the original input

    Returns:
        List containing the gradient for the input
    """
    # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
    sigmoid_x = sigmoid(input_node.data)
    grad_input = grad_output * sigmoid_x * (1.0 - sigmoid_x)

    return [grad_input]
