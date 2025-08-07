import jax.numpy as jnp


def sigmoid(x):
    """
    Forward pass for sigmoid function.

    Args:
        x: Input array

    Returns:
        Sigmoid of x: 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + jnp.exp(-x))
