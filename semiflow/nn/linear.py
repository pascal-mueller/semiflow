import jax
from jax._src.typing import DTypeLike
import jax.numpy as jnp
import numpy as np

from semiflow.node import Node
from semiflow.nn.module import Module
from semiflow.nn.parameter import Parameter


# TODO: Improve the docstring
class Linear(Module):
    r"""
    Linear (fully connected) layer implementation.

    Applies a linear transformation to the incoming data: :math:`y = xW^T + b`

    For input :math:`x \in \mathbb{R}^{N \times in\_features}` and parameters:

    * :math:`W \in \mathbb{R}^{out\_features \times in\_features}` (weights)
    * :math:`b \in \mathbb{R}^{out\_features}` (bias)

    The output is computed as:

    .. math::
       y = xW^T + b

    Where :math:`y \in \mathbb{R}^{N \times out\_features}`

    Parameters
    ----------
    in_features : int
        Size of each input sample
    out_features : int
        Size of each output sample
    dtype : jnp.dtype, optional
        Data type for weights and bias (default: jnp.float32)
    bias : bool, optional
        If set to False, the layer will not learn an additive bias (default: True)

    Attributes
    ----------
    weights : Parameter
        The learnable weights of shape :math:`(out\_features, in\_features)`
    bias : Parameter or None
        The learnable bias of shape :math:`(out\_features,)`. Only present if bias=True.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from semiflow.node import Node
    >>> layer = Linear(in_features=3, out_features=2)
    >>> input_node = Node(jnp.array([[1.0, 2.0, 3.0]]))
    >>> output = layer(input_node)
    >>> print(output.data.shape)
    (1, 2)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: jnp.dtype = jax.numpy.float32,
        bias: bool = True,
    ):
        super().__init__()

        self.weights: Parameter = Parameter(
            jnp.ones((out_features, in_features), dtype=dtype)
        )

        self.bias: Parameter | None = None
        if bias:
            self.bias = Parameter(jnp.zeros((out_features,), dtype=dtype))

    def forward(self, x: Node) -> Node:
        r"""
        Forward pass of the linear layer.

        Parameters
        ----------
        x : Node
            Input node with data of shape :math:`(N, in\_features)`

        Returns
        -------
        Node
            Output node with data of shape :math:`(N, out\_features)`
        """
        if self.bias is not None:
            return x @ self.weights.T + self.bias
        return x @ self.weights.T


# TODO: Improe the docstring
def example():
    r"""
    Linear Layer Example and Test
    =============================

    This example demonstrates the Linear layer with concrete mathematical calculations.

    Theory
    ------
    For a linear layer :math:`y = xW^T + b` with:

    - Input: :math:`x = [1, 2]` (shape: :math:`1 \times 2`)
    - Weights: :math:`W = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}` (shape: :math:`2 \times 2`)
    - Bias: :math:`b = [0, 0]` (shape: :math:`2`)

    Step-by-step calculation:

    .. math::
       y_1 = w_{11} \cdot x_1 + w_{12} \cdot x_2 + b_1 = 1 \cdot 1 + 1 \cdot 2 + 0 = 3

    .. math::
       y_2 = w_{21} \cdot x_1 + w_{22} \cdot x_2 + b_2 = 1 \cdot 1 + 1 \cdot 2 + 0 = 3

    Therefore: :math:`y = [3, 3]`

    Loss: :math:`L = y_1 + y_2 = 6`

    Expected Gradients
    ------------------

    .. math::
       \frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_2} = 2

    .. math::
       \frac{\partial L}{\partial W} = \begin{bmatrix} 3 & 3 \\ 3 & 3 \end{bmatrix}

    .. math::
       \frac{\partial L}{\partial b} = [1, 1]

    Example Output
    --------------
    Input: [[1 2]]

    Weights: [[1. 1.] [1. 1.]]

    Bias: [0. 0.]

    Output: [[3. 3.]]

    Loss: 6.0
    """

    input: Node = Node(jnp.array([[1, 2]]))

    linear_layer: Linear = Linear(2, 2)
    output: Node = linear_layer(input)
    loss = jnp.sum(output.data)

    print("Input:")
    print(input.data)
    print("\nWeights:")
    print(linear_layer.weights.data)
    print("\nBias:")
    print(linear_layer.bias.data)
    print("\nOutput:")
    print(output.data)
    print("\nLoss:")
    print(loss)


if __name__ == "__main__":
    example()
