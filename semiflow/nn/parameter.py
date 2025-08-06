import jax

from semiflow.node import Node


class Parameter(Node):
    def __init__(self, data: jax.Array, name: str | None = None):
        # Initialize as a Node with requires_grad=True by default
        super().__init__(
            data=data,
            name=name or "parameter",
            requires_grad=True,  # Parameters should always be trainable
        )
