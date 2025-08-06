import jax.numpy as jnp

from semiflow.nn.module import Module
from semiflow.nn.parameter import Parameter


class LayerNormalization(Module):
    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()

        self.eps = eps
        self.alpha = Parameter(jnp.ones(features))  # alpha is a learnable parameter
        self.bias = Parameter(jnp.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(axis=-1, keepdims=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(axis=-1, keepdims=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
