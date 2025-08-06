import math
import jax
import jax.numpy as jnp

from semiflow.nn.module import Module
from semiflow.nn.dropout import Dropout


class PositionalEncoding(Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = Dropout(0.5)
        # Create a matrix of shape (seq_len, d_model)
        pe = jnp.zeros((seq_len, d_model))

        # Create a vector of shape (seq_len, 1)
        position = jnp.arange(0, seq_len, dtype=jnp.float32).reshape((seq_len, 1))

        # Create a vector of shape (d_model // 2,)
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2, dtype=jnp.float32)
            * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        # Apply cosine to odd indices (handle odd d_model case)
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[: pe.shape[1] // 2]))

        # Add a batch dimension to the positional encoding
        pe = pe[jnp.newaxis, :, :]  # (1, seq_len, d_model)
        self.pe = pe

    def forward(self, x: jax.Array) -> jax.Array:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.shape[1], :]
        # For dropout, use your own dropout implementation or a JAX/Flax/Haiku one
        return x  # (batch, seq_len, d_model)
