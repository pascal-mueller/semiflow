import math
import jax

from semiflow.nn.module import Module
from semiflow.nn.parameter import Parameter


# TODO: Add docstring
class Embedding(Module):
    def __init__(
        self, d_model: int, vocab_size: int, key: jax.Array | None = None
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.vocab_size: int = vocab_size

        if key is None:
            key = jax.random.PRNGKey(0)
        self.key: jax.Array = key

        # Initialize embedding matrix (vocab_size, d_model)
        self.embedding: Parameter = Parameter(
            jax.random.normal(key, (vocab_size, d_model))
        )

    def forward(self, x: jax.Array) -> jax.Array:
        # x: (batch, seq_len) of token IDs
        # Output: (batch, seq_len, d_model)
        return self.embedding[x].data * math.sqrt(self.d_model)


# TODO: Add docstring
if __name__ == "__main__":
    import jax.numpy as jnp

    # Test parameters
    vocab_size = 10
    d_model = 4

    # Create embedding layer
    emb = Embedding(d_model=d_model, vocab_size=vocab_size)

    # Test single token
    token_id = 3
    single_embedding = emb(jnp.array([token_id]))

    # Test multiple tokens
    token_ids = jnp.array([0, 1, 5, 8])
    multiple_embeddings = emb(token_ids)

    # Test batch of sequences
    batch_tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
    batch_embeddings = emb(batch_tokens)

    print("Embedding layer test:")
    print(f"Vocab size: {vocab_size}, d_model: {d_model}")
    print(f"Embedding matrix shape: {emb.embedding.data.shape}")
    print()

    print(f"Single token ({token_id}) embedding shape: {single_embedding.shape}")
    print(
        f"Multiple tokens {token_ids.tolist()} embedding shape: {multiple_embeddings.shape}"
    )
    print(
        f"Batch tokens {batch_tokens.tolist()} embedding shape: {batch_embeddings.shape}"
    )
    print()

    print("Sample embedding values:")
    print(f"Token {token_id} embedding: {single_embedding[0][:2]}... (first 2 dims)")
    print(f"Embedding scaling factor: {math.sqrt(d_model)}")
