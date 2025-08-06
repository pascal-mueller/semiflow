import jax
import jax.numpy as jnp

from semiflow.nn.parameter import Parameter


# TODO: Add docstring
class Dropout:
    def __init__(self, p: float, key: jax.Array | None = None):
        super().__init__()

        self.p = p
        if key is None:
            key = jax.random.PRNGKey(0)
        self.key: jax.Array = key

    # TODO: Maybe we put the training to the module level somehow?
    def __call__(self, x: jnp.ndarray, training: bool = True):
        if not training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p

        self.key, subkey = jax.random.split(self.key)
        mask = jax.random.bernoulli(subkey, keep_prob, x.shape)

        return x * mask / keep_prob


# TODO: Add example
