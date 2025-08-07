import jax
import jax.numpy as jnp

from semiflow.nn.parameter import Parameter


class Adam:
    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        foreach: bool = False,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool = False,
        decoupled_weight_decay: bool = False,
    ):
        self.params = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.maximize = maximize
        self.weight_decay = weight_decay
        self.num_called = 0

        self.m_old = [0] * len(self.params)  # First moment
        self.v_old = [0] * len(self.params)  # Second moment

    def step(self):
        self.num_called += 1

        for i, param in enumerate(self.params):
            grads = param.grads

            if self.maximize:
                # g_t = - nabla_params f_t(param_t-1)
                grads = -1.0 * grads

            # TODO: Implement weight decay
            self.m_old[i] = (
                self.betas[0] * self.m_old[i] + (1.0 - self.betas[0]) * grads
            )
            self.v_old[i] = (
                self.betas[1] * self.v_old[i] + (1.0 - self.betas[1]) * grads * grads
            )

            # bias-corrected first moment estimate
            # TODO: Understand this.
            m_hat = self.m_old[i] / (1.0 - jnp.power(self.betas[1], self.num_called))

            # TODO: implement amsgrad
            v_hat = self.v_old[i] / (1.0 - jnp.power(self.betas[1], self.num_called))

            new_weight = param.data - self.lr * m_hat / (jnp.sqrt(v_hat) + self.eps)

            assert self.params[i].data.shape == new_weight.shape, f"""
                    New weight has different shape than old weight. Probably a braodcasting issue.
                    self.params[i].data.shape = {self.params[i].data.shape}
                    new_weight.shape = {new_weight.shape}
                    self.params[i].name = {self.params[i].name}
                """

            self.params[i].data = new_weight

    def zero_grads(self):
        for param in self.params:
            param.zero_grad()
