import jax
import jax.numpy as jnp

from semiflow.node import Node
from semiflow.nn.module import Module
from semiflow.nn.optimizers.adam import Adam
from semiflow.nn.linear import Linear


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(1, 10)
        self.linear2 = Linear(10, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.sigmoid()

        return x


def make_data(n_samples=1000):
    # Generate integers from 1 to 10
    x = jax.random.randint(jax.random.PRNGKey(42), (n_samples, 1), 1, 11)
    y = (x > 5).astype(jnp.int32)

    return Node(x), Node(y)


if __name__ == "__main__":
    model = MLP()
    optimizer = Adam(model.parameters(), lr=0.001)
    x, y = make_data(10000)

    for epoch in range(1, 100):
        output = model(x)
        loss = ((output - y) * (output - y)).sum()
        loss.backward()

        optimizer.step()

        sum_grads = 0.0
        sum_weights = 0.0
        for param in model.parameters():
            sum_grads += param.grads.sum()
            sum_weights += param.data.sum()

        sum_output = 0.0
        for i, item in enumerate(output.data):
            sum_output += item.sum()

        optimizer.zero_grads()

        print(
            f"Epoch {epoch}: {loss.data:.6f}  Summed Grads: {sum_grads:.6f}  Summed Output: {sum_output:.6f}  Summed Weights: {sum_weights:.6f}"
        )

    # Run some tests
    x1 = Node(jnp.array([[7]]))  # Shape: (1, 1) - batch_size=1, features=1
    x2 = Node(jnp.array([[2]]))  # Shape: (1, 1) - batch_size=1, features=1
    result1 = model(x1).data[0] > 0.5
    result2 = model(x2).data[0] > 0.5

    print(f"{x1.data[0][0]} > 5 is {result1[0]}")
    print(f"{x2.data[0][0]} > 5 is {result2[0]}")

    # Run a big test
    x, y = make_data(1234)

    output = model(x)
    breakpoint()
    predictions = (output.data > 0.5).astype(jnp.int32)
    correct = (predictions == y.data).astype(jnp.int32)
    num_correct = correct.sum()
    total = correct.shape[0]
    accuracy = num_correct / total
    print(f"Accuracy: {accuracy:.2f} ({num_correct}/{total})")
