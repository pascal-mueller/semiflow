from semiflow import Node


# y = w * x
# 4 = w * 2
# w = 2 <-- wanna find this
x = Node([2.0], name="x")
y = Node([4.0], name="y")
w = Node([0.1], name="w", requires_grad=True)  # initial guess

# Goal: turn w into 2.0
lr = 0.1
for epoch in range(10):
    # TODO: This should be the model
    y_pred = w * x
    L = (y_pred - y) * (y_pred - y)
    L.backward()
    # TODO: This should be the optimizer (step)
    w.data = w.data - lr * w.grads
    print(
        f"epoch {epoch}: w.data = {w.data[0]:.4f}, w.grads = {w.grads[0]:.4f} loss = {L.data[0]:.4f}"
    )
    # This should be optimzer.zero_grad()
    L.zero_grad()
    x.zero_grad()
    y.zero_grad()
    w.zero_grad()

print(f"Final w: {w.data[0]:.4f}, expectation: 2.0")

# TODO
# 1. model
# 2. model.parameters() (generator, items of type torch.nn.parameter.Parameter)
# 3. Optimizer
# 4. Loss function

"""
    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""
