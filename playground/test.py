from semiflow import Node


# y = w * x => 4 = w * 2 => w = 2
w = Node([0.1], name="w")
x = Node([2.0], name="x")
y = Node([4.0], name="y")


lr = 0.1
for epoch in range(10):
    y_pred = w * x
    L = (y_pred - y) * (y_pred - y)
    L.backward()
    w.data = w.data - lr * w.grads
    print(
        f"epoch {epoch}: w.data = {w.data[0]:.4f}, w.grads = {w.grads[0]:.4f} loss = {L.data[0]:.4f}"
    )
    L.zero_grad()
