from semiflow import Node

a = Node([1.1], name="a")
b = Node([2.4], name="b")
c = Node([3.8], name="c")

h1 = a + b  # 1.1 + 2.4 = 3.5
h2 = b + c  # 2.4 + 3.8 = 6.2
L = h1 + h2  # 3.5 + 6.2 = 9.7

L.backward()

print("a.grad =", a.grads)  # 1
print("b.grad =", b.grads)  # 2
print("c.grad =", c.grads)  # 1
