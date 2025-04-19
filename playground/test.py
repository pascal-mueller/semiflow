from semiflow import Node

a = Node([1])
b = Node([2])
c = Node([3])
d = Node([4])

h1 = a + b
h2 = b + c
h3 = h1 + h2
h4 = h1 - h2
L = h3 + h4

print(f"Result: {L}")

breakpoint()
