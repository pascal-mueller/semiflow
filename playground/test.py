from semiflow import Node


x = Node([1])
y = Node([2])
z = Node([3])

h1 = x + y  # 1 + 2 = 3
h2 = 2 * y + 3 * z  # 2*2 + 3*3 = 13
L = h1 + h2  # 3 + 13 = 16

print(f"Result: {L}")

breakpoint()
