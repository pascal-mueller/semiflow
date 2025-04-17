from semiflow import Node

x = Node([1, 2, 3])
y = Node([4, 5, 6])
z = Node([7, 8, 9])

# Works
r: Node = x + y + z

print(r)

breakpoint()

# Fails
# print(x + 2)
