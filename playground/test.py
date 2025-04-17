from semiflow import Node

x = Node([1, 2, 3])
y = Node([1, 2, 3])

# Works
print(x + y)

# Fails
# print(x + 2)
