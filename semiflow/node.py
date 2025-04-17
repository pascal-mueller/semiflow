import torch


class Node:
    def __init__(self, data):
        self.tensor = torch.tensor(data)

    def __repr__(self):
        return str(self.tensor)

    def __add__(self, other_node):
        if isinstance(other_node, Node):
            return Node(self.tensor + other_node.tensor)
        else:
            raise TypeError(
                f"Unsupported type for addition with Node. Can't add {type(other_node)} to Node."
            )
