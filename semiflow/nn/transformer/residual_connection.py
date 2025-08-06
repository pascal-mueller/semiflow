from semiflow.node import Node
from semiflow.nn.module import Module
from semiflow.nn.dropout import Dropout
from semiflow.nn.transformer.layer_normalization import LayerNormalization


class ResidualConnection(Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()

        self.dropout = Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: Node, sublayer):
        return self.dropout(sublayer(self.norm(x))) + x
