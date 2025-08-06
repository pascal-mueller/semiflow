from semiflow.nn.module import Module
from semiflow.nn.dropout import Dropout
from semiflow.nn.linear import Linear
from semiflow.nn.activation.relu import relu


class FeedForwardBlock(Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        self.linear_1 = Linear(d_model, d_ff)  # w1 and b1
        self.dropout = Dropout(dropout)
        self.linear_2 = Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(relu(self.linear_1(x))))
