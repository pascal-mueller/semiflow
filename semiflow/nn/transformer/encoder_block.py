from semiflow.nn.module import Module
from semiflow.nn.transformer.multi_head_attention_block import MultiHeadAttentionBlock
from semiflow.nn.transformer.feed_forward_block import FeedForwardBlock
from semiflow.nn.transformer.residual_connection import ResidualConnection


class EncoderBlock(Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.skip_attention = ResidualConnection(features, dropout)
        self.skip_feed_forward = ResidualConnection(features, dropout)

    def forward(self, x, src_mask):
        x = self.skip_attention(
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.skip_feed_forward(x, self.feed_forward_block)
        return x
