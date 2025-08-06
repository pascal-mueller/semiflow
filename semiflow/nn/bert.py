import jax.numpy as jnp

from semiflow.node import Node
from semiflow.nn.embedding import Embedding
from semiflow.nn.transformer.positional_encoding import PositionalEncoding
from semiflow.nn.transformer.multi_head_attention_block import MultiHeadAttentionBlock
from semiflow.nn.transformer.feed_forward_block import FeedForwardBlock
from semiflow.nn.transformer.encoder_block import EncoderBlock

D_MODEL_DIM = 512  # Internal processing dimension
VOCAB_SIZE = 10  # We have 10 "words" in our vocabulary

input = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1])

embeddings = Embedding(d_model=D_MODEL_DIM, vocab_size=VOCAB_SIZE)
pos_encodings = PositionalEncoding(d_model=D_MODEL_DIM, seq_len=len(input), dropout=0.1)

input_embeddings = embeddings(input)
input_embeddings_encoded = pos_encodings(input_embeddings)

multi_head_attention_block = MultiHeadAttentionBlock(
    d_model=D_MODEL_DIM, h=8, dropout=0.1
)
feed_forward_block = FeedForwardBlock(d_model=D_MODEL_DIM, d_ff=20, dropout=0.1)

encoder_block = EncoderBlock(
    features=D_MODEL_DIM,
    self_attention_block=multi_head_attention_block,
    feed_forward_block=feed_forward_block,
    dropout=0.1,
)

output = encoder_block.forward(input_embeddings_encoded, src_mask=None)
