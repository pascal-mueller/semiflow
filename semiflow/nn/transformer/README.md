BERT encoder consists of:

Input
 => Embedding
 => Positional Encoding
    - skip to add & norm
 => Mutli-Head Attention
 => Add & Norm
    - skip to add & norm
 => Feed Forward
 => Add & Norm
 => Output

Embedding: `embedding.py`
Pos. Enc.: `positionalEndocing.py`
1st skip: `residual_connection.py`
Multi-Head Att.: `multiheadAttentionBlock.py`
Add: `residual_connection.py`
Norm: `layer_normalization.py`
2nd skip: `residual_connection.py` 
Feed Forward: `FeedForwardBlock.py`
Add: `residual_connection.py`
Norm: `layer_normalization.py`
