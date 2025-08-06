import torch
import torch.nn as nn
from torchviz import make_dot

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=2):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Create model instance
model = SimpleModel()

# Create input tensor with gradient tracking
x = torch.randn(1, 3, requires_grad=True)  # batch_size=1, input_dim=3

# Forward pass
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output:", output)

# Create computational graph visualization
# The graph will show all operations and their connections
dot = make_dot(output, params=dict(list(model.named_parameters()) + [('input', x)]))

# Save the graph as PDF (you can also use 'png', 'svg', etc.)
dot.format = 'pdf'
dot.render('computational_graph', cleanup=True)
print("\nComputational graph saved as 'computational_graph.pdf'")

# Optional: Display graph structure in text
print("\n--- Graph Structure ---")
print(dot.source[:500] + "..." if len(dot.source) > 500 else dot.source)

# Let's also do a simple example with transpose to see the TBackward node
print("\n\n--- Example with Transpose ---")
y = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
z = y.T
w = z.sum()

print(f"Original tensor grad_fn: {y.grad_fn}")  # None (leaf)
print(f"Transposed tensor grad_fn: {z.grad_fn}")  # <TBackward0>
print(f"Sum grad_fn: {w.grad_fn}")  # <SumBackward0>

# Visualize the transpose graph
dot_transpose = make_dot(w, params={'y': y})
dot_transpose.format = 'pdf'
dot_transpose.render('transpose_graph', cleanup=True)
print("Transpose graph saved as 'transpose_graph.pdf'")
