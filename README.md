JAX based deep learning framework implementing semiring backpropagation

It uses JAX for elemental operations but implements its own computational
graph and (semiring) backpropagation.

# Setup
Clone this repo and do:

- `pip install -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `pip install -e .`

This creates an editable install of the package. You can now use it inside the
`.venv` virtual enfironment by using `from semiflow import <module>`.

# JAX Array Shape Convention

In JAX (and NumPy), the shape of a 2D array (matrix) is **(rows, columns)**.

- For a 2D array:  
  `arr.shape = (rows, columns)`

- For higher-dimensional arrays (e.g., batches of matrices), the first dimension(s) are typically batch or other axes, and the last two are rows and columns:  
  `arr.shape = (batch_size, ..., rows, columns)`

- Therefore:
  - `arr.shape[-2]` is the number of rows
  - `arr.shape[-1]` is the number of columns

A 1D array (e.g., `jnp.array([1, 2, 3])`) is just a vector and is not explicitly a row or column vector. For matrix multiplication, it is treated as a row vector by default unless

## Why Do We Use the Transpose in Linear Layers?

In deep learning, a linear (fully connected) layer applies an affine transformation to its input:

\[
y = xA^T + b
\]

Let's break down **why the transpose appears** by starting with the basics.

### 1. Standard Matrix-Vector Multiplication

Suppose you have a weight matrix \( A \) and an input vector \( x \):

\[
A =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1d} \\
a_{21} & a_{22} & \cdots & a_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
a_{k1} & a_{k2} & \cdots & a_{kd}
\end{bmatrix}
\qquad
x =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_d
\end{bmatrix}
\]

Here, \( A \) has shape \((k, d)\) and \( x \) has shape \((d, 1)\). The product \( Ax \) yields a vector of shape \((k, 1)\):

\[
Ax =
\begin{bmatrix}
\sum_{j=1}^d a_{1j} x_j \\
\sum_{j=1}^d a_{2j} x_j \\
\vdots \\
\sum_{j=1}^d a_{kj} x_j
\end{bmatrix}
\]

### 2. Batch Inputs in Deep Learning

In deep learning, we process **batches** of inputs for efficiency. Instead of a single vector \( x \), we have a matrix \( X \) where each row is a sample:

\[
X =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1d} \\
x_{21} & x_{22} & \cdots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{nd}
\end{bmatrix}
\]

Here, \( X \) has shape \((n, d)\) where \( n \) is the batch size and \( d \) is the number of input features.

### 3. Why the Transpose?

If we try to multiply \( X \) and \( A \) directly:

- \( X \) is \((n, d)\)
- \( A \) is \((k, d)\)

But matrix multiplication \( X A \) is **not defined** because the inner dimensions do not match (\( d \neq k \)).  
To fix this, we **transpose** \( A \):

- \( A^T \) is \((d, k)\)
- Now \( X A^T \) is \((n, d) \times (d, k) = (n, k) \)

This gives us an output where each of the \( n \) samples is mapped to a \( k \)-dimensional output vector.

### 4. Summary

- **Single sample:** \( y = Ax + b \)
- **Batch of samples:** \( Y = X A^T + b \)
- The transpose ensures the matrix multiplication is valid and produces the correct output shape.

### 5. Why Store Weights as \((\text{out\_features}, \text{in\_features})\)?

- Each row of \( A \) corresponds to the weights for one output feature.
- This layout is efficient for memory access (row-major order) and matches mathematical conventions.
- The transpose in the computation aligns the dimensions for batch processing.

---

**In code:**

```python
# X: (batch_size, in_features)
# weights: (out_features, in_features)
output = X @ weights.T +