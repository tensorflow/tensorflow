### `tf.cholesky_solve(chol, rhs, name=None)` {#cholesky_solve}

Solves systems of linear eqns `A X = RHS`, given Cholesky factorizations.

```python
# Solve 10 separate 2x2 linear systems:
A = ... # shape 10 x 2 x 2
RHS = ... # shape 10 x 2 x 1
chol = tf.cholesky(A)  # shape 10 x 2 x 2
X = tf.cholesky_solve(chol, RHS)  # shape 10 x 2 x 1
# tf.matmul(A, X) ~ RHS
X[3, :, 0]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 0]

# Solve five linear systems (K = 5) for every member of the length 10 batch.
A = ... # shape 10 x 2 x 2
RHS = ... # shape 10 x 2 x 5
...
X[3, :, 2]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 2]
```

##### Args:


*  <b>`chol`</b>: A `Tensor`.  Must be `float32` or `float64`, shape is `[..., M, M]`.
    Cholesky factorization of `A`, e.g. `chol = tf.cholesky(A)`.
    For that reason, only the lower triangular parts (including the diagonal)
    of the last two dimensions of `chol` are used.  The strictly upper part is
    assumed to be zero and not accessed.
*  <b>`rhs`</b>: A `Tensor`, same type as `chol`, shape is `[..., M, K]`.
*  <b>`name`</b>: A name to give this `Op`.  Defaults to `cholesky_solve`.

##### Returns:

  Solution to `A x = rhs`, shape `[..., M, K]`.

