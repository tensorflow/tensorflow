### `tf.cholesky_solve(chol, rhs, name=None)` {#cholesky_solve}

Solve linear equations `A X = RHS`, given Cholesky factorization of `A`.

```python
# Solve one system of linear equations (K = 1).
A = [[3, 1], [1, 3]]
RHS = [[2], [22]]  # shape 2 x 1
chol = tf.cholesky(A)
X = tf.cholesky_solve(chol, RHS)
# tf.matmul(A, X) ~ RHS
X[:, 0]  # Solution to the linear system A x = RHS[:, 0]

# Solve five systems of linear equations (K = 5).
A = [[3, 1], [1, 3]]
RHS = [[1, 2, 3, 4, 5], [11, 22, 33, 44, 55]]  # shape 2 x 5
...
X[:, 2]  # Solution to the linear system A x = RHS[:, 2]
```

##### Args:


*  <b>`chol`</b>: A `Tensor`.  Must be `float32` or `float64`, shape is `[M, M]`.
    Cholesky factorization of `A`, e.g. `chol = tf.cholesky(A)`.  For that
    reason, only the lower triangular part (including the diagonal) of `chol`
    is used.  The strictly upper part is assumed to be zero and not accessed.
*  <b>`rhs`</b>: A `Tensor`, same type as `chol`, shape is `[M, K]`, designating `K`
    systems of linear equations.
*  <b>`name`</b>: A name to give this `Op`.  Defaults to `cholesky_solve`.

##### Returns:

  Solution to `A X = RHS`, shape `[M, K]`.  The solutions to the `K` systems.

