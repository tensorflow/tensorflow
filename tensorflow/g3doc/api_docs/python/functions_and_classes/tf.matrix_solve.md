### `tf.matrix_solve(matrix, rhs, adjoint=None, name=None)` {#matrix_solve}

Solves a system of linear equations. Checks for invertibility.

##### Args:


*  <b>`matrix`</b>: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    Shape is `[M, M]`.
*  <b>`rhs`</b>: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
*  <b>`adjoint`</b>: An optional `bool`. Defaults to `False`.
    Boolean indicating whether to solve with `matrix` or its adjoint.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `matrix`.
  Shape is `[M, K]`. If `adjoint` is `False` then `output` that solves
  `matrix` * `output` = `rhs`. If `adjoint` is `True` then `output` that solves
  `adjoint(matrix)` * `output` = `rhs`.

