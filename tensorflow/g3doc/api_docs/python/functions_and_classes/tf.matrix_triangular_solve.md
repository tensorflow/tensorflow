### `tf.matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None)` {#matrix_triangular_solve}

Solves a system of linear equations with an upper or lower triangular matrix by

backsubstitution.

`matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
upper triangular part of `matrix` is assumed to be zero and not accessed.
If `lower` is False then the strictly lower triangular part of `matrix` is
assumed to be zero and not accessed.
`rhs` is a matrix of shape [M, K]`.

The output is a matrix of shape `[M, K]`. If `adjoint` is `False` the output
satisfies the matrix equation `matrix` * `output` = `rhs`.
If `adjoint` is `False` then `output` satisfies the matrix equation
`matrix` * `output` = `rhs`.
If `adjoint` is `True` then `output` satisfies the matrix equation
`adjoint(matrix)` * `output` = `rhs`.

##### Args:


*  <b>`matrix`</b>: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    Shape is `[M, M]`.
*  <b>`rhs`</b>: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
*  <b>`lower`</b>: An optional `bool`. Defaults to `True`.
    Boolean indicating whether `matrix` is lower or upper triangular
*  <b>`adjoint`</b>: An optional `bool`. Defaults to `False`.
    Boolean indicating whether to solve with `matrix` or its adjoint.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `matrix`. Shape is `[M, K]`.

