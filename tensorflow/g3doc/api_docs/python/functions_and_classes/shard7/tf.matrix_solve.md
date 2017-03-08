### `tf.matrix_solve(matrix, rhs, adjoint=None, name=None)` {#matrix_solve}

Solves systems of linear equations.

`Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
If `adjoint` is `True` then each output matrix satisfies
`adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

##### Args:


*  <b>`matrix`</b>: A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
    Shape is `[..., M, M]`.
*  <b>`rhs`</b>: A `Tensor`. Must have the same type as `matrix`.
    Shape is `[..., M, K]`.
*  <b>`adjoint`</b>: An optional `bool`. Defaults to `False`.
    Boolean indicating whether to solve with `matrix` or its (block-wise)
    adjoint.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.

