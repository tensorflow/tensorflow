### `tf.matrix_inverse(input, adjoint=None, name=None)` {#matrix_inverse}

Computes the inverse of a square invertible matrix or its adjoint (conjugate

transpose).

The op uses LU decomposition with partial pivoting to compute the inverse.

If the matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    Shape is `[M, M]`.
*  <b>`adjoint`</b>: An optional `bool`. Defaults to `False`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  Shape is `[M, M]`. If `adjoint` is `False` then `output` contains the
  matrix inverse of `input`. If `adjoint` is `True` then `output` contains the
  matrix inverse of the adjoint of `input`.

