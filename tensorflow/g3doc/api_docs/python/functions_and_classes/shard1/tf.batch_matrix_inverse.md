### `tf.batch_matrix_inverse(input, adjoint=None, name=None)` {#batch_matrix_inverse}

Calculates the inverse of square invertible matrices or their adjoints

(conjugate transposes).

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor of the same shape as the input
containing the inverse for all input submatrices `[..., :, :]`.

The op uses LU decomposition with partial pivoting to compute the inverses.

If a matrix is not invertible there is no guarantee what the op does. It
may detect the condition and raise an exception or it may simply return a
garbage result.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    Shape is `[..., M, M]`.
*  <b>`adjoint`</b>: An optional `bool`. Defaults to `False`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.

