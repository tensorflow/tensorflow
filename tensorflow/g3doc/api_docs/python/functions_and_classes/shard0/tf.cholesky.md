### `tf.cholesky(input, name=None)` {#cholesky}

Computes the Cholesky decomposition of one or more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix Cholesky
decomposition above. The output is a tensor of the same shape as the input
containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    Shape is `[..., M, M]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.

