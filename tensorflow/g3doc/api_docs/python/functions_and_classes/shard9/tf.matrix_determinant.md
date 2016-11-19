### `tf.matrix_determinant(input, name=None)` {#matrix_determinant}

Computes the determinant of one ore more square matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices. The output is a tensor containing the determinants
for all input submatrices `[..., :, :]`.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    Shape is `[..., M, M]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`. Shape is `[...]`.

