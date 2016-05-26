### `tf.batch_self_adjoint_eig(input, name=None)` {#batch_self_adjoint_eig}

Calculates the Eigen Decomposition of a batch of square self-adjoint matrices.

The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
form square matrices, with the same constraints as the single matrix
SelfAdjointEig.

The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    Shape is `[..., M, M]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`. Shape is `[..., M+1, M]`.

