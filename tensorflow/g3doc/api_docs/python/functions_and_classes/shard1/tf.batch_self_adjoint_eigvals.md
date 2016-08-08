### `tf.batch_self_adjoint_eigvals(tensor, name=None)` {#batch_self_adjoint_eigvals}

Computes the eigenvalues of a batch of self-adjoint matrices.

##### Args:


*  <b>`tensor`</b>: `Tensor` of shape `[..., N, N]`.
*  <b>`name`</b>: string, optional name of the operation.

##### Returns:


*  <b>`e`</b>: Eigenvalues. Shape is `[..., N]`. The vector `e[..., :]` contains the `N`
    eigenvalues of `tensor[..., :, :]`.

