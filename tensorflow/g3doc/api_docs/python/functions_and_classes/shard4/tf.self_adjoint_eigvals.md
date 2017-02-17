### `tf.self_adjoint_eigvals(tensor, name=None)` {#self_adjoint_eigvals}

Computes the eigenvalues of one or more self-adjoint matrices.

##### Args:


*  <b>`tensor`</b>: `Tensor` of shape `[..., N, N]`.
*  <b>`name`</b>: string, optional name of the operation.

##### Returns:


*  <b>`e`</b>: Eigenvalues. Shape is `[..., N]`. The vector `e[..., :]` contains the `N`
    eigenvalues of `tensor[..., :, :]`.

