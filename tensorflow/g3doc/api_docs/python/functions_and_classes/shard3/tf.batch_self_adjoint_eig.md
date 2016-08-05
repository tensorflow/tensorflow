### `tf.batch_self_adjoint_eig(tensor, name=None)` {#batch_self_adjoint_eig}

Computes the eigen decomposition of a batch of self-adjoint matrices.

Computes the eigenvalues and eigenvectors of the innermost N-by-N matrices
in `tensor` such that
`tensor[...,:,:] * v[..., :,i] = e(..., i) * v[...,:,i]`, for i=0...N-1.

##### Args:


*  <b>`tensor`</b>: `Tensor` of shape `[..., N, N]`.
*  <b>`name`</b>: string, optional name of the operation.

##### Returns:


*  <b>`e`</b>: Eigenvalues. Shape is `[..., N]`.
*  <b>`v`</b>: Eigenvectors. Shape is `[..., N, N]`. The columns of the inner most
  matrices
    contain eigenvectors of the corresponding matrices in `tensor`

