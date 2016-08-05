### `tf.self_adjoint_eig(matrix, name=None)` {#self_adjoint_eig}

Computes the eigen decomposition of a self-adjoint matrix.

Computes the eigenvalues and eigenvectors of an N-by-N matrix `matrix` such
that `matrix * v[:,i] = e(i) * v[:,i]`, for i=0...N-1.

##### Args:


*  <b>`matrix`</b>: `Tensor` of shape `[N, N]`.
*  <b>`name`</b>: string, optional name of the operation.

##### Returns:


*  <b>`e`</b>: Eigenvalues. Shape is `[N]`.
*  <b>`v`</b>: Eigenvectors. Shape is `[N, N]`. The columns contain the eigenvectors of
    `matrix`.

