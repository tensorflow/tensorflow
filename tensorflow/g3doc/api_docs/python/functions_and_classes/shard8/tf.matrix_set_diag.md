### `tf.matrix_set_diag(input, diagonal, name=None)` {#matrix_set_diag}

Returns a batched matrix tensor with new batched diagonal values.

Given `input` and `diagonal`, this operation returns a tensor with the
same shape and values as `input`, except for the main diagonal of the
innermost matrices.  These will be overwritten by the values in `diagonal`.

The output is computed as follows:

Assume `input` has `k+1` dimensions `[I, J, K, ..., M, N]` and `diagonal` has
`k` dimensions `[I, J, K, ..., min(M, N)]`.  Then the output is a
tensor of rank `k+1` with dimensions `[I, J, K, ..., M, N]` where:

  * `output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]` for `m == n`.
  * `output[i, j, k, ..., m, n] = input[i, j, k, ..., m, n]` for `m != n`.

##### Args:


*  <b>`input`</b>: A `Tensor`. Rank `k+1`, where `k >= 1`.
*  <b>`diagonal`</b>: A `Tensor`. Must have the same type as `input`.
    Rank `k`, where `k >= 1`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor`. Has the same type as `input`.
  Rank `k+1`, with `output.shape = input.shape`.

