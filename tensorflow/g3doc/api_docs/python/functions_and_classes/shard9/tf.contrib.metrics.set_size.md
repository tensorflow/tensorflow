### `tf.contrib.metrics.set_size(a, validate_indices=True)` {#set_size}

Compute number of unique elements along last dimension of `a`.

##### Args:


*  <b>`a`</b>: `SparseTensor`, with indices sorted in row-major order.
*  <b>`validate_indices`</b>: Whether to validate the order and range of sparse indices
     in `a`.

##### Returns:

  For `a` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
  `n-1` dimensions as `a`. Each value is the number of unique elements in
  the corresponding `[0...n-1]` dimension of `a`.

##### Raises:


*  <b>`TypeError`</b>: If `a` is an invalid types.

