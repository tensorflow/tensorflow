### `tf.contrib.metrics.set_union(a, b, validate_indices=True)` {#set_union}

Compute set union of elements in last dimension of `a` and `b`.

All but the last dimension of `a` and `b` must match.

##### Args:


*  <b>`a`</b>: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
*  <b>`b`</b>: `Tensor` or `SparseTensor` of the same type as `a`. Must be
      `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
      sorted in row-major order.
*  <b>`validate_indices`</b>: Whether to validate the order and range of sparse indices
     in `a` and `b`.

##### Returns:

  A `SparseTensor` with the same rank as `a` and `b`, and all but the last
  dimension the same. Elements along the last dimension contain the
  unions.

