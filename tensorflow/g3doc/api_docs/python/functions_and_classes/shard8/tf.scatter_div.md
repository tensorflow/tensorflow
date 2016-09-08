### `tf.scatter_div(ref, indices, updates, use_locking=None, name=None)` {#scatter_div}

Divides a variable reference by sparse updates.

This operation computes

    # Scalar indices
    ref[indices, ...] /= updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] /= updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] /= updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

Duplicate entries are handled correctly: if multiple `indices` reference
the same location, their contributions divide.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    Should be from a `Variable` node.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A tensor of indices into the first dimension of `ref`.
*  <b>`updates`</b>: A `Tensor`. Must have the same type as `ref`.
    A tensor of values that `ref` is divided by.
*  <b>`use_locking`</b>: An optional `bool`. Defaults to `False`.
    If True, the operation will be protected by a lock;
    otherwise the behavior is undefined, but may exhibit less contention.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.

