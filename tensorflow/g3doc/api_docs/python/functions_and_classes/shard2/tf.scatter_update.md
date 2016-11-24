### `tf.scatter_update(ref, indices, updates, use_locking=None, name=None)` {#scatter_update}

Applies sparse updates to a variable reference.

This operation computes

    # Scalar indices
    ref[indices, ...] = updates[...]

    # Vector indices (for each i)
    ref[indices[i], ...] = updates[i, ...]

    # High rank indices (for each i, ..., j)
    ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]

This operation outputs `ref` after the update is done.
This makes it easier to chain operations that need to use the reset value.

If values in `ref` is to be updated more than once, because there are
duplicate entries in `indices`, the order at which the updates happen
for each value is undefined.

Requires `updates.shape = indices.shape + ref.shape[1:]`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/ScatterUpdate.png" alt>
</div>

##### Args:


*  <b>`ref`</b>: A mutable `Tensor`. Should be from a `Variable` node.
*  <b>`indices`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    A tensor of indices into the first dimension of `ref`.
*  <b>`updates`</b>: A `Tensor`. Must have the same type as `ref`.
    A tensor of updated values to store in `ref`.
*  <b>`use_locking`</b>: An optional `bool`. Defaults to `True`.
    If True, the assignment will be protected by a lock;
    otherwise the behavior is undefined, but may exhibit less contention.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  Same as `ref`.  Returned as a convenience for operations that want
  to use the updated values after the update is done.

