### `tf.unstack(value, num=None, axis=0, name='unstack')` {#unstack}

Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
If `num` is not specified (the default), it is inferred from `value`'s shape.
If `value.shape[axis]` is not known, `ValueError` is raised.

For example, given a tensor of shape `(A, B, C, D)`;

If `axis == 0` then the i'th tensor in `output` is the slice
  `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
  (Note that the dimension unpacked along is gone, unlike `split`).

If `axis == 1` then the i'th tensor in `output` is the slice
  `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
Etc.

This is the opposite of pack.  The numpy equivalent is

    tf.unstack(x, n) = list(x)

##### Args:


*  <b>`value`</b>: A rank `R > 0` `Tensor` to be unstacked.
*  <b>`num`</b>: An `int`. The length of the dimension `axis`. Automatically inferred
    if `None` (the default).
*  <b>`axis`</b>: An `int`. The axis to unstack along. Defaults to the first
    dimension. Supports negative indexes.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The list of `Tensor` objects unstacked from `value`.

##### Raises:


*  <b>`ValueError`</b>: If `num` is unspecified and cannot be inferred.
*  <b>`ValueError`</b>: If `axis` is out of the range [-R, R).

