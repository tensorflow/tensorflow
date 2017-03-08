### `tf.unpack(value, num=None, axis=0, name='unpack')` {#unpack}

Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` along the given dimension.
If `num` is not specified (the default), it is inferred from `value`'s shape.
If `value.shape[axis]` is not known, `ValueError` is raised.

The ith tensor in `output` is the slice `value[i, ...]`. Each tensor in
`output` has shape `value.shape[1:]`.

This is the opposite of pack.  The numpy equivalent is

    tf.unpack(x, n) = list(x)

##### Args:


*  <b>`value`</b>: A rank `R > 0` `Tensor` to be unpacked.
*  <b>`num`</b>: An `int`. The length of the dimension `axis`. Automatically inferred
    if `None` (the default).
*  <b>`axis`</b>: An `int`. The axis to unpack along. Defaults to the first
    dimension. Supports negative indexes.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The list of `Tensor` objects unpacked from `value`.

##### Raises:


*  <b>`ValueError`</b>: If `num` is unspecified and cannot be inferred.
*  <b>`ValueError`</b>: If `axis` is out of the range [-R, R).

