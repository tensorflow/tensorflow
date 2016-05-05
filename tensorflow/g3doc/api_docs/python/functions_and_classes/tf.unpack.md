### `tf.unpack(value, num=None, name='unpack')` {#unpack}

Unpacks the outer dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

Unpacks `num` tensors from `value` along the first dimension.
If `num` is not specified (the default), it is inferred from `value`'s shape.
If `value.shape[0]` is not known, `ValueError` is raised.

The ith tensor in `output` is the slice `value[i, ...]`. Each tensor in
`output` has shape `value.shape[1:]`.

This is the opposite of pack.  The numpy equivalent is

    tf.unpack(x, n) = list(x)

##### Args:


*  <b>`value`</b>: A rank `R > 0` `Tensor` to be unpacked.
*  <b>`num`</b>: An `int`. The first dimension of value. Automatically inferred if
    `None` (the default).
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The list of `Tensor` objects unpacked from `value`.

##### Raises:


*  <b>`ValueError`</b>: If `num` is unspecified and cannot be inferred.

