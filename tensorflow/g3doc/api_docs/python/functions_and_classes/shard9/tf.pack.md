### `tf.pack(values, name='pack')` {#pack}

Packs a list of rank-`R` tensors into one rank-`(R+1)` tensor.

Packs tensors in `values` into a tensor with rank one higher than each tensor
in `values` and shape `[len(values)] + values[0].shape`. The output satisfies
`output[i, ...] = values[i][...]`.

This is the opposite of unpack.  The numpy equivalent is

    tf.pack([x, y, z]) = np.asarray([x, y, z])

##### Args:


*  <b>`values`</b>: A list of `Tensor` objects with the same shape and type.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`output`</b>: A packed `Tensor` with the same type as `values`.

