### `tf.to_bfloat16(x, name='ToBFloat16')` {#to_bfloat16}

Casts a tensor to type `bfloat16`.

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x` with type `bfloat16`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `bfloat16`.

