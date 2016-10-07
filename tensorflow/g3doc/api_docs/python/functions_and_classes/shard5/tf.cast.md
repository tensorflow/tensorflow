### `tf.cast(x, dtype, name=None)` {#cast}

Casts a tensor to a new type.

The operation casts `x` (in case of `Tensor`) or `x.values`
(in case of `SparseTensor`) to `dtype`.

For example:

```python
# tensor `a` is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
```

##### Args:


*  <b>`x`</b>: A `Tensor` or `SparseTensor`.
*  <b>`dtype`</b>: The destination type.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` or `SparseTensor` with same shape as `x`.

##### Raises:


*  <b>`TypeError`</b>: If `x` cannot be cast to the `dtype`.

