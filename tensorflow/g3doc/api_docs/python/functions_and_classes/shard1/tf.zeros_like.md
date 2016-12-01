### `tf.zeros_like(tensor, dtype=None, name=None, optimize=True)` {#zeros_like}

Creates a tensor with all elements set to zero.

Given a single tensor (`tensor`), this operation returns a tensor of the
same type and shape as `tensor` with all elements set to zero. Optionally,
you can use `dtype` to specify a new type for the returned tensor.

For example:

```python
# 'tensor' is [[1, 2, 3], [4, 5, 6]]
tf.zeros_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`.
*  <b>`dtype`</b>: A type for the returned `Tensor`. Must be `float32`, `float64`,
  `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`.

*  <b>`name`</b>: A name for the operation (optional).
*  <b>`optimize`</b>: if true, attempt to statically determine the shape of 'tensor'
  and encode it as a constant.

##### Returns:

  A `Tensor` with all elements set to zero.

