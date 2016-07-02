### `tf.ones_like(tensor, dtype=None, name=None)` {#ones_like}

Creates a tensor with all elements set to 1.

Given a single tensor (`tensor`), this operation returns a tensor of the same
type and shape as `tensor` with all elements set to 1. Optionally, you can
specify a new type (`dtype`) for the returned tensor.

For example:

```python
# 'tensor' is [[1, 2, 3], [4, 5, 6]]
tf.ones_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]
```

##### Args:


*  <b>`tensor`</b>: A `Tensor`.
*  <b>`dtype`</b>: A type for the returned `Tensor`. Must be `float32`, `float64`,
  `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`.

*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with all elements set to 1.

