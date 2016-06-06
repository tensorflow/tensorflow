### `tf.shape(input, name=None)` {#shape}

Returns the shape of a tensor.

This operation returns a 1-D integer tensor representing the shape of `input`.

For example:

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
shape(t) ==> [2, 2, 3]
```

##### Args:


*  <b>`input`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int32`.

