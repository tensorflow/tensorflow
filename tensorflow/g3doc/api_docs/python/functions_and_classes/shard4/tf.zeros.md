### `tf.zeros(shape, dtype=tf.float32, name=None)` {#zeros}

Creates a tensor with all elements set to zero.

This operation returns a tensor of type `dtype` with shape `shape` and
all elements set to zero.

For example:

```python
tf.zeros([3, 4], int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

##### Args:


*  <b>`shape`</b>: Either a list of integers, or a 1-D `Tensor` of type `int32`.
*  <b>`dtype`</b>: The type of an element in the resulting `Tensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with all elements set to zero.

