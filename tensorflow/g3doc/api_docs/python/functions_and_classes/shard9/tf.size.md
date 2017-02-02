### `tf.size(input, name=None, out_type=tf.int32)` {#size}

Returns the size of a tensor.

This operation returns an integer representing the number of elements in
`input`.

For example:

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
size(t) ==> 12
```

##### Args:


*  <b>`input`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`out_type`</b>: (Optional) The specified output type of the operation
    (`int32` or `int64`). Defaults to tf.int32.

##### Returns:

  A `Tensor` of type `out_type`. Defaults to tf.int32.

