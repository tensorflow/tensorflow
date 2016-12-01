### `tf.rank(input, name=None)` {#rank}

Returns the rank of a tensor.

This operation returns an integer representing the rank of `input`.

For example:

```python
# 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
# shape of tensor 't' is [2, 2, 3]
rank(t) ==> 3
```

**Note**: The rank of a tensor is not the same as the rank of a matrix. The
rank of a tensor is the number of indices required to uniquely select each
element of the tensor. Rank is also known as "order", "degree", or "ndims."

##### Args:


*  <b>`input`</b>: A `Tensor` or `SparseTensor`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int32`.

@compatibility(numpy)
Equivalent to np.ndim
@end_compatibility

