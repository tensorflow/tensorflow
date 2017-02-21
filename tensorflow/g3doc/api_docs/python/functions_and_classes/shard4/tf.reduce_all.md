### `tf.reduce_all(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)` {#reduce_all}

Computes the "logical and" of elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

For example:

```python
# 'x' is [[True,  True]
#         [False, False]]
tf.reduce_all(x) ==> False
tf.reduce_all(x, 0) ==> [False, False]
tf.reduce_all(x, 1) ==> [True, False]
```

##### Args:


*  <b>`input_tensor`</b>: The boolean tensor to reduce.
*  <b>`axis`</b>: The dimensions to reduce. If `None` (the default),
    reduces all dimensions.
*  <b>`keep_dims`</b>: If true, retains reduced dimensions with length 1.
*  <b>`name`</b>: A name for the operation (optional).
*  <b>`reduction_indices`</b>: The old (deprecated) name for axis.

##### Returns:

  The reduced tensor.

@compatibility(numpy)
Equivalent to np.all
@end_compatibility

