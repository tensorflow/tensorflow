### `tf.reduce_nnz(input_tensor, reduction_indices=None, keep_dims=False, dtype=tf.int32, name=None)` {#reduce_nnz}

Computes number of nonzero elements across dimensions of a tensor.

Reduces `input_tensor` along the dimensions given in `reduction_indices`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `reduction_indices` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

**NOTE** Floating point comparison to zero is done by exact floating point
equality check.  Small values are **not** rounded to zero for purposes of
the nonzero check.

For example:

```python
# 'x' is [[0, 1, 0]
#         [1, 1, 0]]
tf.reduce_nnz(x) ==> 3
tf.reduce_nnz(x, 0) ==> [1, 2, 0]
tf.reduce_nnz(x, 1) ==> [1, 2]
tf.reduce_nnz(x, 1, keep_dims=True) ==> [[1], [2]]
tf.reduce_nnz(x, [0, 1]) ==> 3
```

##### Args:


*  <b>`input_tensor`</b>: The tensor to reduce. Should be of numeric type, or `bool`.
*  <b>`reduction_indices`</b>: The dimensions to reduce. If `None` (the default),
    reduces all dimensions.
*  <b>`keep_dims`</b>: If true, retains reduced dimensions with length 1.
*  <b>`dtype`</b>: The output dtype; defaults to `tf.int32`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The reduced tensor.

