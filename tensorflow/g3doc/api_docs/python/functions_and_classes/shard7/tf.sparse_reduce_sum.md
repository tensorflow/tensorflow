### `tf.sparse_reduce_sum(sp_input, reduction_axes=None, keep_dims=False)` {#sparse_reduce_sum}

Computes the sum of elements across dimensions of a SparseTensor.

This Op takes a SparseTensor and is the sparse counterpart to
`tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
instead of a sparse one.

Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
with length 1.

If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
with a single element is returned.  Additionally, the axes can be negative,
similar to the indexing rules in Python.

For example:

```python
# 'x' represents [[1, ?, 1]
#                 [?, 1, ?]]
# where ? is implictly-zero.
tf.sparse_reduce_sum(x) ==> 3
tf.sparse_reduce_sum(x, 0) ==> [1, 1, 1]
tf.sparse_reduce_sum(x, 1) ==> [2, 1]  # Can also use -1 as the axis.
tf.sparse_reduce_sum(x, 1, keep_dims=True) ==> [[2], [1]]
tf.sparse_reduce_sum(x, [0, 1]) ==> 3
```

##### Args:


*  <b>`sp_input`</b>: The SparseTensor to reduce. Should have numeric type.
*  <b>`reduction_axes`</b>: The dimensions to reduce; list or scalar. If `None` (the
    default), reduces all dimensions.
*  <b>`keep_dims`</b>: If true, retain reduced dimensions with length 1.

##### Returns:

  The reduced Tensor.

