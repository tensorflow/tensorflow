### `tf.reduce_logsumexp(input_tensor, reduction_indices=None, keep_dims=False, name=None)` {#reduce_logsumexp}

Computes log(sum(exp(elements across dimensions of a tensor))).

Reduces `input_tensor` along the dimensions given in `reduction_indices`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `reduction_indices` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

This function is more numerically stable than log(sum(exp(input))). It avoids
overflows caused by taking the exp of large inputs and underflows caused by
taking the log of small inputs.

For example:

```python
# 'x' is [[0, 0, 0]]
#         [0, 0, 0]]
tf.reduce_logsumexp(x) ==> log(6)
tf.reduce_logsumexp(x, 0) ==> [log(2), log(2), log(2)]
tf.reduce_logsumexp(x, 1) ==> [log(3), log(3)]
tf.reduce_logsumexp(x, 1, keep_dims=True) ==> [[log(3)], [log(3)]]
tf.reduce_logsumexp(x, [0, 1]) ==> log(6)
```

##### Args:


*  <b>`input_tensor`</b>: The tensor to reduce. Should have numeric type.
*  <b>`reduction_indices`</b>: The dimensions to reduce. If `None` (the default),
    reduces all dimensions.
*  <b>`keep_dims`</b>: If true, retains reduced dimensions with length 1.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  The reduced tensor.

