### `tf.batch_matrix_transpose(a, name='batch_matrix_transpose')` {#batch_matrix_transpose}

Transposes last two dimensions of batch matrix `a`.

For example:

```python
# Matrix with no batch dimension.
# 'x' is [[1 2 3]
#         [4 5 6]]
tf.batch_matrixtranspose(x) ==> [[1 4]
                                 [2 5]
                                 [3 6]]

# Matrix with two batch dimensions.
# x.shape is [1, 2, 3, 4]
# tf.batch_matrix_transpose(x) is shape [1, 2, 4, 3]
```

##### Args:


*  <b>`a`</b>: A `Tensor` with `rank >= 2`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A transposed batch matrix `Tensor`.

##### Raises:


*  <b>`ValueError`</b>: If `a` is determined statically to have `rank < 2`.

