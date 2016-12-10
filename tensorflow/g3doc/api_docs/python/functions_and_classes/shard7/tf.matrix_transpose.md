### `tf.matrix_transpose(a, name='matrix_transpose')` {#matrix_transpose}

Transposes last two dimensions of tensor `a`.

For example:

```python
# Matrix with no batch dimension.
# 'x' is [[1 2 3]
#         [4 5 6]]
tf.matrix_transpose(x) ==> [[1 4]
                                 [2 5]
                                 [3 6]]

# Matrix with two batch dimensions.
# x.shape is [1, 2, 3, 4]
# tf.matrix_transpose(x) is shape [1, 2, 4, 3]
```

##### Args:


*  <b>`a`</b>: A `Tensor` with `rank >= 2`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A transposed batch matrix `Tensor`.

##### Raises:


*  <b>`ValueError`</b>: If `a` is determined statically to have `rank < 2`.

