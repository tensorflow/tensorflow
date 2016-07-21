### `tf.matmul(a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None)` {#matmul}

Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

The inputs must be two-dimensional matrices, with matching inner dimensions,
possibly after transposition.

Both matrices must be of the same type. The supported types are:
`float32`, `float64`, `int32`, `complex64`.

Either matrix can be transposed on the fly by setting the corresponding flag
to `True`. This is `False` by default.

If one or both of the matrices contain a lot of zeros, a more efficient
multiplication algorithm can be used by setting the corresponding
`a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.

For example:

```python
# 2-D tensor `a`
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                      [4. 5. 6.]]
# 2-D tensor `b`
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                         [9. 10.]
                                                         [11. 12.]]
c = tf.matmul(a, b) => [[58 64]
                        [139 154]]
```

##### Args:


*  <b>`a`</b>: `Tensor` of type `float32`, `float64`, `int32` or `complex64`.
*  <b>`b`</b>: `Tensor` with same type as `a`.
*  <b>`transpose_a`</b>: If `True`, `a` is transposed before multiplication.
*  <b>`transpose_b`</b>: If `True`, `b` is transposed before multiplication.
*  <b>`a_is_sparse`</b>: If `True`, `a` is treated as a sparse matrix.
*  <b>`b_is_sparse`</b>: If `True`, `b` is treated as a sparse matrix.
*  <b>`name`</b>: Name for the operation (optional).

##### Returns:

  A `Tensor` of the same type as `a`.

