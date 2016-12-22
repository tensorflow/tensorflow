### `tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)` {#matmul}

Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

The inputs must be matrices (or tensors of rank > 2, representing batches of
matrices), with matching inner dimensions, possibly after transposition.

Both matrices must be of the same type. The supported types are:
`float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

Either matrix can be transposed or adjointed (conjugated and transposed) on
the fly by setting one of the corresponding flag to `True`. These are `False`
by default.

If one or both of the matrices contain a lot of zeros, a more efficient
multiplication algorithm can be used by setting the corresponding
`a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
This optimization is only available for plain matrices (rank-2 tensors) with
datatypes `bfloat16` or `float32`.

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


# 3-D tensor `a`
a = tf.constant(np.arange(1, 13, dtype=np.int32),
                shape=[2, 2, 3])                  => [[[ 1.  2.  3.]
                                                       [ 4.  5.  6.]],
                                                      [[ 7.  8.  9.]
                                                       [10. 11. 12.]]]

# 3-D tensor `b`
b = tf.constant(np.arange(13, 25, dtype=np.int32),
                shape=[2, 3, 2])                   => [[[13. 14.]
                                                        [15. 16.]
                                                        [17. 18.]],
                                                       [[19. 20.]
                                                        [21. 22.]
                                                        [23. 24.]]]
c = tf.matmul(a, b) => [[[ 94 100]
                         [229 244]],
                        [[508 532]
                         [697 730]]]
```

##### Args:


*  <b>`a`</b>: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
    `complex128` and rank > 1.
*  <b>`b`</b>: `Tensor` with same type and rank as `a`.
*  <b>`transpose_a`</b>: If `True`, `a` is transposed before multiplication.
*  <b>`transpose_b`</b>: If `True`, `b` is transposed before multiplication.
*  <b>`adjoint_a`</b>: If `True`, `a` is conjugated and transposed before
    multiplication.
*  <b>`adjoint_b`</b>: If `True`, `b` is conjugated and transposed before
    multiplication.
*  <b>`a_is_sparse`</b>: If `True`, `a` is treated as a sparse matrix.
*  <b>`b_is_sparse`</b>: If `True`, `b` is treated as a sparse matrix.
*  <b>`name`</b>: Name for the operation (optional).

##### Returns:

  A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
  the product of the corresponding matrices in `a` and `b, e.g. if all
  transpose or adjoint attributes are `False`:

  `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
  for all indices i, j.


*  <b>`Note`</b>: This is matrix product, not element-wise product.


##### Raises:


*  <b>`ValueError`</b>: If transpose_a and adjoint_a, or transpose_b and adjoint_b
    are both set to True.

