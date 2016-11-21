### `tf.orthogonal_initializer(gain=1.0, dtype=tf.float32, seed=None)` {#orthogonal_initializer}

Returns an initializer that generates an orthogonal matrix or a reshaped 
orthogonal matrix.

If the shape of the tensor to initialize is two-dimensional, i is initialized
with an orthogonal matrix obtained from the singular value decomposition of a
matrix of uniform random numbers.

If the shape of the tensor to initialize is more than two-dimensional, a matrix
of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])` is initialized, where
`n` is the length of the shape vector. The matrix is subsequently reshaped to
give a tensor of the desired shape.

##### Args:


*  <b>`gain`</b>: multiplicative factor to apply to the orthogonal matrix
*  <b>`dtype`</b>: The type of the output.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.

##### Returns:

  An initializer that generates orthogonal tensors

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type or if `shape` has fewer than two entries.

