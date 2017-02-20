Initializer that generates an orthogonal matrix.

If the shape of the tensor to initialize is two-dimensional, i is initialized
with an orthogonal matrix obtained from the singular value decomposition of a
matrix of uniform random numbers.

If the shape of the tensor to initialize is more than two-dimensional,
a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
is initialized, where `n` is the length of the shape vector.
The matrix is subsequently reshaped to give a tensor of the desired shape.

Args:
  gain: multiplicative factor to apply to the orthogonal matrix
  dtype: The type of the output.
  seed: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
- - -

#### `tf.orthogonal_initializer.__call__(shape, dtype=None, partition_info=None)` {#orthogonal_initializer.__call__}




- - -

#### `tf.orthogonal_initializer.__init__(gain=1.0, dtype=tf.float32, seed=None)` {#orthogonal_initializer.__init__}




