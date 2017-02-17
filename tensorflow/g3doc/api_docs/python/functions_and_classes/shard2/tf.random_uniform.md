### `tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)` {#random_uniform}

Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range
`[minval, maxval)`. The lower bound `minval` is included in the range, while
the upper bound `maxval` is excluded.

For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
be specified explicitly.

In the integer case, the random integers are slightly biased unless
`maxval - minval` is an exact power of two.  The bias is small for values of
`maxval - minval` significantly smaller than the range of the output (either
`2**32` or `2**64`).

##### Args:


*  <b>`shape`</b>: A 1-D integer Tensor or Python array. The shape of the output tensor.
*  <b>`minval`</b>: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
    range of random values to generate.  Defaults to 0.
*  <b>`maxval`</b>: A 0-D Tensor or Python value of type `dtype`. The upper bound on
    the range of random values to generate.  Defaults to 1 if `dtype` is
    floating point.
*  <b>`dtype`</b>: The type of the output: `float32`, `float64`, `int32`, or `int64`.
*  <b>`seed`</b>: A Python integer. Used to create a random seed for the distribution.
    See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tensor of the specified shape filled with random uniform values.

##### Raises:


*  <b>`ValueError`</b>: If `dtype` is integral and `maxval` is not specified.

