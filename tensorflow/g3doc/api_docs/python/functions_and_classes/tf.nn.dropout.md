### `tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)` {#dropout}

Computes dropout.

With probability `keep_prob`, outputs the input element scaled up by
`1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
sum is unchanged.

By default, each element is kept or dropped independently.  If `noise_shape`
is specified, it must be
[broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
kept independently and each row and column will be kept or not kept together.

##### Args:


*  <b>`x`</b>: A tensor.
*  <b>`keep_prob`</b>: A scalar `Tensor` with the same type as x. The probability
    that each element is kept.
*  <b>`noise_shape`</b>: A 1-D `Tensor` of type `int32`, representing the
    shape for randomly generated keep/drop flags.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A Tensor of the same shape of `x`.

##### Raises:


*  <b>`ValueError`</b>: If `keep_prob` is not in `(0, 1]`.

