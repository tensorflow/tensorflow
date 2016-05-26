### `tf.uniform_unit_scaling_initializer(factor=1.0, seed=None, dtype=tf.float32, full_shape=None)` {#uniform_unit_scaling_initializer}

Returns an initializer that generates tensors without scaling variance.

When initializing a deep network, it is in principle advantageous to keep
the scale of the input variance constant, so it does not explode or diminish
by reaching the final layer. If the input is `x` and the operation `x * W`,
and we want to initialize `W` uniformly at random, we need to pick `W` from

    [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
A similar calculation for convolutional networks gives an analogous result
with `dim` equal to the product of the first 3 dimensions.  When
nonlinearities are present, we need to multiply this by a constant `factor`.
See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation, experiments
and the calculation of constants. In section 2.3 there, the constants were
numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

If the shape tuple `full_shape` is provided, the scale will be calculated from
this predefined shape.  This is useful when a `Variable` is being partitioned
across several shards, and each shard has a smaller shape than the whole.
Since the shards are usually concatenated when used, the scale should be
based on the shape of the whole.

##### Args:


*  <b>`factor`</b>: Float.  A multiplicative factor by which the values will be scaled.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.
*  <b>`full_shape`</b>: Tuple or list of integers.  The shape used for calculating
    scale normalization (instead of the shape passed at creation time).
    Useful when creating sharded variables via partitioning.

##### Returns:

  An initializer that generates tensors with unit variance.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.

