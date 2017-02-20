Initializer that generates tensors without scaling variance.

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

Args:
  factor: Float.  A multiplicative factor by which the values will be scaled.
  seed: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
  dtype: The data type. Only floating point types are supported.
- - -

#### `tf.uniform_unit_scaling_initializer.__call__(shape, dtype=None, partition_info=None)` {#uniform_unit_scaling_initializer.__call__}




- - -

#### `tf.uniform_unit_scaling_initializer.__init__(factor=1.0, seed=None, dtype=tf.float32)` {#uniform_unit_scaling_initializer.__init__}




