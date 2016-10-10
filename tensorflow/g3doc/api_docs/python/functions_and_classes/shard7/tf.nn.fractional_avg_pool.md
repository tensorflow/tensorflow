### `tf.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random=None, overlapping=None, deterministic=None, seed=None, seed2=None, name=None)` {#fractional_avg_pool}

Performs fractional average pooling on the input.

Fractional average pooling is similar to Fractional max pooling in the pooling
region generation step. The only difference is that after pooling regions are
generated, a mean operation is performed instead of a max operation in each
pooling region.

##### Args:


*  <b>`value`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
    4-D with shape `[batch, height, width, channels]`.
*  <b>`pooling_ratio`</b>: A list of `floats` that has length `>= 4`.
    Pooling ratio for each dimension of `value`, currently only
    supports row and col dimension and should be >= 1.0. For example, a valid
    pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
    must be 1.0 because we don't allow pooling on batch and channels
    dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
    respectively.
*  <b>`pseudo_random`</b>: An optional `bool`. Defaults to `False`.
    When set to True, generates the pooling sequence in a
    pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
    Graham, Fractional Max-Pooling] (http://arxiv.org/abs/1412.6071) for
    difference between pseudorandom and random.
*  <b>`overlapping`</b>: An optional `bool`. Defaults to `False`.
    When set to True, it means when pooling, the values at the boundary
    of adjacent pooling cells are used by both cells. For example:

    `index  0  1  2  3  4`

    `value  20 5  16 3  7`

    If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
    The result would be [41/3, 26/3] for fractional avg pooling.

*  <b>`deterministic`</b>: An optional `bool`. Defaults to `False`.
    When set to True, a fixed pooling region will be used when
    iterating over a FractionalAvgPool node in the computation graph. Mainly used
    in unit test to make FractionalAvgPool deterministic.
*  <b>`seed`</b>: An optional `int`. Defaults to `0`.
    If either seed or seed2 are set to be non-zero, the random number
    generator is seeded by the given seed.  Otherwise, it is seeded by a
    random seed.
*  <b>`seed2`</b>: An optional `int`. Defaults to `0`.
    An second seed to avoid seed collision.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

*  <b>`output`</b>: A `Tensor`. Has the same type as `value`. output tensor after fractional avg pooling.
*  <b>`row_pooling_sequence`</b>: A `Tensor` of type `int64`. row pooling sequence, needed to calculate gradient.
*  <b>`col_pooling_sequence`</b>: A `Tensor` of type `int64`. column pooling sequence, needed to calculate gradient.

