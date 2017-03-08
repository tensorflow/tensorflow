### `tf.nn.weighted_moments(x, axes, frequency_weights, name=None, keep_dims=False)` {#weighted_moments}

Returns the frequency-weighted mean and variance of `x`.

##### Args:


*  <b>`x`</b>: A tensor.
*  <b>`axes`</b>: 1-d tensor of int32 values; these are the axes along which
    to compute mean and variance.
*  <b>`frequency_weights`</b>: A tensor of positive weights which can be
    broadcast with x.
*  <b>`name`</b>: Name used to scope the operation.
*  <b>`keep_dims`</b>: Produce moments with the same dimensionality as the input.

##### Returns:

  Two tensors: `weighted_mean` and `weighted_variance`.

