### `tf.nn.moments(x, axes, shift=None, name=None, keep_dims=False)` {#moments}

Calculate the mean and variance of `x`.

The mean and variance are calculated by aggregating the contents of `x`
across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
and variance of a vector.

When using these moments for batch normalization (see
`tf.nn.batch_normalization`):
  * for so-called "global normalization", used with convolutional filters with
    shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
  * for simple batch normalization pass `axes=[0]` (batch only).

##### Args:


*  <b>`x`</b>: A `Tensor`.
*  <b>`axes`</b>: array of ints.  Axes along which to compute mean and
    variance.
*  <b>`shift`</b>: A `Tensor` containing the value by which to shift the data for
    numerical stability, or `None` if no shift is to be performed. A shift
    close to the true mean provides the most numerically stable results.
*  <b>`name`</b>: Name used to scope the operations that compute the moments.
*  <b>`keep_dims`</b>: produce moments with the same dimensionality as the input.

##### Returns:

  Two `Tensor` objects: `mean` and `variance`.

