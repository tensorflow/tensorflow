### `tf.nn.sufficient_statistics(x, axes, shift=False, keep_dims=False, name=None)` {#sufficient_statistics}

Calculate the sufficient statistics for the mean and variance of `x`.

These sufficient statistics are computed using the one pass algorithm on
an input that's optionally shifted using the value of the 1st element in `x`.
See:
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data
Unfortunately, in some cases using a random individual sample as the shift
value leads experimentally to very poor numerical stability, so it is disabled
by default. The one-pass approach might have to be revised accordingly.

##### Args:


*  <b>`x`</b>: A `Tensor`.
*  <b>`axes`</b>: Array of ints. Axes along which to compute mean and variance.
*  <b>`shift`</b>: If true, shift the data to provide more numerically stable results.
*  <b>`keep_dims`</b>: produce statistics with the same dimensionality as the input.
*  <b>`name`</b>: Name used to scope the operations that compute the sufficient stats.

##### Returns:

  Four `Tensor` objects of the same type as `x`:
  * the count (number of elements to average over).
  * the (possibly shifted) sum of the elements in the array.
  * the (possibly shifted) sum of squares of the elements in the array.
  * the shift by which the mean must be corrected or None if `shift` is False.

