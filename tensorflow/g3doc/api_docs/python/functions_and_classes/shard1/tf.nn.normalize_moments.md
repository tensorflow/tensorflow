### `tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift, name=None)` {#normalize_moments}

Calculate the mean and variance of based on the sufficient statistics.

##### Args:


*  <b>`counts`</b>: An `Output` containing a the total count of the data (one value).
*  <b>`mean_ss`</b>: An `Output` containing the mean sufficient statistics: the
    (possibly shifted) sum of the elements to average over.
*  <b>`variance_ss`</b>: An `Output` containing the variance sufficient statistics: the
    (possibly shifted) squared sum of the data to compute the variance over.
*  <b>`shift`</b>: An `Output` containing the value by which the data is shifted for
    numerical stability, or `None` if no shift was performed.
*  <b>`name`</b>: Name used to scope the operations that compute the moments.

##### Returns:

  Two `Output` objects: `mean` and `variance`.

