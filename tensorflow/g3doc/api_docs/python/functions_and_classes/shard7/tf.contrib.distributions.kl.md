### `tf.contrib.distributions.kl(dist_a, dist_b, allow_nan=False, name=None)` {#kl}

Get the KL-divergence KL(dist_a || dist_b).

##### Args:


*  <b>`dist_a`</b>: instance of distributions.Distribution.
*  <b>`dist_b`</b>: instance of distributions.Distribution.
*  <b>`allow_nan`</b>: If `False` (default), a runtime error is raised
    if the KL returns NaN values for any batch entry of the given
    distributions.  If `True`, the KL may return a NaN for the given entry.
*  <b>`name`</b>: (optional) Name scope to use for created operations.

##### Returns:

  A Tensor with the batchwise KL-divergence between dist_a and dist_b.

##### Raises:


*  <b>`TypeError`</b>: If dist_a or dist_b is not an instance of Distribution.
*  <b>`NotImplementedError`</b>: If no KL method is defined for distribution types
    of dist_a and dist_b.

