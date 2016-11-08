### `tf.contrib.distributions.kl(dist_a, dist_b, allow_nan=False, name=None)` {#kl}

Get the KL-divergence KL(dist_a || dist_b).

If there is no KL method registered specifically for `type(dist_a)` and
`type(dist_b)`, then the class hierarchies of these types are searched.

If one KL method is registered between any pairs of classes in these two
parent hierarchies, it is used.

If more than one such registered method exists, the method whose registered
classes have the shortest sum MRO paths to the input types is used.

If more than one such shortest path exists, the first method
identified in the search is used (favoring a shorter MRO distance to
`type(dist_a)`).

##### Args:


*  <b>`dist_a`</b>: The first distribution.
*  <b>`dist_b`</b>: The second distribution.
*  <b>`allow_nan`</b>: If `False` (default), a runtime error is raised
    if the KL returns NaN values for any batch entry of the given
    distributions.  If `True`, the KL may return a NaN for the given entry.
*  <b>`name`</b>: (optional) Name scope to use for created operations.

##### Returns:

  A Tensor with the batchwise KL-divergence between dist_a and dist_b.

##### Raises:


*  <b>`NotImplementedError`</b>: If no KL method is defined for distribution types
    of dist_a and dist_b.

