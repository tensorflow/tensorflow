### `tf.contrib.metrics.streaming_mean(values, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean}

Computes the (weighted) mean of the given values.

The `streaming_mean` function creates two local variables, `total` and `count`
that are used to compute the average of `values`. This average is ultimately
returned as `mean` which is an idempotent operation that simply divides
`total` by `count`. To facilitate the estimation of a mean over a stream
of data, the function creates an `update_op` operation whose behavior is
dependent on the value of `weights`. If `weights` is None, then `update_op`
increments `total` with the reduced sum of `values` and increments `count`
with the number of elements in `values`. If `weights` is not `None`, then
`update_op` increments `total` with the reduced sum of the product of `values`
and `weights` and increments `count` with the reduced sum of weights.
In addition to performing the updates, `update_op` also returns the
`mean`.

##### Args:


*  <b>`values`</b>: A `Tensor` of arbitrary dimensions.
*  <b>`weights`</b>: An optional set of weights of the same shape as `values`. If
    `weights` is not None, the function computes a weighted mean.
*  <b>`metrics_collections`</b>: An optional list of collections that `mean`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op`
    should be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean`</b>: A tensor representing the current mean, the value of `total` divided
    by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_value`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match `values`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.

