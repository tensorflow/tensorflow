### `tf.contrib.metrics.streaming_mean_cosine_distance(predictions, labels, dim, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_cosine_distance}

Computes the cosine distance between the labels and predictions.

The `streaming_mean_cosine_distance` function creates two local variables,
`total` and `count` that are used to compute the average cosine distance
between `predictions` and `labels`. This average is weighted by `weights`,
and it is ultimately returned as `mean_distance`, which is an idempotent
operation that simply divides `total` by `count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`mean_distance`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of the same shape as `labels`.
*  <b>`labels`</b>: A `Tensor` of arbitrary shape.
*  <b>`dim`</b>: The dimension along which the cosine distance is computed.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`,
    and whose dimension `dim` is 1.
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`mean_distance`</b>: A tensor representing the current mean, the value of `total`
    divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.

