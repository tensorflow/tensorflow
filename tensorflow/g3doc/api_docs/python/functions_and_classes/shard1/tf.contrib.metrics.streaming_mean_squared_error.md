### `tf.contrib.metrics.streaming_mean_squared_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_squared_error}

Computes the mean squared error between the labels and predictions.

The `streaming_mean_squared_error` function creates two local variables,
`total` and `count` that are used to compute the mean squared error.
This average is weighted by `weights`, and it is ultimately returned as
`mean_squared_error`: an idempotent operation that simply divides `total` by
`count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`mean_squared_error`. Internally, a `squared_error` operation computes the
element-wise square of the difference between `predictions` and `labels`. Then
`update_op` increments `total` with the reduced sum of the product of
`weights` and `squared_error`, and it increments `count` with the reduced sum
of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_squared_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`mean_squared_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_squared_error`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.

