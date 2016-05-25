### `tf.contrib.metrics.streaming_mean_squared_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_squared_error}

Computes the mean squared error between the labels and predictions.

The `streaming_mean_squared_error` function creates two local variables,
`total` and `count` that are used to compute the mean squared error.
This average is ultimately returned as `mean_squared_error`: an idempotent
operation that simply divides `total` by `count`. To facilitate the estimation
of the mean squared error over a stream of data, the function utilizes two
operations. First, a `squared_error` operation computes the element-wise
square of the difference between `predictions` and `labels`. Second, an
`update_op` operation whose behavior is dependent on the value of `weights`.
If `weights` is None, then `update_op` increments `total` with the
reduced sum of `squared_error` and increments `count` with the number of
elements in `squared_error`. If `weights` is not `None`, then `update_op`
increments `total` with the reduced sum of the product of `weights` and
`squared_error` and increments `count` with the reduced sum of `weights`. In
addition to performing the updates, `update_op` also returns the
`mean_squared_error` value.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional set of weights of the same shape as `predictions`. If
    `weights` is not None, the function computes a weighted mean.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_squared_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean_squared_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_squared_error`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match
    `predictions` or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.

