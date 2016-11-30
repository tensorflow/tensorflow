### `tf.contrib.metrics.streaming_accuracy(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_accuracy}

Calculates how often `predictions` matches `labels`.

The `streaming_accuracy` function creates two local variables, `total` and
`count` that are used to compute the frequency with which `predictions`
matches `labels`. This frequency is ultimately returned as `accuracy`: an
idempotent operation that simply divides `total` by `count`.

For estimation of the metric  over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the `accuracy`.
Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
where the corresponding elements of `predictions` and `labels` match and 0.0
otherwise. Then `update_op` increments `total` with the reduced sum of the
product of `weights` and `is_correct`, and it increments `count` with the
reduced sum of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: The predicted values, a `Tensor` of any shape.
*  <b>`labels`</b>: The ground truth values, a `Tensor` whose shape matches
    `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that `accuracy` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`accuracy`</b>: A `Tensor` representing the accuracy, the value of `total` divided
    by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `accuracy`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.

