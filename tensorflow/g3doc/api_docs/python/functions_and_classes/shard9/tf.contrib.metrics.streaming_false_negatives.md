### `tf.contrib.metrics.streaming_false_negatives(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_false_negatives}

Computes the total number of false positives.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: The predicted values, a `bool` `Tensor` of arbitrary
    dimensions.
*  <b>`labels`</b>: The ground truth values, a `bool` `Tensor` whose dimensions must
    match `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`value_tensor`</b>: A `Tensor` representing the current value of the metric.
*  <b>`update_op`</b>: An operation that accumulates the error from a batch of data.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match `values`,
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.

