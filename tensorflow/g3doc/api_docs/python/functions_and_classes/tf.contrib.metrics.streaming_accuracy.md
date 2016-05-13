### `tf.contrib.metrics.streaming_accuracy(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_accuracy}

Calculates how often `predictions` matches `labels`.

The `streaming_accuracy` function creates two local variables, `total` and
`count` that are used to compute the frequency with which `predictions`
matches `labels`. This frequency is ultimately returned as `accuracy`: an
idempotent operation that simply divides `total` by `count`.
To facilitate the estimation of the accuracy over a stream of data, the
function utilizes two operations. First, an `is_correct` operation that
computes a tensor whose shape matches `predictions` and whose elements are
set to 1.0 when the corresponding values of `predictions` and `labels match
and 0.0 otherwise. Second, an `update_op` operation whose behavior is
dependent on the value of `weights`. If `weights` is None, then `update_op`
increments `total` with the number of elements of `predictions` that match
`labels` and increments `count` with the number of elements in `values`. If
`weights` is not `None`, then `update_op` increments `total` with the reduced
sum of the product of `weights` and `is_correct` and increments `count` with
the reduced sum of `weights`. In addition to performing the updates,
`update_op` also returns the `accuracy` value.

##### Args:


*  <b>`predictions`</b>: The predicted values, a `Tensor` of any shape.
*  <b>`labels`</b>: The ground truth values, a `Tensor` whose shape matches
    `predictions`.
*  <b>`weights`</b>: An optional set of weights whose shape matches `predictions`
    which, when not `None`, produces a weighted mean accuracy.
*  <b>`metrics_collections`</b>: An optional list of collections that `accuracy` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`accuracy`</b>: A tensor representing the accuracy, the value of `total` divided
    by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `accuracy`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `weight` is not `None` and its shape doesn't match `predictions` or
    if either `metrics_collections` or `updates_collections` are not
    a list or tuple.

