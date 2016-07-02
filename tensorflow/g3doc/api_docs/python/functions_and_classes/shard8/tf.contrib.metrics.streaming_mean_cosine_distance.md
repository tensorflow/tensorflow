### `tf.contrib.metrics.streaming_mean_cosine_distance(predictions, labels, dim, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_cosine_distance}

Computes the cosine distance between the labels and predictions.

The `streaming_mean_cosine_distance` function creates two local variables,
`total` and `count` that are used to compute the average cosine distance
between `predictions` and `labels`. This average is ultimately returned as
`mean_distance` which is an idempotent operation that simply divides `total`
by `count. To facilitate the estimation of a mean over multiple batches
of data, the function creates an `update_op` operation whose behavior is
dependent on the value of `weights`. If `weights` is None, then `update_op`
increments `total` with the reduced sum of `values and increments `count` with
the number of elements in `values`. If `weights` is not `None`, then
`update_op` increments `total` with the reduced sum of the product of `values`
and `weights` and increments `count` with the reduced sum of weights.

##### Args:


*  <b>`predictions`</b>: A tensor of the same size as labels.
*  <b>`labels`</b>: A tensor of arbitrary size.
*  <b>`dim`</b>: The dimension along which the cosine distance is computed.
*  <b>`weights`</b>: An optional set of weights which indicates which predictions to
    ignore during metric computation. Its size matches that of labels except
    for the value of 'dim' which should be 1. For example if labels has
    dimensions [32, 100, 200, 3], then `weights` should have dimensions
    [32, 100, 200, 1].
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean_distance`</b>: A tensor representing the current mean, the value of `total`
    divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately.

##### Raises:


*  <b>`ValueError`</b>: If labels and predictions are of different sizes or if the
    ignore_mask is of the wrong size or if either `metrics_collections` or
    `updates_collections` are not a list or tuple.

