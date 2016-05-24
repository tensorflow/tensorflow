### `tf.contrib.metrics.streaming_percentage_less(values, threshold, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_percentage_less}

Computes the percentage of values less than the given threshold.

The `streaming_percentage_less` function creates two local variables,
`total` and `count` that are used to compute the percentage of `values` that
fall below `threshold`. This rate is ultimately returned as `percentage`
which is an idempotent operation that simply divides `total` by `count.
To facilitate the estimation of the percentage of values that fall under
`threshold` over multiple batches of data, the function creates an
`update_op` operation whose behavior is dependent on the value of
`ignore_mask`. If `ignore_mask` is None, then `update_op`
increments `total` with the number of elements of `values` that are less
than `threshold` and `count` with the number of elements in `values`. If
`ignore_mask` is not `None`, then `update_op` increments `total` with the
number of elements of `values` that are less than `threshold` and whose
corresponding entries in `ignore_mask` are False, and `count` is incremented
with the number of elements of `ignore_mask` that are False.

##### Args:


*  <b>`values`</b>: A numeric `Tensor` of arbitrary size.
*  <b>`threshold`</b>: A scalar threshold.
*  <b>`ignore_mask`</b>: An optional mask of the same shape as 'values' which indicates
    which elements to ignore during metric computation.
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`percentage`</b>: A tensor representing the current mean, the value of `total`
    divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately.

##### Raises:


*  <b>`ValueError`</b>: If `ignore_mask` is not None and its shape doesn't match `values
    or if either `metrics_collections` or `updates_collections` are supplied
    but are not a list or tuple.

