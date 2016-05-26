### `tf.contrib.metrics.streaming_precision(predictions, labels, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_precision}

Computes the precision of the predictions with respect to the labels.

The `streaming_precision` function creates two local variables,
`true_positives` and `false_positives`, that are used to compute the
precision. This value is ultimately returned as `precision`, an idempotent
operation that simply divides `true_positives` by the sum of `true_positives`
and `false_positives`. To facilitate the calculation of the precision over a
stream of data, the function creates an `update_op` operation whose behavior
is dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
`update_op` increments `true_positives` with the number of elements of
`predictions` and `labels` that are both `True` and increments
`false_positives` with the number of elements of `predictions` that are `True`
whose corresponding `labels` element is `False`. If `ignore_mask` is not
`None`, then the increments for `true_positives` and `false_positives` are
only computed using elements of `predictions` and `labels` whose corresponding
values in `ignore_mask` are `False`. In addition to performing the updates,
`update_op` also returns the value of `precision`.

##### Args:


*  <b>`predictions`</b>: The predicted values, a binary `Tensor` of arbitrary shape.
*  <b>`labels`</b>: The ground truth values, a binary `Tensor` whose dimensions must
    match `predictions`.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that `precision` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`precision`</b>: Scalar float `Tensor` with the value of `true_positives`
    divided by the sum of `true_positives` and `false_positives`.
*  <b>`update_op`</b>: `Operation` that increments `true_positives` and
    `false_positives` variables appropriately and whose value matches
    `precision`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `ignore_mask` is not `None` and its shape doesn't match `predictions`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.

