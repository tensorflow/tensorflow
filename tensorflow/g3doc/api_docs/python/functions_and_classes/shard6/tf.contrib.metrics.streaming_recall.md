### `tf.contrib.metrics.streaming_recall(predictions, labels, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_recall}

Computes the recall of the predictions with respect to the labels.

The `streaming_recall` function creates two local variables,
`true_positives` and `false_negatives`, that are used to compute the
recall. This value is ultimately returned as `recall`, an idempotent
operation that simply divides `true_positives` by the sum of `true_positives`
and `false_negatives`. To facilitate the calculation of the recall over a
stream of data, the function creates an `update_op` operation whose behavior
is dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
`update_op` increments `true_positives` with the number of elements of
`predictions` and `labels` that are both `True` and increments
`false_negatives` with the number of elements of `predictions` that are
`False` whose corresponding `labels` element is `False`. If `ignore_mask` is
not `None`, then the increments for `true_positives` and `false_negatives` are
only computed using elements of `predictions` and `labels` whose corresponding
values in `ignore_mask` are `False`. In addition to performing the updates,
`update_op` also returns the value of `recall`.

##### Args:


*  <b>`predictions`</b>: The predicted values, a binary `Tensor` of arbitrary shape.
*  <b>`labels`</b>: The ground truth values, a binary `Tensor` whose dimensions must
    match `predictions`.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that `recall` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`recall`</b>: Scalar float `Tensor` with the value of `true_positives` divided
    by the sum of `true_positives` and `false_negatives`.
*  <b>`update_op`</b>: `Operation` that increments `true_positives` and
    `false_negatives` variables appropriately and whose value matches
    `recall`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `ignore_mask` is not `None` and its shape doesn't match `predictions`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.

