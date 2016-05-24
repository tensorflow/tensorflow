### `tf.contrib.metrics.streaming_recall_at_k(predictions, labels, k, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_recall_at_k}

Computes the recall@k of the predictions with respect to dense labels.

The `streaming_recall_at_k` function creates two local variables, `total` and
`count`, that are used to compute the recall@k frequency. This frequency is
ultimately returned as `recall_at_<k>`: an idempotent operation that simply
divides `total` by `count`. To facilitate the estimation of recall@k over a
stream of data, the function utilizes two operations. First, an `in_top_k`
operation computes a tensor with shape [batch_size] whose elements indicate
whether or not the corresponding label is in the top `k` predictions of the
`predictions` `Tensor`. Second, an `update_op` operation whose behavior is
dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
`update_op` increments `total` with the number of elements of `in_top_k` that
are set to `True` and increments `count` with the batch size. If `ignore_mask`
is not `None`, then `update_op` increments `total` with the number of elements
in `in_top_k` that are `True` whose corresponding element in `ignore_mask` is
`False`. In addition to performing the updates, `update_op` also returns the
recall value.

##### Args:


*  <b>`predictions`</b>: A floating point tensor of dimension [batch_size, num_classes]
*  <b>`labels`</b>: A tensor of dimension [batch_size] whose type is in `int32`,
    `int64`.
*  <b>`k`</b>: The number of top elements to look at for computing precision.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `labels`. If an
    element of `ignore_mask` is True, the corresponding prediction and label
    pair is used to compute the metrics. Otherwise, the pair is ignored.
*  <b>`metrics_collections`</b>: An optional list of collections that `recall_at_k`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections `update_op` should be
    added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`recall_at_k`</b>: A tensor representing the recall@k, the fraction of labels
    which fall into the top `k` predictions.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `recall_at_k`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `ignore_mask` is not `None` and its shape doesn't match `predictions`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.

