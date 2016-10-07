### `tf.contrib.metrics.streaming_recall_at_k(*args, **kwargs)` {#streaming_recall_at_k}

Computes the recall@k of the predictions with respect to dense labels. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

  The `streaming_recall_at_k` function creates two local variables, `total` and
  `count`, that are used to compute the recall@k frequency. This frequency is
  ultimately returned as `recall_at_<k>`: an idempotent operation that simply
  divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall_at_<k>`. Internally, an `in_top_k` operation computes a `Tensor` with
  shape [batch_size] whose elements indicate whether or not the corresponding
  label is in the top `k` `predictions`. Then `update_op` increments `total`
  with the reduced sum of `weights` where `in_top_k` is `True`, and it
  increments `count` with the reduced sum of `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: A floating point tensor of dimension [batch_size, num_classes]
    labels: A tensor of dimension [batch_size] whose type is in `int32`,
      `int64`.
    k: The number of top elements to look at for computing recall.
    ignore_mask: An optional, `bool` `Tensor` whose shape matches `predictions`.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`.
    metrics_collections: An optional list of collections that `recall_at_k`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    recall_at_k: A tensor representing the recall@k, the fraction of labels
      which fall into the top `k` predictions.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `recall_at_k`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`, or
      if `weights` is not `None` and its shape doesn't match `predictions`, or
      if either `metrics_collections` or `updates_collections` are not a list or
      tuple.

