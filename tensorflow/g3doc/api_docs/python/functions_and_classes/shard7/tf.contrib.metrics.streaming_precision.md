### `tf.contrib.metrics.streaming_precision(*args, **kwargs)` {#streaming_precision}

Computes the precision of the predictions with respect to the labels. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

  The `streaming_precision` function creates two local variables,
  `true_positives` and `false_positives`, that are used to compute the
  precision. This value is ultimately returned as `precision`, an idempotent
  operation that simply divides `true_positives` by the sum of `true_positives`
  and `false_positives`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision`. `update_op` weights each prediction by the corresponding value in
  `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: The predicted values, a `bool` `Tensor` of arbitrary shape.
    labels: The ground truth values, a `bool` `Tensor` whose dimensions must
      match `predictions`.
    ignore_mask: An optional, `bool` `Tensor` whose shape matches `predictions`.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`.
    metrics_collections: An optional list of collections that `precision` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    precision: Scalar float `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately and whose value matches
      `precision`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`, or
      if `weights` is not `None` and its shape doesn't match `predictions`, or
      if either `metrics_collections` or `updates_collections` are not a list or
      tuple.

