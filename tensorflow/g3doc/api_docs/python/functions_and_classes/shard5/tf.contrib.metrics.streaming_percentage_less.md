### `tf.contrib.metrics.streaming_percentage_less(*args, **kwargs)` {#streaming_percentage_less}

Computes the percentage of values less than the given threshold. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

  The `streaming_percentage_less` function creates two local variables,
  `total` and `count` that are used to compute the percentage of `values` that
  fall below `threshold`. This rate is weighted by `weights`, and it is
  ultimately returned as `percentage` which is an idempotent operation that
  simply divides `total` by `count`.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `percentage`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    values: A numeric `Tensor` of arbitrary size.
    threshold: A scalar threshold.
    ignore_mask: An optional, `bool` `Tensor` whose shape matches `values`.
    weights: An optional `Tensor` whose shape is broadcastable to `values`.
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.

  Returns:
    percentage: A tensor representing the current mean, the value of `total`
      divided by `count`.
    update_op: An operation that increments the `total` and `count` variables
      appropriately.

  Raises:
    ValueError: If `ignore_mask` is not `None` and its shape doesn't match
      `values`, or if `weights` is not `None` and its shape doesn't match
      `values`, or if either `metrics_collections` or `updates_collections` are
      not a list or tuple.

