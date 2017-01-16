### `tf.contrib.metrics.streaming_recall_at_k(*args, **kwargs)` {#streaming_recall_at_k}

Computes the recall@k of the predictions with respect to dense labels. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-08.
Instructions for updating:
Please use `streaming_sparse_recall_at_k`, and reshape labels from [batch_size] to [batch_size, 1].

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

##### Args:


*  <b>`predictions`</b>: A float `Tensor` of dimension [batch_size, num_classes].
*  <b>`labels`</b>: A `Tensor` of dimension [batch_size] whose type is in `int32`,
    `int64`.
*  <b>`k`</b>: The number of top elements to look at for computing recall.
*  <b>`weights`</b>: `Tensor` whose rank is either 0, or the same rank as `labels`, and
    must be broadcastable to `labels` (i.e., all dimensions must be either
    `1`, or the same as the corresponding `labels` dimension).
*  <b>`metrics_collections`</b>: An optional list of collections that `recall_at_k`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections `update_op` should be
    added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`recall_at_k`</b>: A `Tensor` representing the recall@k, the fraction of labels
    which fall into the top `k` predictions.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `recall_at_k`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.

