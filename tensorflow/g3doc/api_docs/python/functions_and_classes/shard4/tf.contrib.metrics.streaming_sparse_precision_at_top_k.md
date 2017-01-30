### `tf.contrib.metrics.streaming_sparse_precision_at_top_k(top_k_predictions, labels, class_id=None, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_sparse_precision_at_top_k}

Computes precision@k of top-k predictions with respect to sparse labels.

If `class_id` is not specified, we calculate precision as the ratio of
    true positives (i.e., correct predictions, items in `top_k_predictions`
    that are found in the corresponding row in `labels`) to positives (all
    `top_k_predictions`).
If `class_id` is specified, we calculate precision by considering only the
    rows in the batch for which `class_id` is in the top `k` highest
    `predictions`, and computing the fraction of them for which `class_id` is
    in the corresponding row in `labels`.

We expect precision to decrease as `k` increases.

`streaming_sparse_precision_at_top_k` creates two local variables,
`true_positive_at_k` and `false_positive_at_k`, that are used to compute
the precision@k frequency. This frequency is ultimately returned as
`precision_at_k`: an idempotent operation that simply divides
`true_positive_at_k` by total (`true_positive_at_k` + `false_positive_at_k`).

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`precision_at_k`. Internally, set operations applied to `top_k_predictions`
and `labels` calculate the true positives and false positives weighted by
`weights`. Then `update_op` increments `true_positive_at_k` and
`false_positive_at_k` using these values.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`top_k_predictions`</b>: Integer `Tensor` with shape [D1, ... DN, k] where
    N >= 1. Commonly, N=1 and top_k_predictions has shape [batch size, k].
    The final dimension contains the indices of top-k labels. [D1, ... DN]
    must match `labels`.
*  <b>`labels`</b>: `int64` `Tensor` or `SparseTensor` with shape
    [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
    target classes for the associated prediction. Commonly, N=1 and `labels`
    has shape [batch_size, num_labels]. [D1, ... DN] must match
    `top_k_predictions`. Values should be in range [0, num_classes), where
    num_classes is the last dimension of `predictions`. Values outside this
    range are ignored.
*  <b>`class_id`</b>: Integer class ID for which we want binary metrics. This should be
    in range [0, num_classes), where num_classes is the last dimension of
    `predictions`. If `class_id` is outside this range, the method returns
    NAN.
*  <b>`weights`</b>: `Tensor` whose rank is either 0, or n-1, where n is the rank of
    `labels`. If the latter, it must be broadcastable to `labels` (i.e., all
    dimensions must be either `1`, or the same as the corresponding `labels`
    dimension).
*  <b>`metrics_collections`</b>: An optional list of collections that values should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that updates should
    be added to.
*  <b>`name`</b>: Name of new update operation, and namespace for other dependent ops.

##### Returns:


*  <b>`precision`</b>: Scalar `float64` `Tensor` with the value of `true_positives`
    divided by the sum of `true_positives` and `false_positives`.
*  <b>`update_op`</b>: `Operation` that increments `true_positives` and
    `false_positives` variables appropriately, and whose value matches
    `precision`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match
    `predictions`, or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.
*  <b>`ValueError`</b>: If `top_k_predictions` has rank < 2.

