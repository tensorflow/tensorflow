### `tf.contrib.metrics.streaming_sparse_precision_at_k(predictions, labels, k, class_id=None, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_sparse_precision_at_k}

Computes precision@k of the predictions with respect to sparse labels.

If `class_id` is specified, we calculate precision by considering only the
    entries in the batch for which `class_id` is in the top-k highest
    `predictions`, and computing the fraction of them for which `class_id` is
    indeed a correct label.
If `class_id` is not specified, we'll calculate precision as how often on
    average a class among the top-k classes with the highest predicted values
    of a batch entry is correct and can be found in the label for that entry.

`streaming_sparse_precision_at_k` creates two local variables,
`true_positive_at_<k>` and `false_positive_at_<k>`, that are used to compute
the precision@k frequency. This frequency is ultimately returned as
`recall_at_<k>`: an idempotent operation that simply divides
`true_positive_at_<k>` by total (`true_positive_at_<k>` + `recall_at_<k>`). To
facilitate the estimation of precision@k over a stream of data, the function
utilizes three steps.
* A `top_k` operation computes a tensor whose elements indicate the top `k`
  predictions of the `predictions` `Tensor`.
* Set operations are applied to `top_k` and `labels` to calculate true
  positives and false positives.
* An `update_op` operation increments `true_positive_at_<k>` and
  `false_positive_at_<k>`. It also returns the recall value.

##### Args:


*  <b>`predictions`</b>: Float `Tensor` with shape [D1, ... DN, num_classes] where
    N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
    The final dimension contains the logit values for each class. [D1, ... DN]
    must match `labels`.
*  <b>`labels`</b>: `int64` `Tensor` or `SparseTensor` with shape
    [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
    target classes for the associated prediction. Commonly, N=1 and `labels`
    has shape [batch_size, num_labels]. [D1, ... DN] must match
    `predictions_idx`. Values should be in range [0, num_classes], where
    num_classes is the last dimension of `predictions`.
*  <b>`k`</b>: Integer, k for @k metric.
*  <b>`class_id`</b>: Integer class ID for which we want binary metrics. This should be
    in range [0, num_classes], where num_classes is the last dimension of
    `predictions`.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose shape is broadcastable to the
    the first [D1, ... DN] dimensions of `predictions_idx` and `labels`.
*  <b>`metrics_collections`</b>: An optional list of collections that values should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that updates should
    be added to.
*  <b>`name`</b>: Name of new update operation, and namespace for other dependant ops.

##### Returns:


*  <b>`precision`</b>: Scalar `float64` `Tensor` with the value of `true_positives`
    divided by the sum of `true_positives` and `false_positives`.
*  <b>`update_op`</b>: `Operation` that increments `true_positives` and
    `false_positives` variables appropriately, and whose value matches
    `precision`.

