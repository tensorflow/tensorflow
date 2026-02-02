### `tf.contrib.metrics.streaming_mean_iou(predictions, labels, num_classes, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_iou}

Calculate per-step mean Intersection-Over-Union (mIOU).

Mean Intersection-Over-Union is a common evaluation metric for
semantic image segmentation, which first computes the IOU for each
semantic class and then computes the average over classes.

##### IOU is defined as follows:

  IOU = true_positive / (true_positive + false_positive + false_negative).
The predictions are accumulated in a confusion matrix, and mIOU is then
calculated from it.

##### Args:


*  <b>`predictions`</b>: A tensor of prediction results for semantic labels, whose
    shape is [batch size] and type `int32` or `int64`. The tensor will be
    flattened, if its rank > 1.
*  <b>`labels`</b>: A tensor of ground truth labels with shape [batch size] and of
    type `int32` or `int64`. The tensor will be flattened, if its rank > 1.
*  <b>`num_classes`</b>: The possible number of labels the prediction task can
    have. This value must be provided, since a confusion matrix of
    dimension = [num_classes, num_classes] will be allocated.
*  <b>`ignore_mask`</b>: An optional, boolean tensor whose size matches `labels`. If an
    element of `ignore_mask` is True, the corresponding prediction and label
    pair is NOT used to compute the metrics. Otherwise, the pair is included.
*  <b>`metrics_collections`</b>: An optional list of collections that `mean_iou`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections `update_op` should be
    added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean_iou`</b>: A tensor representing the mean intersection-over-union.
*  <b>`update_op`</b>: An operation that increments the confusion matrix.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `ignore_mask` is not `None` and its shape doesn't match `labels`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.

