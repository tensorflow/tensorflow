### `tf.contrib.metrics.streaming_auc(predictions, labels, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None)` {#streaming_auc}

Computes the approximate AUC via a Riemann sum.

The `streaming_auc` function creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the AUC. To discretize the AUC curve, a linearly spaced set of
thresholds is used to compute pairs of recall and precision values. The area
under the ROC-curve is therefore computed using the height of the recall
values by the false positive rate, while the area under the PR-curve is the
computed using the height of the precision values by the recall.

This value is ultimately returned as `auc`, an idempotent
operation the computes the area under a discretized curve of precision versus
recall values (computed using the afformentioned variables). The
`num_thresholds` variable controls the degree of discretization with larger
numbers of thresholds more closely approximating the true AUC.

To faciliate the estimation of the AUC over a stream of data, the function
creates an `update_op` operation. `update_op` increments the
`true_positives`, `true_negatives`, `false_positives` and `false_negatives`
counts with the weighted number of each found in the current `predictions`
and `labels` `Tensors`. If `weights` is `None`, it is assumed that all
entries have weight 1. Note that a weight of 0 can be used to effectively
mask out and ignore specific entries. In addition to performing the updates,
`update_op` also returns the `auc`.

##### Args:


*  <b>`predictions`</b>: A floating point `Tensor` of arbitrary shape and whose values
    are in the range `[0, 1]`.
*  <b>`labels`</b>: A binary `Tensor` whose shape matches `predictions`.
*  <b>`weights`</b>: An optional, floating point `Tensor` of same shape as
    `predictions`.
*  <b>`num_thresholds`</b>: The number of thresholds to use when discretizing the roc
    curve.
*  <b>`metrics_collections`</b>: An optional list of collections that `auc` should be
    added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`curve`</b>: Specifies the name of the curve to be computed, 'ROC' [default] or
  'PR' for the Precision-Recall-curve.

*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`auc`</b>: A scalar tensor representing the current area-under-curve.
*  <b>`update_op`</b>: An operation that increments the `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` variables
    appropriately and whose value matches `auc`.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` and `labels` do not match or if
    `weights` is not `None` and its shape doesn't match `predictions` or
    if either `metrics_collections` or `updates_collections` are not a list or
    tuple.

