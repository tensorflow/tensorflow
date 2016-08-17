### `tf.contrib.metrics.streaming_specificity_at_sensitivity(predictions, labels, sensitivity, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None)` {#streaming_specificity_at_sensitivity}

Computes the the specificity at a given sensitivity.

The `streaming_specificity_at_sensitivity` function creates four local
variables, `true_positives`, `true_negatives`, `false_positives` and
`false_negatives` that are used to compute the specificity at the given
sensitivity value. The threshold for the given sensitivity value is computed
and used to evaluate the corresponding specificity.

To faciliate the estimation of the metric over a stream of data, the function
creates an `update_op` operation. `update_op` increments the
`true_positives`, `true_negatives`, `false_positives` and `false_negatives`
counts with the weighted number of each found in the current `predictions`
and `labels` `Tensors`. If `weights` is `None`, it is assumed that all
entries have weight 1. Note that a weight of 0 can be used to effectively
mask out and ignore specific entries. In addition to performing the updates,
`update_op` also returns the `specificity`.

For additional information about specificity and sensitivity, see the
following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

##### Args:


*  <b>`predictions`</b>: A floating point `Tensor` of arbitrary shape and whose values
    are in the range `[0, 1]`.
*  <b>`labels`</b>: A binary `Tensor` whose shape matches `predictions`.
*  <b>`sensitivity`</b>: A scalar value in range `[0, 1]`.
*  <b>`weights`</b>: An optional, floating point `Tensor` of same shape as
    `predictions`.
*  <b>`num_thresholds`</b>: The number of thresholds to use for matching the given
    sensitivity.
*  <b>`metrics_collections`</b>: An optional list of collections that `specificity`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`specificity`</b>: A scalar tensor representing the specificity at the given
    `specificity` value.
*  <b>`update_op`</b>: An operation that increments the `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` variables
    appropriately and whose value matches `specificity`.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` and `labels` do not match or if
    `weights` is not `None` and its shape doesn't match `predictions` or
    `sensitivity` is not between 0 and 1 or if either `metrics_collections` or
    `updates_collections` are not a list or tuple.

