### `tf.contrib.metrics.streaming_sensitivity_at_specificity(predictions, labels, specificity, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None)` {#streaming_sensitivity_at_specificity}

Computes the the specificity at a given sensitivity.

The `streaming_sensitivity_at_specificity` function creates four local
variables, `true_positives`, `true_negatives`, `false_positives` and
`false_negatives` that are used to compute the sensitivity at the given
specificity value. The threshold for the given specificity value is computed
and used to evaluate the corresponding sensitivity.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`sensitivity`. `update_op` increments the `true_positives`, `true_negatives`,
`false_positives` and `false_negatives` counts with the weight of each case
found in the `predictions` and `labels`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

For additional information about specificity and sensitivity, see the
following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

##### Args:


*  <b>`predictions`</b>: A floating point `Tensor` of arbitrary shape and whose values
    are in the range `[0, 1]`.
*  <b>`labels`</b>: A `bool` `Tensor` whose shape matches `predictions`.
*  <b>`specificity`</b>: A scalar value in range `[0, 1]`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`num_thresholds`</b>: The number of thresholds to use for matching the given
    specificity.
*  <b>`metrics_collections`</b>: An optional list of collections that `sensitivity`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`sensitivity`</b>: A scalar tensor representing the sensitivity at the given
    `specificity` value.
*  <b>`update_op`</b>: An operation that increments the `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` variables
    appropriately and whose value matches `sensitivity`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    `specificity` is not between 0 and 1, or if either `metrics_collections`
    or `updates_collections` are not a list or tuple.

