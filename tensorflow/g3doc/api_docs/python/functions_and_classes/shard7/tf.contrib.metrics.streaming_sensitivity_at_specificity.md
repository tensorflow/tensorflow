### `tf.contrib.metrics.streaming_sensitivity_at_specificity(predictions, labels, specificity, ignore_mask=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None)` {#streaming_sensitivity_at_specificity}

Computes the the specificity at a given sensitivity.

The `streaming_sensitivity_at_specificity` function creates four local
variables, `true_positives`, `true_negatives`, `false_positives` and
`false_negatives` that are used to compute the sensitivity at the given
specificity value. The threshold for the given specificity value is computed
and used to evaluate the corresponding sensitivity.

To faciliate the estimation of the metric over a stream of data, the
function creates an `update_op` operation whose behavior is dependent on the
value of `ignore_mask`. If `ignore_mask` is None, then `update_op`
increments the `true_positives`, `true_negatives`, `false_positives` and
`false_negatives` counts with the number of each found in the current
`predictions` and `labels` `Tensors`. If `ignore_mask` is not `None`, then
the increment is performed using only the elements of `predictions` and
`labels` whose corresponding value in `ignore_mask` is `False`. In addition
to performing the updates, `update_op` also returns the `sensitivity`.

For additional information about specificity and sensitivity, see the
following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

##### Args:


*  <b>`predictions`</b>: A floating point `Tensor` of arbitrary shape and whose values
    are in the range `[0, 1]`.
*  <b>`labels`</b>: A binary `Tensor` whose shape matches `predictions`.
*  <b>`specificity`</b>: A scalar value in range `[0, 1]`.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `predictions`.
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


*  <b>`ValueError`</b>: If the shape of `predictions` and `labels` do not match or if
    `ignore_mask` is not `None` and its shape doesn't match `predictions` or
    `specificity` is not between 0 and 1 or if either `metrics_collections` or
    `updates_collections` are not a list or tuple.

