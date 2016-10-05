<!-- This file is machine generated: DO NOT EDIT! -->

# Metrics (contrib)
[TOC]

##Ops for evaluation metrics and summary statistics.

### API

This module provides functions for computing streaming metrics: metrics computed
on dynamically valued `Tensors`. Each metric declaration returns a
"value_tensor", an idempotent operation that returns the current value of the
metric, and an "update_op", an operation that accumulates the information
from the current value of the `Tensors` being measured as well as returns the
value of the "value_tensor".

To use any of these metrics, one need only declare the metric, call `update_op`
repeatedly to accumulate data over the desired number of `Tensor` values (often
each one is a single batch) and finally evaluate the value_tensor. For example,
to use the `streaming_mean`:

```python
value = ...
mean_value, update_op = tf.contrib.metrics.streaming_mean(values)
sess.run(tf.initialize_local_variables())

for i in range(number_of_batches):
  print('Mean after batch %d: %f' % (i, update_op.eval())
print('Final Mean: %f' % mean_value.eval())
```

Each metric function adds nodes to the graph that hold the state necessary to
compute the value of the metric as well as a set of operations that actually
perform the computation. Every metric evaluation is composed of three steps

* Initialization: initializing the metric state.
* Aggregation: updating the values of the metric state.
* Finalization: computing the final metric value.

In the above example, calling streaming_mean creates a pair of state variables
that will contain (1) the running sum and (2) the count of the number of samples
in the sum.  Because the streaming metrics use local variables,
the Initialization stage is performed by running the op returned
by `tf.initialize_local_variables()`. It sets the sum and count variables to
zero.

Next, Aggregation is performed by examining the current state of `values`
and incrementing the state variables appropriately. This step is executed by
running the `update_op` returned by the metric.

Finally, finalization is performed by evaluating the "value_tensor"

In practice, we commonly want to evaluate across many batches and multiple
metrics. To do so, we need only run the metric computation operations multiple
times:

```python
labels = ...
predictions = ...
accuracy, update_op_acc = tf.contrib.metrics.streaming_accuracy(
    labels, predictions)
error, update_op_error = tf.contrib.metrics.streaming_mean_absolute_error(
    labels, predictions)

sess.run(tf.initialize_local_variables())
for batch in range(num_batches):
  sess.run([update_op_acc, update_op_error])

accuracy, mean_absolute_error = sess.run([accuracy, mean_absolute_error])
```

Note that when evaluating the same metric multiple times on different inputs,
one must specify the scope of each metric to avoid accumulating the results
together:

```python
labels = ...
predictions0 = ...
predictions1 = ...

accuracy0 = tf.contrib.metrics.accuracy(labels, predictions0, name='preds0')
accuracy1 = tf.contrib.metrics.accuracy(labels, predictions1, name='preds1')
```

Certain metrics, such as streaming_mean or streaming_accuracy, can be weighted
via a `weights` argument. The `weights` tensor must be the same size as the
labels and predictions tensors and results in a weighted average of the metric.

Other metrics, such as streaming_recall, streaming_precision, and streaming_auc,
are not well defined with regard to weighted samples. However, a binary
`ignore_mask` argument can be used to ignore certain values at graph executation
time.

## Metric `Ops`

- - -

### `tf.contrib.metrics.streaming_accuracy(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_accuracy}

Calculates how often `predictions` matches `labels`.

The `streaming_accuracy` function creates two local variables, `total` and
`count` that are used to compute the frequency with which `predictions`
matches `labels`. This frequency is ultimately returned as `accuracy`: an
idempotent operation that simply divides `total` by `count`.

For estimation of the metric  over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the `accuracy`.
Internally, an `is_correct` operation computes a `Tensor` with elements 1.0
where the corresponding elements of `predictions` and `labels` match and 0.0
otherwise. Then `update_op` increments `total` with the reduced sum of the
product of `weights` and `is_correct`, and it increments `count` with the
reduced sum of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: The predicted values, a `Tensor` of any shape.
*  <b>`labels`</b>: The ground truth values, a `Tensor` whose shape matches
    `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that `accuracy` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`accuracy`</b>: A tensor representing the accuracy, the value of `total` divided
    by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `accuracy`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

### `tf.contrib.metrics.streaming_mean(values, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean}

Computes the (weighted) mean of the given values.

The `streaming_mean` function creates two local variables, `total` and `count`
that are used to compute the average of `values`. This average is ultimately
returned as `mean` which is an idempotent operation that simply divides
`total` by `count`.

For estimation of the metric  over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the `mean`.
`update_op` increments `total` with the reduced sum of the product of `values`
and `weights`, and it increments `count` with the reduced sum of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`values`</b>: A `Tensor` of arbitrary dimensions.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `values`.
*  <b>`metrics_collections`</b>: An optional list of collections that `mean`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op`
    should be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`mean`</b>: A tensor representing the current mean, the value of `total` divided
    by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_value`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match `values`,
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.


- - -

### `tf.contrib.metrics.streaming_recall(*args, **kwargs)` {#streaming_recall}

Computes the recall of the predictions with respect to the labels. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

  The `streaming_recall` function creates two local variables, `true_positives`
  and `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives`  and `false_negatives`.

  For estimation of the metric  over a stream of data, the function creates an
  `update_op` that updates these variables and returns the `recall`. `update_op`
  weights each prediction by the corresponding value in `weights`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: The predicted values, a `bool` `Tensor` of arbitrary shape.
    labels: The ground truth values, a `bool` `Tensor` whose dimensions must
      match `predictions`.
    ignore_mask: An optional, `bool` `Tensor` whose shape matches `predictions`.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`.
    metrics_collections: An optional list of collections that `recall` should
      be added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    name: An optional variable_scope name.

  Returns:
    recall: Scalar float `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately and whose value matches
      `recall`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`, or
      if `weights` is not `None` and its shape doesn't match `predictions`, or
      if either `metrics_collections` or `updates_collections` are not a list or
      tuple.


- - -

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


- - -

### `tf.contrib.metrics.streaming_auc(predictions, labels, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None)` {#streaming_auc}

Computes the approximate AUC via a Riemann sum.

The `streaming_auc` function creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the AUC. To discretize the AUC curve, a linearly spaced set of
thresholds is used to compute pairs of recall and precision values. The area
under the ROC-curve is therefore computed using the height of the recall
values by the false positive rate, while the area under the PR-curve is the
computed using the height of the precision values by the recall.

This value is ultimately returned as `auc`, an idempotent operation that
computes the area under a discretized curve of precision versus recall values
(computed using the aforementioned variables). The `num_thresholds` variable
controls the degree of discretization with larger numbers of thresholds more
closely approximating the true AUC.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the `auc`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A floating point `Tensor` of arbitrary shape and whose values
    are in the range `[0, 1]`.
*  <b>`labels`</b>: A `bool` `Tensor` whose shape matches `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
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


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

### `tf.contrib.metrics.streaming_recall_at_k(*args, **kwargs)` {#streaming_recall_at_k}

Computes the recall@k of the predictions with respect to dense labels. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

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
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: A floating point tensor of dimension [batch_size, num_classes]
    labels: A tensor of dimension [batch_size] whose type is in `int32`,
      `int64`.
    k: The number of top elements to look at for computing recall.
    ignore_mask: An optional, `bool` `Tensor` whose shape matches `predictions`.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`.
    metrics_collections: An optional list of collections that `recall_at_k`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    recall_at_k: A tensor representing the recall@k, the fraction of labels
      which fall into the top `k` predictions.
    update_op: An operation that increments the `total` and `count` variables
      appropriately and whose value matches `recall_at_k`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`, or
      if `weights` is not `None` and its shape doesn't match `predictions`, or
      if either `metrics_collections` or `updates_collections` are not a list or
      tuple.


- - -

### `tf.contrib.metrics.streaming_mean_absolute_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_absolute_error}

Computes the mean absolute error between the labels and predictions.

The `streaming_mean_absolute_error` function creates two local variables,
`total` and `count` that are used to compute the mean absolute error. This
average is weighted by `weights`, and it is ultimately returned as
`mean_absolute_error`: an idempotent operation that simply divides `total` by
`count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`mean_absolute_error`. Internally, an `absolute_errors` operation computes the
absolute value of the differences between `predictions` and `labels`. Then
`update_op` increments `total` with the reduced sum of the product of
`weights` and `absolute_errors`, and it increments `count` with the reduced
sum of `weights`

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_absolute_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`mean_absolute_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_absolute_error`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

### `tf.contrib.metrics.streaming_mean_iou(*args, **kwargs)` {#streaming_mean_iou}

Calculate per-step mean Intersection-Over-Union (mIOU). (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

  Mean Intersection-Over-Union is a common evaluation metric for
  semantic image segmentation, which first computes the IOU for each
  semantic class and then computes the average over classes.
  IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
  The predictions are accumulated in a confusion matrix, weighted by `weights`,
  and mIOU is then calculated from it.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean_iou`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: A tensor of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened, if its rank > 1.
    labels: A tensor of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened, if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    ignore_mask: An optional, `bool` `Tensor` whose shape matches `predictions`.
    weights: An optional `Tensor` whose shape is broadcastable to `predictions`.
    metrics_collections: An optional list of collections that `mean_iou`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    mean_iou: A tensor representing the mean intersection-over-union.
    update_op: An operation that increments the confusion matrix.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `ignore_mask` is not `None` and its shape doesn't match `predictions`, or
      if `weights` is not `None` and its shape doesn't match `predictions`, or
      if either `metrics_collections` or `updates_collections` are not a list or
      tuple.


- - -

### `tf.contrib.metrics.streaming_mean_relative_error(predictions, labels, normalizer, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_relative_error}

Computes the mean relative error by normalizing with the given values.

The `streaming_mean_relative_error` function creates two local variables,
`total` and `count` that are used to compute the mean relative absolute error.
This average is weighted by `weights`, and it is ultimately returned as
`mean_relative_error`: an idempotent operation that simply divides `total` by
`count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`mean_reative_error`. Internally, a `relative_errors` operation divides the
absolute value of the differences between `predictions` and `labels` by the
`normalizer`. Then `update_op` increments `total` with the reduced sum of the
product of `weights` and `relative_errors`, and it increments `count` with the
reduced sum of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`normalizer`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_relative_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`mean_relative_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_relative_error`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

### `tf.contrib.metrics.streaming_mean_squared_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_squared_error}

Computes the mean squared error between the labels and predictions.

The `streaming_mean_squared_error` function creates two local variables,
`total` and `count` that are used to compute the mean squared error.
This average is weighted by `weights`, and it is ultimately returned as
`mean_squared_error`: an idempotent operation that simply divides `total` by
`count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`mean_squared_error`. Internally, a `squared_error` operation computes the
element-wise square of the difference between `predictions` and `labels`. Then
`update_op` increments `total` with the reduced sum of the product of
`weights` and `squared_error`, and it increments `count` with the reduced sum
of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_squared_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`mean_squared_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_squared_error`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

### `tf.contrib.metrics.streaming_root_mean_squared_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_root_mean_squared_error}

Computes the root mean squared error between the labels and predictions.

The `streaming_root_mean_squared_error` function creates two local variables,
`total` and `count` that are used to compute the root mean squared error.
This average is weighted by `weights`, and it is ultimately returned as
`root_mean_squared_error`: an idempotent operation that takes the square root
of the division of `total` by `count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`root_mean_squared_error`. Internally, a `squared_error` operation computes
the element-wise square of the difference between `predictions` and `labels`.
Then `update_op` increments `total` with the reduced sum of the product of
`weights` and `squared_error`, and it increments `count` with the reduced sum
of `weights`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `root_mean_squared_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`root_mean_squared_error`</b>: A tensor representing the current mean, the value
    of `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `root_mean_squared_error`.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

### `tf.contrib.metrics.streaming_covariance(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_covariance}

Computes the unbiased sample covariance between `predictions` and `labels`.

The `streaming_covariance` function creates four local variables,
`comoment`, `mean_prediction`, `mean_label`, and `count`, which are used to
compute the sample covariance between predictions and labels across multiple
batches of data. The covariance is ultimately returned as an idempotent
operation that simply divides `comoment` by `count` - 1. We use `count` - 1
in order to get an unbiased estimate.

The algorithm used for this online computation is described in
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
Specifically, the formula used to combine two sample comoments is
`C_AB = C_A + C_B + (E[x_A] - E[x_B]) * (E[y_A] - E[y_B]) * n_A * n_B / n_AB`
The comoment for a single batch of data is simply
`sum((x - E[x]) * (y - E[y]))`, optionally weighted.

If `weights` is not None, then it is used to compute weighted comoments,
means, and count. NOTE: these weights are treated as "frequency weights", as
opposed to "reliability weights". See discussion of the difference on
https://wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

To facilitate the computation of covariance across multiple batches of data,
the function creates an `update_op` operation, which updates underlying
variables and returns the updated covariance.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary size.
*  <b>`labels`</b>: A `Tensor` of the same size as `predictions`.
*  <b>`weights`</b>: An optional set of weights which indicates the frequency with which
    an example is sampled. Must be broadcastable with `labels`.
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`covariance`</b>: A `Tensor` representing the current unbiased sample covariance,
    `comoment` / (`count` - 1).
*  <b>`update_op`</b>: An operation that updates the local variables appropriately.

##### Raises:


*  <b>`ValueError`</b>: If labels and predictions are of different sizes or if either
    `metrics_collections` or `updates_collections` are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_pearson_correlation(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_pearson_correlation}

Computes Pearson correlation coefficient between `predictions`, `labels`.

The `streaming_pearson_correlation` function delegates to
`streaming_covariance` the tracking of three [co]variances:

- `streaming_covariance(predictions, labels)`, i.e. covariance
- `streaming_covariance(predictions, predictions)`, i.e. variance
- `streaming_covariance(labels, labels)`, i.e. variance

The product-moment correlation ultimately returned is an idempotent operation
`cov(predictions, labels) / sqrt(var(predictions) * var(labels))`. To
facilitate correlation computation across multiple batches, the function
groups the `update_op`s of the underlying streaming_covariance and returns an
`update_op`.

If `weights` is not None, then it is used to compute a weighted correlation.
NOTE: these weights are treated as "frequency weights", as opposed to
"reliability weights". See discussion of the difference on
https://wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary size.
*  <b>`labels`</b>: A `Tensor` of the same size as predictions.
*  <b>`weights`</b>: An optional set of weights which indicates the frequency with which
    an example is sampled. Must be broadcastable with `labels`.
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`pearson_r`</b>: A tensor representing the current Pearson product-moment
    correlation coefficient, the value of
    `cov(predictions, labels) / sqrt(var(predictions) * var(labels))`.
*  <b>`update_op`</b>: An operation that updates the underlying variables appropriately.

##### Raises:


*  <b>`ValueError`</b>: If `labels` and `predictions` are of different sizes, or if
    `weights` is the wrong size, or if either `metrics_collections` or
    `updates_collections` are not a `list` or `tuple`.


- - -

### `tf.contrib.metrics.streaming_mean_cosine_distance(predictions, labels, dim, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_cosine_distance}

Computes the cosine distance between the labels and predictions.

The `streaming_mean_cosine_distance` function creates two local variables,
`total` and `count` that are used to compute the average cosine distance
between `predictions` and `labels`. This average is weighted by `weights`,
and it is ultimately returned as `mean_distance`, which is an idempotent
operation that simply divides `total` by `count`.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`mean_distance`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of the same shape as `labels`.
*  <b>`labels`</b>: A `Tensor` of arbitrary shape.
*  <b>`dim`</b>: The dimension along which the cosine distance is computed.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`,
    and whose dimension `dim` is 1.
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`mean_distance`</b>: A tensor representing the current mean, the value of `total`
    divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately.

##### Raises:


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, or if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

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


- - -

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


- - -

### `tf.contrib.metrics.streaming_sparse_average_precision_at_k(predictions, labels, k, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_sparse_average_precision_at_k}

Computes average precision@k of predictions with respect to sparse labels.

See `sparse_average_precision_at_k` for details on formula. `weights` are
applied to the result of `sparse_average_precision_at_k`

`streaming_sparse_average_precision_at_k` creates two local variables,
`average_precision_at_<k>/count` and `average_precision_at_<k>/total`, that
are used to compute the frequency. This frequency is ultimately returned as
`precision_at_<k>`: an idempotent operation that simply divides
`true_positive_at_<k>` by total (`true_positive_at_<k>` +
`false_positive_at_<k>`).

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
indicating the top `k` `predictions`. Set operations applied to `top_k` and
`labels` calculate the true positives and false positives weighted by
`weights`. Then `update_op` increments `true_positive_at_<k>` and
`false_positive_at_<k>` using these values.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

##### Args:


*  <b>`predictions`</b>: Float `Tensor` with shape [D1, ... DN, num_classes] where
    N >= 1. Commonly, N=1 and `predictions` has shape
    [batch size, num_classes]. The final dimension contains the logit values
    for each class. [D1, ... DN] must match `labels`.
*  <b>`labels`</b>: `int64` `Tensor` or `SparseTensor` with shape
    [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
    target classes for the associated prediction. Commonly, N=1 and `labels`
    has shape [batch_size, num_labels]. [D1, ... DN] must match
    `predictions_idx`. Values should be in range [0, num_classes], where
    num_classes is the last dimension of `predictions`.
*  <b>`k`</b>: Integer, k for @k metric. This will calculate an average precision for
    range `[1,k]`, as documented above.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to the the first
    [D1, ... DN] dimensions of `predictions` and `labels`.
*  <b>`metrics_collections`</b>: An optional list of collections that values should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that updates should
    be added to.
*  <b>`name`</b>: Name of new update operation, and namespace for other dependent ops.

##### Returns:


*  <b>`mean_average_precision`</b>: Scalar `float64` `Tensor` with the mean average
    precision values.
*  <b>`update`</b>: `Operation` that increments  variables appropriately, and whose
    value matches `metric`.


- - -

### `tf.contrib.metrics.streaming_sparse_precision_at_k(*args, **kwargs)` {#streaming_sparse_precision_at_k}

Computes precision@k of the predictions with respect to sparse labels. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

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
  `precision_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` +
  `false_positive_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `precision_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false positives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_positive_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match
      `predictions_idx`. Values should be in range [0, num_classes], where
      num_classes is the last dimension of `predictions`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes], where num_classes is the last dimension of
      `predictions`.
    ignore_mask: An optional, `bool` `Tensor` whose shape is broadcastable to
      the the first [D1, ... DN] dimensions of `predictions` and `labels`.
    weights: An optional `Tensor` whose shape is broadcastable to the the first
      [D1, ... DN] dimensions of `predictions` and `labels`.
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    precision: Scalar `float64` `Tensor` with the value of `true_positives`
      divided by the sum of `true_positives` and `false_positives`.
    update_op: `Operation` that increments `true_positives` and
      `false_positives` variables appropriately, and whose value matches
      `precision`.

  Raises:
    ValueError: If `ignore_mask` is not `None` and its shape doesn't match
      `predictions`, or if `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_sparse_recall_at_k(*args, **kwargs)` {#streaming_sparse_recall_at_k}

Computes recall@k of the predictions with respect to sparse labels. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-10-19.
Instructions for updating:
`ignore_mask` is being deprecated. Instead use `weights` with values 0.0 and 1.0 to mask values. For example, `weights=tf.logical_not(mask)`.

  If `class_id` is specified, we calculate recall by considering only the
      entries in the batch for which `class_id` is in the label, and computing
      the fraction of them for which `class_id` is in the top-k `predictions`.
  If `class_id` is not specified, we'll calculate recall as how often on
      average a class among the labels of a batch entry is in the top-k
      `predictions`.

  `streaming_sparse_recall_at_k` creates two local variables,
  `true_positive_at_<k>` and `false_negative_at_<k>`, that are used to compute
  the recall_at_k frequency. This frequency is ultimately returned as
  `recall_at_<k>`: an idempotent operation that simply divides
  `true_positive_at_<k>` by total (`true_positive_at_<k>` + `recall_at_<k>`).

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the
  `recall_at_<k>`. Internally, a `top_k` operation computes a `Tensor`
  indicating the top `k` `predictions`. Set operations applied to `top_k` and
  `labels` calculate the true positives and false negatives weighted by
  `weights`. Then `update_op` increments `true_positive_at_<k>` and
  `false_negative_at_<k>` using these values.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Alternatively, if `ignore_mask` is not `None`, then mask values where
  `ignore_mask` is `True`.

  Args:
    predictions: Float `Tensor` with shape [D1, ... DN, num_classes] where
      N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
      The final dimension contains the logit values for each class. [D1, ... DN]
      must match `labels`.
    labels: `int64` `Tensor` or `SparseTensor` with shape
      [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
      target classes for the associated prediction. Commonly, N=1 and `labels`
      has shape [batch_size, num_labels]. [D1, ... DN] must match `labels`.
      Values should be in range [0, num_classes], where num_classes is the last
      dimension of `predictions`.
    k: Integer, k for @k metric.
    class_id: Integer class ID for which we want binary metrics. This should be
      in range [0, num_classes], where num_classes is the last dimension of
      `predictions`.
    ignore_mask: An optional, `bool` `Tensor` whose shape is broadcastable to
      the the first [D1, ... DN] dimensions of `predictions` and `labels`.
    weights: An optional `Tensor` whose shape is broadcastable to the the first
      [D1, ... DN] dimensions of `predictions` and `labels`.
    metrics_collections: An optional list of collections that values should
      be added to.
    updates_collections: An optional list of collections that updates should
      be added to.
    name: Name of new update operation, and namespace for other dependent ops.

  Returns:
    recall: Scalar `float64` `Tensor` with the value of `true_positives` divided
      by the sum of `true_positives` and `false_negatives`.
    update_op: `Operation` that increments `true_positives` and
      `false_negatives` variables appropriately, and whose value matches
      `recall`.

  Raises:
    ValueError: If `ignore_mask` is not `None` and its shape doesn't match
      `predictions`, or if `weights` is not `None` and its shape doesn't match
      `predictions`, or if either `metrics_collections` or `updates_collections`
      are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_specificity_at_sensitivity(predictions, labels, sensitivity, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None)` {#streaming_specificity_at_sensitivity}

Computes the the specificity at a given sensitivity.

The `streaming_specificity_at_sensitivity` function creates four local
variables, `true_positives`, `true_negatives`, `false_positives` and
`false_negatives` that are used to compute the specificity at the given
sensitivity value. The threshold for the given sensitivity value is computed
and used to evaluate the corresponding specificity.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that updates these variables and returns the
`specificity`. `update_op` increments the `true_positives`, `true_negatives`,
`false_positives` and `false_negatives` counts with the weight of each case
found in the `predictions` and `labels`.

If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

For additional information about specificity and sensitivity, see the
following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

##### Args:


*  <b>`predictions`</b>: A floating point `Tensor` of arbitrary shape and whose values
    are in the range `[0, 1]`.
*  <b>`labels`</b>: A `bool` `Tensor` whose shape matches `predictions`.
*  <b>`sensitivity`</b>: A scalar value in range `[0, 1]`.
*  <b>`weights`</b>: An optional `Tensor` whose shape is broadcastable to `predictions`.
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


*  <b>`ValueError`</b>: If `predictions` and `labels` have mismatched shapes, if
    `weights` is not `None` and its shape doesn't match `predictions`, or if
    `sensitivity` is not between 0 and 1, or if either `metrics_collections`
    or `updates_collections` are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_concat(values, axis=0, max_size=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_concat}

Concatenate values along an axis across batches.

The function `streaming_concat` creates two local variables, `array` and
`size`, that are used to store concatenated values. Internally, `array` is
used as storage for a dynamic array (if `maxsize` is `None`), which ensures
that updates can be run in amortized constant time.

For estimation of the metric over a stream of data, the function creates an
`update_op` operation that appends the values of a tensor and returns the
`value` of the concatenated tensors.

This op allows for evaluating metrics that cannot be updated incrementally
using the same framework as other streaming metrics.

##### Args:


*  <b>`values`</b>: tensor to concatenate. Rank and the shape along all axes other than
    the axis to concatenate along must be statically known.
*  <b>`axis`</b>: optional integer axis to concatenate along.
*  <b>`max_size`</b>: optional integer maximum size of `value` along the given axis.
    Once the maximum size is reached, further updates are no-ops. By default,
    there is no maximum size: the array is resized as necessary.
*  <b>`metrics_collections`</b>: An optional list of collections that `value`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections `update_op` should be
    added to.
*  <b>`name`</b>: An optional variable_scope name.

##### Returns:


*  <b>`value`</b>: A tensor representing the concatenated values.
*  <b>`update_op`</b>: An operation that concatenates the next values.

##### Raises:


*  <b>`ValueError`</b>: if `values` does not have a statically known rank, `axis` is
    not in the valid range or the size of `values` is not statically known
    along any axis other than `axis`.



- - -

### `tf.contrib.metrics.auc_using_histogram(boolean_labels, scores, score_range, nbins=100, collections=None, check_shape=True, name=None)` {#auc_using_histogram}

AUC computed by maintaining histograms.

Rather than computing AUC directly, this Op maintains Variables containing
histograms of the scores associated with `True` and `False` labels.  By
comparing these the AUC is generated, with some discretization error.
See: "Efficient AUC Learning Curve Calculation" by Bouckaert.

This AUC Op updates in `O(batch_size + nbins)` time and works well even with
large class imbalance.  The accuracy is limited by discretization error due
to finite number of bins.  If scores are concentrated in a fewer bins,
accuracy is lower.  If this is a concern, we recommend trying different
numbers of bins and comparing results.

##### Args:


*  <b>`boolean_labels`</b>: 1-D boolean `Tensor`.  Entry is `True` if the corresponding
    record is in class.
*  <b>`scores`</b>: 1-D numeric `Tensor`, same shape as boolean_labels.
*  <b>`score_range`</b>: `Tensor` of shape `[2]`, same dtype as `scores`.  The min/max
    values of score that we expect.  Scores outside range will be clipped.
*  <b>`nbins`</b>: Integer number of bins to use.  Accuracy strictly increases as the
    number of bins increases.
*  <b>`collections`</b>: List of graph collections keys. Internal histogram Variables
    are added to these collections. Defaults to `[GraphKeys.LOCAL_VARIABLES]`.
*  <b>`check_shape`</b>: Boolean.  If `True`, do a runtime shape check on the scores
    and labels.
*  <b>`name`</b>: A name for this Op.  Defaults to "auc_using_histogram".

##### Returns:


*  <b>`auc`</b>: `float32` scalar `Tensor`.  Fetching this converts internal histograms
    to auc value.
*  <b>`update_op`</b>: `Op`, when run, updates internal histograms.



- - -

### `tf.contrib.metrics.accuracy(predictions, labels, weights=None)` {#accuracy}

Computes the percentage of times that predictions matches labels.

##### Args:


*  <b>`predictions`</b>: the predicted values, a `Tensor` whose dtype and shape
               matches 'labels'.
*  <b>`labels`</b>: the ground truth values, a `Tensor` of any shape and
          bool, integer, or string dtype.
*  <b>`weights`</b>: None or `Tensor` of float values to reweight the accuracy.

##### Returns:

  Accuracy `Tensor`.

##### Raises:


*  <b>`ValueError`</b>: if dtypes don't match or
              if dtype is not bool, integer, or string.


- - -

### `tf.contrib.metrics.confusion_matrix(predictions, labels, num_classes=None, dtype=tf.int32, name=None, weights=None)` {#confusion_matrix}

Computes the confusion matrix from predictions and labels.

Calculate the Confusion Matrix for a pair of prediction and
label 1-D int arrays.

Considering a prediction array such as: `[1, 2, 3]`
And a label array such as: `[2, 2, 3]`

The confusion matrix returned would be the following one:

```python
    [[0, 0, 0]
     [0, 1, 0]
     [0, 1, 0]
     [0, 0, 1]]
```

If `weights` is not None, then the confusion matrix elements are the
corresponding `weights` elements.

Where the matrix rows represent the prediction labels and the columns
represents the real labels. The confusion matrix is always a 2-D array
of shape [n, n], where n is the number of valid labels for a given
classification task. Both prediction and labels must be 1-D arrays of
the same shape in order for this function to work.

##### Args:


*  <b>`predictions`</b>: A 1-D array representing the predictions for a given
               classification.
*  <b>`labels`</b>: A 1-D representing the real labels for the classification task.
*  <b>`num_classes`</b>: The possible number of labels the classification task can
               have. If this value is not provided, it will be calculated
               using both predictions and labels array.
*  <b>`dtype`</b>: Data type of the confusion matrix.
*  <b>`name`</b>: Scope name.
*  <b>`weights`</b>: An optional `Tensor` whose shape matches `predictions`.

##### Returns:

  A k X k matrix representing the confusion matrix, where k is the number of
  possible labels in the classification task.

##### Raises:


*  <b>`ValueError`</b>: If both predictions and labels are not 1-D vectors and have
    mismatched shapes, or if `weights` is not `None` and its shape doesn't
    match `predictions`.



- - -

### `tf.contrib.metrics.aggregate_metrics(*value_update_tuples)` {#aggregate_metrics}

Aggregates the metric value tensors and update ops into two lists.

##### Args:


*  <b>`*value_update_tuples`</b>: a variable number of tuples, each of which contain the
    pair of (value_tensor, update_op) from a streaming metric.

##### Returns:

  a list of value tensors and a list of update ops.

##### Raises:


*  <b>`ValueError`</b>: if `value_update_tuples` is empty.


- - -

### `tf.contrib.metrics.aggregate_metric_map(names_to_tuples)` {#aggregate_metric_map}

Aggregates the metric names to tuple dictionary.

This function is useful for pairing metric names with their associated value
and update ops when the list of metrics is long. For example:

  metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
      'Mean Absolute Error': new_slim.metrics.streaming_mean_absolute_error(
          predictions, labels, weights),
      'Mean Relative Error': new_slim.metrics.streaming_mean_relative_error(
          predictions, labels, labels, weights),
      'RMSE Linear': new_slim.metrics.streaming_root_mean_squared_error(
          predictions, labels, weights),
      'RMSE Log': new_slim.metrics.streaming_root_mean_squared_error(
          predictions, labels, weights),
  })

##### Args:


*  <b>`names_to_tuples`</b>: a map of metric names to tuples, each of which contain the
    pair of (value_tensor, update_op) from a streaming metric.

##### Returns:

  A dictionary from metric names to value ops and a dictionary from metric
  names to update ops.



## Set `Ops`

- - -

### `tf.contrib.metrics.set_difference(a, b, aminusb=True, validate_indices=True)` {#set_difference}

Compute set difference of elements in last dimension of `a` and `b`.

All but the last dimension of `a` and `b` must match.

##### Args:


*  <b>`a`</b>: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
*  <b>`b`</b>: `Tensor` or `SparseTensor` of the same type as `a`. Must be
      `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
      sorted in row-major order.
*  <b>`aminusb`</b>: Whether to subtract `b` from `a`, vs vice versa.
*  <b>`validate_indices`</b>: Whether to validate the order and range of sparse indices
     in `a` and `b`.

##### Returns:

  A `SparseTensor` with the same rank as `a` and `b`, and all but the last
  dimension the same. Elements along the last dimension contain the
  differences.


- - -

### `tf.contrib.metrics.set_intersection(a, b, validate_indices=True)` {#set_intersection}

Compute set intersection of elements in last dimension of `a` and `b`.

All but the last dimension of `a` and `b` must match.

##### Args:


*  <b>`a`</b>: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
*  <b>`b`</b>: `Tensor` or `SparseTensor` of the same type as `a`. Must be
      `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
      sorted in row-major order.
*  <b>`validate_indices`</b>: Whether to validate the order and range of sparse indices
     in `a` and `b`.

##### Returns:

  A `SparseTensor` with the same rank as `a` and `b`, and all but the last
  dimension the same. Elements along the last dimension contain the
  intersections.


- - -

### `tf.contrib.metrics.set_size(a, validate_indices=True)` {#set_size}

Compute number of unique elements along last dimension of `a`.

##### Args:


*  <b>`a`</b>: `SparseTensor`, with indices sorted in row-major order.
*  <b>`validate_indices`</b>: Whether to validate the order and range of sparse indices
     in `a`.

##### Returns:

  `int32` `Tensor` of set sizes. For `a` ranked `n`, this is a `Tensor` with
  rank `n-1`, and the same 1st `n-1` dimensions as `a`. Each value is the
  number of unique elements in the corresponding `[0...n-1]` dimension of `a`.

##### Raises:


*  <b>`TypeError`</b>: If `a` is an invalid types.


- - -

### `tf.contrib.metrics.set_union(a, b, validate_indices=True)` {#set_union}

Compute set union of elements in last dimension of `a` and `b`.

All but the last dimension of `a` and `b` must match.

##### Args:


*  <b>`a`</b>: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
      must be sorted in row-major order.
*  <b>`b`</b>: `Tensor` or `SparseTensor` of the same type as `a`. Must be
      `SparseTensor` if `a` is `SparseTensor`. If sparse, indices must be
      sorted in row-major order.
*  <b>`validate_indices`</b>: Whether to validate the order and range of sparse indices
     in `a` and `b`.

##### Returns:

  A `SparseTensor` with the same rank as `a` and `b`, and all but the last
  dimension the same. Elements along the last dimension contain the
  unions.


