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
To facilitate the estimation of the accuracy over a stream of data, the
function utilizes two operations. First, an `is_correct` operation that
computes a tensor whose shape matches `predictions` and whose elements are
set to 1.0 when the corresponding values of `predictions` and `labels match
and 0.0 otherwise. Second, an `update_op` operation whose behavior is
dependent on the value of `weights`. If `weights` is None, then `update_op`
increments `total` with the number of elements of `predictions` that match
`labels` and increments `count` with the number of elements in `values`. If
`weights` is not `None`, then `update_op` increments `total` with the reduced
sum of the product of `weights` and `is_correct` and increments `count` with
the reduced sum of `weights`. In addition to performing the updates,
`update_op` also returns the `accuracy` value.

##### Args:


*  <b>`predictions`</b>: The predicted values, a `Tensor` of any shape.
*  <b>`labels`</b>: The ground truth values, a `Tensor` whose shape matches
    `predictions`.
*  <b>`weights`</b>: An optional set of weights whose shape matches `predictions`
    which, when not `None`, produces a weighted mean accuracy.
*  <b>`metrics_collections`</b>: An optional list of collections that `accuracy` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`accuracy`</b>: A tensor representing the accuracy, the value of `total` divided
    by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `accuracy`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `weight` is not `None` and its shape doesn't match `predictions` or
    if either `metrics_collections` or `updates_collections` are not
    a list or tuple.


- - -

### `tf.contrib.metrics.streaming_mean(values, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean}

Computes the (weighted) mean of the given values.

The `streaming_mean` function creates two local variables, `total` and `count`
that are used to compute the average of `values`. This average is ultimately
returned as `mean` which is an idempotent operation that simply divides
`total` by `count`. To facilitate the estimation of a mean over a stream
of data, the function creates an `update_op` operation whose behavior is
dependent on the value of `weights`. If `weights` is None, then `update_op`
increments `total` with the reduced sum of `values` and increments `count`
with the number of elements in `values`. If `weights` is not `None`, then
`update_op` increments `total` with the reduced sum of the product of `values`
and `weights` and increments `count` with the reduced sum of weights.
In addition to performing the updates, `update_op` also returns the
`mean`.

##### Args:


*  <b>`values`</b>: A `Tensor` of arbitrary dimensions.
*  <b>`weights`</b>: An optional set of weights of the same shape as `values`. If
    `weights` is not None, the function computes a weighted mean.
*  <b>`metrics_collections`</b>: An optional list of collections that `mean`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op`
    should be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean`</b>: A tensor representing the current mean, the value of `total` divided
    by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_value`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match `values`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.


- - -

### `tf.contrib.metrics.streaming_recall(predictions, labels, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_recall}

Computes the recall of the predictions with respect to the labels.

The `streaming_recall` function creates two local variables,
`true_positives` and `false_negatives`, that are used to compute the
recall. This value is ultimately returned as `recall`, an idempotent
operation that simply divides `true_positives` by the sum of `true_positives`
and `false_negatives`. To facilitate the calculation of the recall over a
stream of data, the function creates an `update_op` operation whose behavior
is dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
`update_op` increments `true_positives` with the number of elements of
`predictions` and `labels` that are both `True` and increments
`false_negatives` with the number of elements of `predictions` that are
`False` whose corresponding `labels` element is `False`. If `ignore_mask` is
not `None`, then the increments for `true_positives` and `false_negatives` are
only computed using elements of `predictions` and `labels` whose corresponding
values in `ignore_mask` are `False`. In addition to performing the updates,
`update_op` also returns the value of `recall`.

##### Args:


*  <b>`predictions`</b>: The predicted values, a binary `Tensor` of arbitrary shape.
*  <b>`labels`</b>: The ground truth values, a binary `Tensor` whose dimensions must
    match `predictions`.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that `precision` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`recall`</b>: Scalar float `Tensor` with the value of `true_positives` divided
    by the sum of `true_positives` and `false_negatives`.
*  <b>`update_op`</b>: `Operation` that increments `true_positives` and
    `false_negatives` variables appropriately and whose value matches
    `recall`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `ignore_mask` is not `None` and its shape doesn't match `predictions`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.


- - -

### `tf.contrib.metrics.streaming_precision(predictions, labels, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_precision}

Computes the precision of the predictions with respect to the labels.

The `streaming_precision` function creates two local variables,
`true_positives` and `false_positives`, that are used to compute the
precision. This value is ultimately returned as `precision`, an idempotent
operation that simply divides `true_positives` by the sum of `true_positives`
and `false_positives`. To facilitate the calculation of the precision over a
stream of data, the function creates an `update_op` operation whose behavior
is dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
`update_op` increments `true_positives` with the number of elements of
`predictions` and `labels` that are both `True` and increments
`false_positives` with the number of elements of `predictions` that are `True`
whose corresponding `labels` element is `False`. If `ignore_mask` is not
`None`, then the increments for `true_positives` and `false_positives` are
only computed using elements of `predictions` and `labels` whose corresponding
values in `ignore_mask` are `False`. In addition to performing the updates,
`update_op` also returns the value of `precision`.

##### Args:


*  <b>`predictions`</b>: The predicted values, a binary `Tensor` of arbitrary shape.
*  <b>`labels`</b>: The ground truth values, a binary `Tensor` whose dimensions must
    match `predictions`.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `predictions`.
*  <b>`metrics_collections`</b>: An optional list of collections that `precision` should
    be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`precision`</b>: Scalar float `Tensor` with the value of `true_positives`
    divided by the sum of `true_positives` and `false_positives`.
*  <b>`update_op`</b>: `Operation` that increments `true_positives` and
    `false_positives` variables appropriately and whose value matches
    `precision`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `ignore_mask` is not `None` and its shape doesn't match `predictions`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.


- - -

### `tf.contrib.metrics.streaming_auc(predictions, labels, ignore_mask=None, num_thresholds=200, metrics_collections=None, updates_collections=None, name=None)` {#streaming_auc}

Computes the approximate AUC via a Riemann sum.

The `streaming_auc` function creates four local variables, `true_positives`,
`true_negatives`, `false_positives` and `false_negatives` that are used to
compute the AUC. To discretize the AUC curve, a linearly spaced set of
thresholds is used to compute pairs of recall and precision values. The area
under the curve is therefore computed using the height of the recall values
by the false positive rate.

This value is ultimately returned as `auc`, an idempotent
operation the computes the area under a discretized curve of precision versus
recall values (computed using the afformentioned variables). The
`num_thresholds` variable controls the degree of discretization with larger
numbers of thresholds more closely approximating the true AUC.

To faciliate the estimation of the AUC over a stream of data, the function
creates an `update_op` operation whose behavior is dependent on the value of
`ignore_mask`. If `ignore_mask` is None, then `update_op` increments the
`true_positives`, `true_negatives`, `false_positives` and `false_negatives`
counts with the number of each found in the current `predictions` and `labels`
`Tensors`. If `ignore_mask` is not `None`, then the increment is performed
using only the elements of `predictions` and `labels` whose corresponding
value in `ignore_mask` is `False`. In addition to performing the updates,
`update_op` also returns the `auc`.

##### Args:


*  <b>`predictions`</b>: A floating point `Tensor` of arbitrary shape and whose values
    are in the range `[0, 1]`.
*  <b>`labels`</b>: A binary `Tensor` whose shape matches `predictions`.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `predictions`.
*  <b>`num_thresholds`</b>: The number of thresholds to use when discretizing the roc
    curve.
*  <b>`metrics_collections`</b>: An optional list of collections that `auc` should be
    added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`auc`</b>: A scalar tensor representing the current area-under-curve.
*  <b>`update_op`</b>: An operation that increments the `true_positives`,
    `true_negatives`, `false_positives` and `false_negatives` variables
    appropriately and whose value matches `auc`.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `predictions` and `labels` do not match or if
    `ignore_mask` is not `None` and its shape doesn't match `predictions` or
    if either `metrics_collections` or `updates_collections` are not a list or
    tuple.


- - -

### `tf.contrib.metrics.streaming_recall_at_k(predictions, labels, k, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_recall_at_k}

Computes the recall@k of the predictions with respect to dense labels.

The `streaming_recall_at_k` function creates two local variables, `total` and
`count`, that are used to compute the recall@k frequency. This frequency is
ultimately returned as `recall_at_<k>`: an idempotent operation that simply
divides `total` by `count`. To facilitate the estimation of recall@k over a
stream of data, the function utilizes two operations. First, an `in_top_k`
operation computes a tensor with shape [batch_size] whose elements indicate
whether or not the corresponding label is in the top `k` predictions of the
`predictions` `Tensor`. Second, an `update_op` operation whose behavior is
dependent on the value of `ignore_mask`. If `ignore_mask` is None, then
`update_op` increments `total` with the number of elements of `in_top_k` that
are set to `True` and increments `count` with the batch size. If `ignore_mask`
is not `None`, then `update_op` increments `total` with the number of elements
in `in_top_k` that are `True` whose corresponding element in `ignore_mask` is
`False`. In addition to performing the updates, `update_op` also returns the
recall value.

##### Args:


*  <b>`predictions`</b>: A floating point tensor of dimension [batch_size, num_classes]
*  <b>`labels`</b>: A tensor of dimension [batch_size] whose type is in `int32`,
    `int64`.
*  <b>`k`</b>: The number of top elements to look at for computing precision.
*  <b>`ignore_mask`</b>: An optional, binary tensor whose size matches `labels`. If an
    element of `ignore_mask` is True, the corresponding prediction and label
    pair is used to compute the metrics. Otherwise, the pair is ignored.
*  <b>`metrics_collections`</b>: An optional list of collections that `recall_at_k`
    should be added to.
*  <b>`updates_collections`</b>: An optional list of collections `update_op` should be
    added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`recall_at_k`</b>: A tensor representing the recall@k, the fraction of labels
    which fall into the top `k` predictions.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `recall_at_k`.

##### Raises:


*  <b>`ValueError`</b>: If the dimensions of `predictions` and `labels` don't match or
    if `ignore_mask` is not `None` and its shape doesn't match `predictions`
    or if either `metrics_collections` or `updates_collections` are not a list
    or tuple.


- - -

### `tf.contrib.metrics.streaming_mean_absolute_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_absolute_error}

Computes the mean absolute error between the labels and predictions.

The `streaming_mean_absolute_error` function creates two local variables,
`total` and `count` that are used to compute the mean absolute error. This
average is ultimately returned as `mean_absolute_error`: an idempotent
operation that simply divides `total` by `count`. To facilitate the estimation
of the mean absolute error over a stream of data, the function utilizes two
operations. First, an `absolute_errors` operation computes the absolute value
of the differences between `predictions` and `labels`. Second, an `update_op`
operation whose behavior is dependent on the value of `weights`. If `weights`
is None, then `update_op` increments `total` with the reduced sum of
`absolute_errors` and increments `count` with the number of elements in
`absolute_errors`. If `weights` is not `None`, then `update_op` increments
`total` with the reduced sum of the product of `weights` and `absolute_errors`
and increments `count` with the reduced sum of `weights`. In addition to
performing the updates, `update_op` also returns the `mean_absolute_error`
value.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional set of weights of the same shape as `predictions`. If
    `weights` is not None, the function computes a weighted mean.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_absolute_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean_absolute_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_absolute_error`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match
    `predictions` or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_mean_relative_error(predictions, labels, normalizer, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_relative_error}

Computes the mean relative error by normalizing with the given values.

The `streaming_mean_relative_error` function creates two local variables,
`total` and `count` that are used to compute the mean relative absolute error.
This average is ultimately returned as `mean_relative_error`: an idempotent
operation that simply divides `total` by `count`. To facilitate the estimation
of the mean relative error over a stream of data, the function utilizes two
operations. First, a `relative_errors` operation divides the absolute value
of the differences between `predictions` and `labels` by the `normalizer`.
Second, an `update_op` operation whose behavior is dependent on the value of
`weights`. If `weights` is None, then `update_op` increments `total` with the
reduced sum of `relative_errors` and increments `count` with the number of
elements in `relative_errors`. If `weights` is not `None`, then `update_op`
increments `total` with the reduced sum of the product of `weights` and
`relative_errors` and increments `count` with the reduced sum of `weights`. In
addition to performing the updates, `update_op` also returns the
`mean_relative_error` value.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`normalizer`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional set of weights of the same shape as `predictions`. If
    `weights` is not None, the function computes a weighted mean.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_relative_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean_relative_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_relative_error`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match
    `predictions` or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_mean_squared_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_squared_error}

Computes the mean squared error between the labels and predictions.

The `streaming_mean_squared_error` function creates two local variables,
`total` and `count` that are used to compute the mean squared error.
This average is ultimately returned as `mean_squared_error`: an idempotent
operation that simply divides `total` by `count`. To facilitate the estimation
of the mean squared error over a stream of data, the function utilizes two
operations. First, a `squared_error` operation computes the element-wise
square of the difference between `predictions` and `labels`. Second, an
`update_op` operation whose behavior is dependent on the value of `weights`.
If `weights` is None, then `update_op` increments `total` with the
reduced sum of `squared_error` and increments `count` with the number of
elements in `squared_error`. If `weights` is not `None`, then `update_op`
increments `total` with the reduced sum of the product of `weights` and
`squared_error` and increments `count` with the reduced sum of `weights`. In
addition to performing the updates, `update_op` also returns the
`mean_squared_error` value.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional set of weights of the same shape as `predictions`. If
    `weights` is not None, the function computes a weighted mean.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `mean_squared_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean_squared_error`</b>: A tensor representing the current mean, the value of
    `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `mean_squared_error`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match
    `predictions` or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_root_mean_squared_error(predictions, labels, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_root_mean_squared_error}

Computes the root mean squared error between the labels and predictions.

The `streaming_root_mean_squared_error` function creates two local variables,
`total` and `count` that are used to compute the root mean squared error.
This average is ultimately returned as `root_mean_squared_error`: an
idempotent operation that takes the square root of the division of `total`
by `count`. To facilitate the estimation of the root mean squared error over a
stream of data, the function utilizes two operations. First, a `squared_error`
operation computes the element-wise square of the difference between
`predictions` and `labels`. Second, an `update_op` operation whose behavior is
dependent on the value of `weights`. If `weights` is None, then `update_op`
increments `total` with the reduced sum of `squared_error` and increments
`count` with the number of elements in `squared_error`. If `weights` is not
`None`, then `update_op` increments `total` with the reduced sum of the
product of `weights` and `squared_error` and increments `count` with the
reduced sum of `weights`. In addition to performing the updates, `update_op`
also returns the `root_mean_squared_error` value.

##### Args:


*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`labels`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: An optional set of weights of the same shape as `predictions`. If
    `weights` is not None, the function computes a weighted mean.
*  <b>`metrics_collections`</b>: An optional list of collections that
    `root_mean_squared_error` should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that `update_op` should
    be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`root_mean_squared_error`</b>: A tensor representing the current mean, the value
    of `total` divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately and whose value matches `root_mean_squared_error`.

##### Raises:


*  <b>`ValueError`</b>: If `weights` is not `None` and its shape doesn't match
    `predictions` or if either `metrics_collections` or `updates_collections`
    are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_mean_cosine_distance(predictions, labels, dim, weights=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_mean_cosine_distance}

Computes the cosine distance between the labels and predictions.

The `streaming_mean_cosine_distance` function creates two local variables,
`total` and `count` that are used to compute the average cosine distance
between `predictions` and `labels`. This average is ultimately returned as
`mean_distance` which is an idempotent operation that simply divides `total`
by `count. To facilitate the estimation of a mean over multiple batches
of data, the function creates an `update_op` operation whose behavior is
dependent on the value of `weights`. If `weights` is None, then `update_op`
increments `total` with the reduced sum of `values and increments `count` with
the number of elements in `values`. If `weights` is not `None`, then
`update_op` increments `total` with the reduced sum of the product of `values`
and `weights` and increments `count` with the reduced sum of weights.

##### Args:


*  <b>`predictions`</b>: A tensor of the same size as labels.
*  <b>`labels`</b>: A tensor of arbitrary size.
*  <b>`dim`</b>: The dimension along which the cosine distance is computed.
*  <b>`weights`</b>: An optional set of weights which indicates which predictions to
    ignore during metric computation. Its size matches that of labels except
    for the value of 'dim' which should be 1. For example if labels has
    dimensions [32, 100, 200, 3], then `weights` should have dimensions
    [32, 100, 200, 1].
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`mean_distance`</b>: A tensor representing the current mean, the value of `total`
    divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately.

##### Raises:


*  <b>`ValueError`</b>: If labels and predictions are of different sizes or if the
    ignore_mask is of the wrong size or if either `metrics_collections` or
    `updates_collections` are not a list or tuple.


- - -

### `tf.contrib.metrics.streaming_percentage_less(values, threshold, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_percentage_less}

Computes the percentage of values less than the given threshold.

The `streaming_percentage_less` function creates two local variables,
`total` and `count` that are used to compute the percentage of `values` that
fall below `threshold`. This rate is ultimately returned as `percentage`
which is an idempotent operation that simply divides `total` by `count.
To facilitate the estimation of the percentage of values that fall under
`threshold` over multiple batches of data, the function creates an
`update_op` operation whose behavior is dependent on the value of
`ignore_mask`. If `ignore_mask` is None, then `update_op`
increments `total` with the number of elements of `values` that are less
than `threshold` and `count` with the number of elements in `values`. If
`ignore_mask` is not `None`, then `update_op` increments `total` with the
number of elements of `values` that are less than `threshold` and whose
corresponding entries in `ignore_mask` are False, and `count` is incremented
with the number of elements of `ignore_mask` that are False.

##### Args:


*  <b>`values`</b>: A numeric `Tensor` of arbitrary size.
*  <b>`threshold`</b>: A scalar threshold.
*  <b>`ignore_mask`</b>: An optional mask of the same shape as 'values' which indicates
    which elements to ignore during metric computation.
*  <b>`metrics_collections`</b>: An optional list of collections that the metric
    value variable should be added to.
*  <b>`updates_collections`</b>: An optional list of collections that the metric update
    ops should be added to.
*  <b>`name`</b>: An optional variable_op_scope name.

##### Returns:


*  <b>`percentage`</b>: A tensor representing the current mean, the value of `total`
    divided by `count`.
*  <b>`update_op`</b>: An operation that increments the `total` and `count` variables
    appropriately.

##### Raises:


*  <b>`ValueError`</b>: If `ignore_mask` is not None and its shape doesn't match `values
    or if either `metrics_collections` or `updates_collections` are supplied
    but are not a list or tuple.


- - -

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


- - -

### `tf.contrib.metrics.streaming_sparse_recall_at_k(predictions, labels, k, class_id=None, ignore_mask=None, metrics_collections=None, updates_collections=None, name=None)` {#streaming_sparse_recall_at_k}

Computes recall@k of the predictions with respect to sparse labels.

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
`true_positive_at_<k>` by total (`true_positive_at_<k>` + `recall_at_<k>`). To
facilitate the estimation of recall@k over a stream of data, the function
utilizes three steps.
* A `top_k` operation computes a tensor whose elements indicate the top `k`
  predictions of the `predictions` `Tensor`.
* Set operations are applied to `top_k` and `labels` to calculate true
  positives and false negatives.
* An `update_op` operation increments `true_positive_at_<k>` and
  `false_negative_at_<k>`. It also returns the recall value.

##### Args:


*  <b>`predictions`</b>: Float `Tensor` with shape [D1, ... DN, num_classes] where
    N >= 1. Commonly, N=1 and predictions has shape [batch size, num_classes].
    The final dimension contains the logit values for each class. [D1, ... DN]
    must match `labels`.
*  <b>`labels`</b>: `int64` `Tensor` or `SparseTensor` with shape
    [D1, ... DN, num_labels], where N >= 1 and num_labels is the number of
    target classes for the associated prediction. Commonly, N=1 and `labels`
    has shape [batch_size, num_labels]. [D1, ... DN] must match `labels`.
    Values should be in range [0, num_classes], where num_classes is the last
    dimension of `predictions`.
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


*  <b>`recall`</b>: Scalar `float64` `Tensor` with the value of `true_positives` divided
    by the sum of `true_positives` and `false_negatives`.
*  <b>`update_op`</b>: `Operation` that increments `true_positives` and
    `false_negatives` variables appropriately, and whose value matches
    `recall`.


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
          integer or string dtype.
*  <b>`weights`</b>: None or `Tensor` of float values to reweight the accuracy.

##### Returns:

  Accuracy `Tensor`.

##### Raises:


*  <b>`ValueError`</b>: if dtypes don't match or
              if dtype is not integer or string.


- - -

### `tf.contrib.metrics.confusion_matrix(predictions, labels, num_classes=None, name=None)` {#confusion_matrix}

Computes the confusion matrix from predictions and labels

Calculate the Confusion Matrix for a pair of prediction and
label 1-D int arrays.

Considering a prediction array such as: `[1, 2, 3]`
And a label array such as: `[2, 2, 3]`

##### The confusion matrix returned would be the following one:

    [[0, 0, 0]
     [0, 1, 0]
     [0, 1, 0]
     [0, 0, 1]]

Where the matrix rows represent the prediction labels and the columns
represents the real labels. The confusion matrix is always a 2-D array
of shape [n, n], where n is the number of valid labels for a given
classification task. Both prediction and labels must be 1-D arrays of
the same shape in order for this function to work.

##### Args:


*  <b>`predictions`</b>: A 1-D array represeting the predictions for a given
               classification.
*  <b>`labels`</b>: A 1-D represeting the real labels for the classification task.
*  <b>`num_classes`</b>: The possible number of labels the classification task can
               have. If this value is not provided, it will be calculated
               using both predictions and labels array.
*  <b>`name`</b>: Scope name.

##### Returns:

  A l X l matrix represeting the confusion matrix, where l in the number of
  possible labels in the classification task.

##### Raises:


*  <b>`ValueError`</b>: If both predictions and labels are not 1-D vectors and do not
              have the same size.



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

  For `a` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
  `n-1` dimensions as `a`. Each value is the number of unique elements in
  the corresponding `[0...n-1]` dimension of `a`.

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


