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
sess.run(tf.local_variables_initializer())

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
by `tf.local_variables_initializer()`. It sets the sum and count variables to
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

sess.run(tf.local_variables_initializer())
for batch in range(num_batches):
  sess.run([update_op_acc, update_op_error])

accuracy, error = sess.run([accuracy, error])
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

## Metric `Ops`

*   `tf.contrib.metrics.streaming_accuracy`
*   `tf.contrib.metrics.streaming_mean`
*   `tf.contrib.metrics.streaming_recall`
*   `tf.contrib.metrics.streaming_recall_at_thresholds`
*   `tf.contrib.metrics.streaming_precision`
*   `tf.contrib.metrics.streaming_precision_at_thresholds`
*   `tf.contrib.metrics.streaming_auc`
*   `tf.contrib.metrics.streaming_recall_at_k`
*   `tf.contrib.metrics.streaming_mean_absolute_error`
*   `tf.contrib.metrics.streaming_mean_iou`
*   `tf.contrib.metrics.streaming_mean_relative_error`
*   `tf.contrib.metrics.streaming_mean_squared_error`
*   `tf.contrib.metrics.streaming_mean_tensor`
*   `tf.contrib.metrics.streaming_root_mean_squared_error`
*   `tf.contrib.metrics.streaming_covariance`
*   `tf.contrib.metrics.streaming_pearson_correlation`
*   `tf.contrib.metrics.streaming_mean_cosine_distance`
*   `tf.contrib.metrics.streaming_percentage_less`
*   `tf.contrib.metrics.streaming_sensitivity_at_specificity`
*   `tf.contrib.metrics.streaming_sparse_average_precision_at_k`
*   `tf.contrib.metrics.streaming_sparse_precision_at_k`
*   `tf.contrib.metrics.streaming_sparse_precision_at_top_k`
*   `tf.contrib.metrics.streaming_sparse_recall_at_k`
*   `tf.contrib.metrics.streaming_specificity_at_sensitivity`
*   `tf.contrib.metrics.streaming_concat`
*   `tf.contrib.metrics.streaming_false_negatives`
*   `tf.contrib.metrics.streaming_false_negatives_at_thresholds`
*   `tf.contrib.metrics.streaming_false_positives`
*   `tf.contrib.metrics.streaming_false_positives_at_thresholds`
*   `tf.contrib.metrics.streaming_true_negatives`
*   `tf.contrib.metrics.streaming_true_negatives_at_thresholds`
*   `tf.contrib.metrics.streaming_true_positives`
*   `tf.contrib.metrics.streaming_true_positives_at_thresholds`
*   `tf.contrib.metrics.auc_using_histogram`
*   `tf.contrib.metrics.accuracy`
*   `tf.contrib.metrics.aggregate_metrics`
*   `tf.contrib.metrics.aggregate_metric_map`
*   `tf.contrib.metrics.confusion_matrix`

## Set `Ops`

*   `tf.contrib.metrics.set_difference`
*   `tf.contrib.metrics.set_intersection`
*   `tf.contrib.metrics.set_size`
*   `tf.contrib.metrics.set_union`
