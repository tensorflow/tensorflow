### `tf.contrib.metrics.aggregate_metrics(*value_update_tuples)` {#aggregate_metrics}

Aggregates the metric value tensors and update ops into two lists.

##### Args:


*  <b>`*value_update_tuples`</b>: a variable number of tuples, each of which contain the
    pair of (value_tensor, update_op) from a streaming metric.

##### Returns:

  a list of value tensors and a list of update ops.

##### Raises:


*  <b>`ValueError`</b>: if `value_update_tuples` is empty.

