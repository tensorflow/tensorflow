### `tf.contrib.metrics.aggregate_metric_map(names_to_tuples)` {#aggregate_metric_map}

Aggregates the metric names to tuple dictionary.

This function is useful for pairing metric names with their associated value
and update ops when the list of metrics is long. For example:

```python
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
```

##### Args:


*  <b>`names_to_tuples`</b>: a map of metric names to tuples, each of which contain the
    pair of (value_tensor, update_op) from a streaming metric.

##### Returns:

  A dictionary from metric names to value ops and a dictionary from metric
  names to update ops.

