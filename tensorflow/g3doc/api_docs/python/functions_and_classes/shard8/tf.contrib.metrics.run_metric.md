### `tf.contrib.metrics.run_metric(metric, predictions, targets, weights=None)` {#run_metric}

Runs a single metric.

This function runs metric on given predictions and targets. weights will be
used if metric contains 'weights' in its argument.

##### Args:


*  <b>`metric`</b>: A function that evaluates targets given predictions.
*  <b>`predictions`</b>: A `Tensor` of arbitrary shape.
*  <b>`targets`</b>: A `Tensor` of the same shape as `predictions`.
*  <b>`weights`</b>: A set of weights that can be used in metric function to compute
    weighted result.

##### Returns:


*  <b>`result`</b>: result returned by metric function.

