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

