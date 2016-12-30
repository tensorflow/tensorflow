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


*  <b>`pearson_r`</b>: A `Tensor` representing the current Pearson product-moment
    correlation coefficient, the value of
    `cov(predictions, labels) / sqrt(var(predictions) * var(labels))`.
*  <b>`update_op`</b>: An operation that updates the underlying variables appropriately.

##### Raises:


*  <b>`ValueError`</b>: If `labels` and `predictions` are of different sizes, or if
    `weights` is the wrong size, or if either `metrics_collections` or
    `updates_collections` are not a `list` or `tuple`.

