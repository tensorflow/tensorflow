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


*  <b>`values`</b>: `Tensor` to concatenate. Rank and the shape along all axes other
    than the axis to concatenate along must be statically known.
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


*  <b>`value`</b>: A `Tensor` representing the concatenated values.
*  <b>`update_op`</b>: An operation that concatenates the next values.

##### Raises:


*  <b>`ValueError`</b>: if `values` does not have a statically known rank, `axis` is
    not in the valid range or the size of `values` is not statically known
    along any axis other than `axis`.

