### `tf.scalar_summary(tags, values, collections=None, name=None)` {#scalar_summary}

Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape.  The generated
summary has a summary value for each tag-value pair in `tags` and `values`.

##### Args:


*  <b>`tags`</b>: A `string` `Tensor`.  Tags for the summaries.
*  <b>`values`</b>: A real numeric Tensor.  Values for the summaries.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.

