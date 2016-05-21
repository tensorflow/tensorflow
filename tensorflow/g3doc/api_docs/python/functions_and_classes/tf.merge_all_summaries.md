### `tf.merge_all_summaries(key='summaries')` {#merge_all_summaries}

Merges all summaries collected in the default graph.

##### Args:


*  <b>`key`</b>: `GraphKey` used to collect the summaries.  Defaults to
    `GraphKeys.SUMMARIES`.

##### Returns:

  If no summaries were collected, returns None.  Otherwise returns a scalar
  `Tensor` of type `string` containing the serialized `Summary` protocol
  buffer resulting from the merging.

