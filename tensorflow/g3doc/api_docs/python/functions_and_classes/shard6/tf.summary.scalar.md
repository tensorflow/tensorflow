### `tf.summary.scalar(name, tensor, collections=None)` {#scalar}

Outputs a `Summary` protocol buffer containing a single scalar value.

The generated Summary has a Tensor.proto containing the input Tensor.

##### Args:


*  <b>`name`</b>: A name for the generated node. Will also serve as the series name in
    TensorBoard.
*  <b>`tensor`</b>: A real numeric Tensor containing a single value.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.

##### Returns:

  A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

##### Raises:


*  <b>`ValueError`</b>: If tensor has the wrong shape or type.

