### `tf.summary.scalar(display_name, tensor, description='', labels=None, collections=None, name=None)` {#scalar}

Outputs a `Summary` protocol buffer containing a single scalar value.

The generated Summary has a Tensor.proto containing the input Tensor.

##### Args:


*  <b>`display_name`</b>: A name to associate with the data series. Will be used to
    organize output data and as a name in visualizers.
*  <b>`tensor`</b>: A tensor containing a single floating point or integer value.
*  <b>`description`</b>: An optional long description of the data being output.
*  <b>`labels`</b>: a list of strings used to attach metadata.
*  <b>`collections`</b>: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
*  <b>`name`</b>: An optional name for the generated node (optional).

##### Returns:

  A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

##### Raises:


*  <b>`ValueError`</b>: If tensor has the wrong shape or type.

