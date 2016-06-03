### `tf.contrib.framework.with_same_shape(expected_tensor, tensor)` {#with_same_shape}

Assert tensors are the same shape, from the same graph.

##### Args:


*  <b>`expected_tensor`</b>: Tensor with expected shape.
*  <b>`tensor`</b>: Tensor of actual values.

##### Returns:

  Tuple of (actual_tensor, label_tensor), possibly with assert ops added.

