### `tf.contrib.layers.avg_pool2d(*args, **kwargs)` {#avg_pool2d}

Adds a 2D average pooling op.

It is assumed that the pooling is done per image but not in batch or channels.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of size [batch_size, height, width, channels].
*  <b>`kernel_size`</b>: A list of length 2: [kernel_height, kernel_width] of the
    pooling kernel over which the op is computed. Can be an int if both
    values are the same.
*  <b>`stride`</b>: A list of length 2: [stride_height, stride_width].
    Can be an int if both strides are the same. Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: The padding method, either 'VALID' or 'SAME'.
*  <b>`outputs_collections`</b>: The collections to which the outputs are added.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  A `Tensor` representing the results of the pooling operation.

