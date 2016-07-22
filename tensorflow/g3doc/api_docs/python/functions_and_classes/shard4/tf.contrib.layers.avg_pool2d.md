### `tf.contrib.layers.avg_pool2d(*args, **kwargs)` {#avg_pool2d}

Adds a Avg Pooling op.

It is assumed by the wrapper that the pooling is only done per image and not
in depth or batch.

##### Args:


*  <b>`inputs`</b>: a tensor of size [batch_size, height, width, depth].
*  <b>`kernel_size`</b>: a list of length 2: [kernel_height, kernel_width] of the
    pooling kernel over which the op is computed. Can be an int if both
    values are the same.
*  <b>`stride`</b>: a list of length 2: [stride_height, stride_width].
    Can be an int if both strides are the same.  Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: the padding method, either 'VALID' or 'SAME'.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`scope`</b>: Optional scope for op_scope.

##### Returns:

  a tensor representing the results of the pooling operation.

