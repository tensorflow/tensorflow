### `tf.contrib.layers.convolution2d_in_plane(*args, **kwargs)` {#convolution2d_in_plane}

Performs the same in-plane convolution to each channel independently.

This is useful for performing various simple channel-independent convolution
operations such as image gradients:

  image = tf.constant(..., shape=(16, 240, 320, 3))
  vert_gradients = layers.conv2d_in_plane(image,
                                          kernel=[1, -1],
                                          kernel_size=[2, 1])
  horz_gradients = layers.conv2d_in_plane(image,
                                          kernel=[1, -1],
                                          kernel_size=[1, 2])

##### Args:


*  <b>`inputs`</b>: a 4-D tensor with dimensions [batch_size, height, width, channels].
*  <b>`kernel_size`</b>: a list of length 2 holding the [kernel_height, kernel_width] of
    of the pooling. Can be an int if both values are the same.
*  <b>`stride`</b>: a list of length 2 `[stride_height, stride_width]`.
    Can be an int if both strides are the same. Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: the padding type to use, either 'SAME' or 'VALID'.
*  <b>`activation_fn`</b>: activation function, set to None to skip it and maintain
    a linear activation.
*  <b>`normalizer_fn`</b>: normalization function to use instead of `biases`. If
    `normalizer_fn` is provided then `biases_initializer` and
    `biases_regularizer` are ignored and `biases` are not created nor added.
    default set to None for no normalizer function
*  <b>`normalizer_params`</b>: normalization function parameters.
*  <b>`weights_initializer`</b>: An initializer for the weights.
*  <b>`weights_regularizer`</b>: Optional regularizer for the weights.
*  <b>`biases_initializer`</b>: An initializer for the biases. If None skip biases.
*  <b>`biases_regularizer`</b>: Optional regularizer for the biases.
*  <b>`reuse`</b>: whether or not the layer and its variables should be reused. To be
    able to reuse the layer scope must be given.
*  <b>`variables_collections`</b>: optional list of collections for all the variables or
    a dictionary containing a different list of collection per variable.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for `variable_scope`.

##### Returns:

  A `Tensor` representing the output of the operation.

