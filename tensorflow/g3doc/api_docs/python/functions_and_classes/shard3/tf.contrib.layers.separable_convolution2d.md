### `tf.contrib.layers.separable_convolution2d(*args, **kwargs)` {#separable_convolution2d}

Adds a depth-separable 2D convolution with optional batch_norm layer.

This op first performs a depthwise convolution that acts separately on
channels, creating a variable called `depthwise_weights`. If `num_outputs`
is not None, it adds a pointwise convolution that mixes channels, creating a
variable called `pointwise_weights`. Then, if `batch_norm_params` is None,
it adds bias to the result, creating a variable called 'biases', otherwise
it adds a batch normalization layer. It finally applies an activation function
to produce the end result.

##### Args:


*  <b>`inputs`</b>: a tensor of size [batch_size, height, width, channels].
*  <b>`num_outputs`</b>: the number of pointwise convolution output filters. If is
    None, then we skip the pointwise convolution stage.
*  <b>`kernel_size`</b>: a list of length 2: [kernel_height, kernel_width] of
    of the filters. Can be an int if both values are the same.
*  <b>`depth_multiplier`</b>: the number of depthwise convolution output channels for
    each input channel. The total number of depthwise convolution output
    channels will be equal to `num_filters_in * depth_multiplier`.
*  <b>`stride`</b>: a list of length 2: [stride_height, stride_width], specifying the
    depthwise convolution stride. Can be an int if both strides are the same.
*  <b>`padding`</b>: one of 'VALID' or 'SAME'.
*  <b>`activation_fn`</b>: activation function.
*  <b>`normalizer_fn`</b>: normalization function to use instead of `biases`. If
    `normalize_fn` is provided then `biases_initializer` and
    `biases_regularizer` are ignored and `biases` are not created nor added.
*  <b>`normalizer_params`</b>: normalization function parameters.
*  <b>`weights_initializer`</b>: An initializer for the weights.
*  <b>`weights_regularizer`</b>: Optional regularizer for the weights.
*  <b>`biases_initializer`</b>: An initializer for the biases. If None skip biases.
*  <b>`biases_regularizer`</b>: Optional regularizer for the biases.
*  <b>`reuse`</b>: whether or not the layer and its variables should be reused. To be
    able to reuse the layer scope must be given.
*  <b>`variables_collections`</b>: optional list of collections for all the variables or
    a dictionay containing a different list of collection per variable.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`trainable`</b>: whether or not the variables should be trainable or not.
*  <b>`scope`</b>: Optional scope for variable_op_scope.

##### Returns:

  A `Tensor` representing the output of the operation.

