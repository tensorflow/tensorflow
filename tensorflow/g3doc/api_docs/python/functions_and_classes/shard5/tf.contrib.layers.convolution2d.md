### `tf.contrib.layers.convolution2d(*args, **kwargs)` {#convolution2d}

Adds a 2D convolution followed by an optional batch_norm layer.

`convolution2d` creates a variable called `weights`, representing the
convolutional kernel, that is convolved with the `inputs` to produce a
`Tensor` of activations. If a `normalizer_fn` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
None and a `biases_initializer` is provided then a `biases` variable would be
created and added the activations. Finally, if `activation_fn` is not `None`,
it is applied to the activations as well.

Performs a'trous convolution with input stride equal to rate if rate is
greater than one.

##### Args:


*  <b>`inputs`</b>: a 4-D tensor  `[batch_size, height, width, channels]`.
*  <b>`num_outputs`</b>: integer, the number of output filters.
*  <b>`kernel_size`</b>: a list of length 2 `[kernel_height, kernel_width]` of
    of the filters. Can be an int if both values are the same.
*  <b>`stride`</b>: a list of length 2 `[stride_height, stride_width]`.
    Can be an int if both strides are the same. Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: one of `VALID` or `SAME`.
*  <b>`rate`</b>: integer. If less than or equal to 1, a standard convolution is used.
    If greater than 1, than the a'trous convolution is applied and `stride`
    must be set to 1.
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
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for `variable_op_scope`.

##### Returns:

  a tensor representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if both 'rate' and `stride` are larger than one.

