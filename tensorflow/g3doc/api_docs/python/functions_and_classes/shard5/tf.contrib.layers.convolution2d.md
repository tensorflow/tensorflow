### `tf.contrib.layers.convolution2d(*args, **kwargs)` {#convolution2d}

Adds an N-D convolution followed by an optional batch_norm layer.

It is required that 1 <= N <= 3.

`convolution` creates a variable called `weights`, representing the
convolutional kernel, that is convolved (actually cross-correlated) with the
`inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
provided (such as `batch_norm`), it is then applied. Otherwise, if
`normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
variable would be created and added the activations. Finally, if
`activation_fn` is not `None`, it is applied to the activations as well.

Performs a'trous convolution with input stride/dilation rate equal to `rate`
if a value > 1 for any dimension of `rate` is specified.  In this case
`stride` values != 1 are not supported.

##### Args:


*  <b>`inputs`</b>: a Tensor of rank N+2 of shape
    `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
    not start with "NC" (default), or
    `[batch_size, in_channels] + input_spatial_shape` if data_format starts
    with "NC".
*  <b>`num_outputs`</b>: integer, the number of output filters.
*  <b>`kernel_size`</b>: a sequence of N positive integers specifying the spatial
    dimensions of of the filters.  Can be a single integer to specify the same
    value for all spatial dimensions.
*  <b>`stride`</b>: a sequence of N positive integers specifying the stride at which to
    compute output.  Can be a single integer to specify the same value for all
    spatial dimensions.  Specifying any `stride` value != 1 is incompatible
    with specifying any `rate` value != 1.
*  <b>`padding`</b>: one of `"VALID"` or `"SAME"`.
*  <b>`data_format`</b>: A string or None.  Specifies whether the channel dimension of
    the `input` and output is the last dimension (default, or if `data_format`
    does not start with "NC"), or the second dimension (if `data_format`
    starts with "NC").  For N=1, the valid values are "NWC" (default) and
    "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".  For
    N=3, currently the only valid value is "NDHWC".
*  <b>`rate`</b>: a sequence of N positive integers specifying the dilation rate to use
    for a'trous convolution.  Can be a single integer to specify the same
    value for all spatial dimensions.  Specifying any `rate` value != 1 is
    incompatible with specifying any `stride` value != 1.
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

  a tensor representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if `data_format` is invalid.
*  <b>`ValueError`</b>: both 'rate' and `stride` are not uniformly 1.

