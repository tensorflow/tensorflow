### `tf.contrib.layers.convolution2d(x, num_output_channels, kernel_size, activation_fn=None, stride=(1, 1), padding='SAME', weight_init=_initializer, bias_init=_initializer, name=None, weight_collections=None, bias_collections=None, output_collections=None, trainable=True, weight_regularizer=None, bias_regularizer=None)` {#convolution2d}

Adds the parameters for a conv2d layer and returns the output.

A neural network convolution layer is generally defined as:
\\(y = f(conv2d(w, x) + b)\\) where **f** is given by `activation_fn`,
**conv2d** is `tf.nn.conv2d` and `x` has shape
`[batch, height, width, channels]`. The output of this op is of shape
`[batch, out_height, out_width, num_output_channels]`, where `out_width` and
`out_height` are determined by the `padding` argument. See `conv2D` for
details.

This op creates `w` and optionally `b` and adds various summaries that can be
useful for visualizing learning or diagnosing training problems. Bias can be
disabled by setting `bias_init` to `None`.

The variable creation is compatible with `tf.variable_scope` and so can be
reused with `tf.variable_scope` or `tf.make_template`.

Most of the details of variable creation can be controlled by specifying the
initializers (`weight_init` and `bias_init`) and which collections to place
the created variables in (`weight_collections` and `bias_collections`).

A per layer regularization can be specified by setting `weight_regularizer`.
This is only applied to weights and not the bias.

##### Args:


*  <b>`x`</b>: A 4-D input `Tensor`.
*  <b>`num_output_channels`</b>: The number of output channels (i.e. the size of the
    last dimension of the output).
*  <b>`kernel_size`</b>: A length 2 `list` or `tuple` containing the kernel size.
*  <b>`activation_fn`</b>: A function that requires a single Tensor that is applied as a
    non-linearity.
*  <b>`stride`</b>: A length 2 `list` or `tuple` specifying the stride of the sliding
    window across the image.
*  <b>`padding`</b>: A `string` from: "SAME", "VALID". The type of padding algorithm to
    use.
*  <b>`weight_init`</b>: An optional initialization. If not specified, uses Xavier
    initialization (see `tf.learn.xavier_initializer`).
*  <b>`bias_init`</b>: An initializer for the bias, defaults to 0. Set to`None` in order
    to disable bias.
*  <b>`name`</b>: The name for this operation is used to name operations and to find
    variables. If specified it must be unique for this scope, otherwise a
    unique name starting with "convolution2d" will be created.  See
    `tf.variable_op_scope` for details.
*  <b>`weight_collections`</b>: List of graph collections to which weights are added.
*  <b>`bias_collections`</b>: List of graph collections to which biases are added.
*  <b>`output_collections`</b>: List of graph collections to which outputs are added.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`weight_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for weights.
*  <b>`bias_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for biases.

##### Returns:

  The result of applying a 2-D convolutional layer.

##### Raises:


*  <b>`ValueError`</b>: If `kernel_size` or `stride` are not length 2.

