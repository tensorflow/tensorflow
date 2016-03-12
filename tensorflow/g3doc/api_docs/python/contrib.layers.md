<!-- This file is machine generated: DO NOT EDIT! -->

# Layers (contrib)
[TOC]

Ops for building neural network layers, regularizers, summaries, etc.

## Higher level ops for building neural network layers.

This package provides several ops that take care of creating variables that are
used internally in a consistent way and provide the building blocks for many
common machine learning algorithms.

- - -

### `tf.contrib.layers.convolution2d(x, num_output_channels, kernel_size, activation_fn=None, stride=(1, 1), padding='SAME', weight_init=_initializer, bias_init=_initializer, name=None, weight_collections=None, bias_collections=None, output_collections=None, weight_regularizer=None, bias_regularizer=None)` {#convolution2d}

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
*  <b>`weight_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for weights.
*  <b>`bias_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for biases.

##### Returns:

  The result of applying a 2-D convolutional layer.

##### Raises:


*  <b>`ValueError`</b>: If `kernel_size` or `stride` are not length 2.


- - -

### `tf.contrib.layers.fully_connected(x, num_output_units, activation_fn=None, weight_init=_initializer, bias_init=_initializer, name=None, weight_collections=('weights',), bias_collections=('biases',), output_collections=('activations',), weight_regularizer=None, bias_regularizer=None)` {#fully_connected}

Adds the parameters for a fully connected layer and returns the output.

A fully connected layer is generally defined as a matrix multiply:
`y = f(w * x + b)` where `f` is given by `activation_fn`. If
`activation_fn` is `None`, the result of `y = w * x + b` is
returned.

This op creates `w` and optionally `b`. Bias (`b`) can be disabled by setting
`bias_init` to `None`.

The variable creation is compatible with `tf.variable_scope` and so can be
reused with `tf.variable_scope` or `tf.make_template`.

Most of the details of variable creation can be controlled by specifying the
initializers (`weight_init` and `bias_init`) and which in collections to place
the created variables (`weight_collections` and `bias_collections`; note that
the variables are always added to the `VARIABLES` collection). The output of
the layer can be placed in custom collections using `output_collections`.
The collections arguments default to `WEIGHTS`, `BIASES` and `ACTIVATIONS`,
respectively.

A per layer regularization can be specified by setting `weight_regularizer`
and `bias_regularizer`, which are applied to the weights and biases
respectively, and whose output is added to the `REGULARIZATION_LOSSES`
collection.

##### Args:


*  <b>`x`</b>: The input `Tensor`.
*  <b>`num_output_units`</b>: The size of the output.
*  <b>`activation_fn`</b>: A function that requires a single Tensor that is applied as a
    non-linearity. If None is used, do not apply any activation.
*  <b>`weight_init`</b>: An optional weight initialization, defaults to
    `xavier_initializer`.
*  <b>`bias_init`</b>: An initializer for the bias, defaults to 0. Set to `None` in
    order to disable bias.
*  <b>`name`</b>: The name for this operation is used to name operations and to find
    variables. If specified it must be unique for this scope, otherwise a
    unique name starting with "fully_connected" will be created.  See
    `tf.variable_op_scope` for details.
*  <b>`weight_collections`</b>: List of graph collections to which weights are added.
*  <b>`bias_collections`</b>: List of graph collections to which biases are added.
*  <b>`output_collections`</b>: List of graph collections to which outputs are added.
*  <b>`weight_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for weights.
*  <b>`bias_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for biases.

##### Returns:

  The output of the fully connected layer.



Aliases for fully_connected which set a default activation function are
available: `relu`, `relu6` and `linear`.

## Regularizers

Regularization can help prevent overfitting. These have the signature
`fn(weights)`. The loss is typically added to `tf.GraphKeys.REGULARIZATION_LOSS`

- - -

### `tf.contrib.layers.l1_regularizer(scale)` {#l1_regularizer}

Returns a function that can be used to apply L1 regularization to weights.

L1 regularization encourages sparsity.

##### Args:


*  <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

##### Returns:

  A function with signature `l1(weights, name=None)` that apply L1
  regularization.

##### Raises:


*  <b>`ValueError`</b>: If scale is outside of the range [0.0, 1.0] or if scale is not a
  float.


- - -

### `tf.contrib.layers.l2_regularizer(scale)` {#l2_regularizer}

Returns a function that can be used to apply L2 regularization to weights.

Small values of L2 can help prevent overfitting the training data.

##### Args:


*  <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

##### Returns:

  A function with signature `l2(weights, name=None)` that applies L2
  regularization.

##### Raises:


*  <b>`ValueError`</b>: If scale is outside of the range [0.0, 1.0] or if scale is not a
  float.



## Initializers

Initializers are used to initialize variables with sensible values given their
size, data type, and purpose.

- - -

### `tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)` {#xavier_initializer}

Returns an initializer performing "Xavier" initialization for weights.

This function implements the weight initialization from:

Xavier Glorot and Yoshua Bengio (2010):
         Understanding the difficulty of training deep feedforward neural
         networks. International conference on artificial intelligence and
         statistics.

This initializer is designed to keep the scale of the gradients roughly the
same in all layers. In uniform distribution this ends up being the range:
`x = sqrt(6. / (in + out)); [-x, x]` and for normal distribution a standard
deviation of `sqrt(3. / (in + out))` is used.

The returned initializer assumes that the shape of the weight matrix to be
initialized is `[in, out]`.

##### Args:


*  <b>`uniform`</b>: Whether to use uniform or normal distributed random initialization.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer for a 2-D weight matrix.

##### Raises:


*  <b>`TypeError`</b>: If dtype is not a floating point type.


- - -

### `tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)` {#xavier_initializer_conv2d}

Returns an "Xavier" initializer for 2D convolution weights.

For details on the initialization performed, see `xavier_initializer`. This
function initializes a convolution weight variable which is assumed to be 4-D.
The first two dimensions are expected to be the kernel size, the third
dimension is the number of input channels, and the last dimension is the
number of output channels.

The number of inputs is therefore `shape[0]*shape[1]*shape[2]`, and the number
of outputs is `shape[0]*shape[1]*shape[3]`.

##### Args:


*  <b>`uniform`</b>: Whether to use uniform or normal distributed random initialization.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer for a 4-D weight matrix.

##### Raises:


*  <b>`TypeError`</b>: If dtype is not a floating point type.



## Summaries

Helper functions to summarize specific variables or ops.

- - -

### `tf.contrib.layers.summarize_activation(op)` {#summarize_activation}

Summarize an activation.

This applies the given activation and adds useful summaries specific to the
activation.

##### Args:


*  <b>`op`</b>: The tensor to summarize (assumed to be a layer activation).

##### Returns:

  The summary op created to summarize `op`.


- - -

### `tf.contrib.layers.summarize_tensor(tensor, tag=None)` {#summarize_tensor}

Summarize a tensor using a suitable summary type.

This function adds a summary op for `tensor`. The type of summary depends on
the shape of `tensor`. For scalars, a `scalar_summary` is created, for all
other tensors, `histogram_summary` is used.

##### Args:


*  <b>`tensor`</b>: The tensor to summarize
*  <b>`tag`</b>: The tag to use, if None then use tensor's op's name.

##### Returns:

  The summary op created.


- - -

### `tf.contrib.layers.summarize_tensors(tensors, summarizer=summarize_tensor)` {#summarize_tensors}

Summarize a set of tensors.


- - -

### `tf.contrib.layers.summarize_collection(collection, name_filter=None, summarizer=summarize_tensor)` {#summarize_collection}

Summarize a graph collection of tensors, possibly filtered by name.



The layers module defines convenience functions `summarize_variables`,
`summarize_weights` and `summarize_biases`, which set the `collection` argument
of `summarize_collection` to `VARIABLES`, `WEIGHTS` and `BIASES`, respectively.

- - -

### `tf.contrib.layers.summarize_activations(name_filter=None, summarizer=summarize_activation)` {#summarize_activations}

Summarize activations, using `summarize_activation` to summarize.



## Utilities

- - -

### `tf.contrib.layers.assert_same_float_dtype(tensors=None, dtype=None)` {#assert_same_float_dtype}

Validate and return float type based on `tensors` and `dtype`.

For ops such as matrix multiplication, inputs and weights must be of the
same float type. This function validates that all `tensors` are the same type,
validates that type is `dtype` (if supplied), and returns the type. Type must
be `dtypes.float32` or `dtypes.float64`. If neither `tensors` nor
`dtype` is supplied, default to `dtypes.float32`.

##### Args:


*  <b>`tensors`</b>: Tensors of input values. Can include `None` elements, which will be
      ignored.
*  <b>`dtype`</b>: Expected type.

##### Returns:

  Validated type.

##### Raises:


*  <b>`ValueError`</b>: if neither `tensors` nor `dtype` is supplied, or result is not
      float.


- - -

### `tf.contrib.layers.assert_scalar_int(tensor)` {#assert_scalar_int}

Assert `tensor` is 0-D, of type `tf.int32` or `tf.int64`.

##### Args:


*  <b>`tensor`</b>: Tensor to test.

##### Returns:

  `tensor`, for chaining.

##### Raises:


*  <b>`ValueError`</b>: if `tensor` is not 0-D, of type `tf.int32` or `tf.int64`.


- - -

### `tf.contrib.layers.is_numeric_tensor(tensor)` {#is_numeric_tensor}





## Other Functions and Classes
- - -

### `tf.contrib.layers.absolute_loss(predicted, target, name=None)` {#absolute_loss}

Computes and returns the per-example absolute loss.

Computes the per-example absolute value of the difference between
the target and predicted tensors. The tensors must have the same
shape.

##### Args:


*  <b>`predicted`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
    of predicted values.
*  <b>`target`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
    target values. The shape of the target tensor should match the
    `predicted` tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `[batch_size, dim_1, ..., dim_n]` tensor of per-example absolute losses.

##### Raises:


*  <b>`ValueError`</b>: If `predicted` and `target` shapes do not match.


- - -

### `tf.contrib.layers.mean_absolute_loss(predicted, target, name=None)` {#mean_absolute_loss}

Calculates the mean absolute loss across batches.

Computes the absolute difference between the target and predicted
tensors, averaged across all dimensions except dimension 0:

      losses = reduce_batch_mean(absolute_loss(predicted, target))

where `losses` is a tensor with dimensions [batch_size].

The tensors must have the same shape.

This loss function is a form of L1 loss.

##### Args:


*  <b>`predicted`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
    of predicted values.
*  <b>`target`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
    target values. The shape of the target tensor should match the
    `predicted` tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `[batch_size]` tensor of absolute differences, averaged across all
  dimensions except dimension 0.

##### Raises:


*  <b>`ValueError`</b>: If `predicted` and `target` shapes do not match.


- - -

### `tf.contrib.layers.mean_squared_loss(predicted, target, name=None)` {#mean_squared_loss}

Calculates the mean squared loss across batches.

Computes the squared difference between the target and predicted
tensors, and averages across all dimensions except dimension 0:

      losses = reduce_batch_mean(squared_loss(predicted, target))

where `losses` is a tensor with dimensions [batch_size].

The tensors must have the same shape.

##### Args:


*  <b>`predicted`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
    of predicted values.
*  <b>`target`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
    target values. The shape of the target tensor should match the
    `predicted` tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `[batch_size]` tensor of squared differences, averaged across
  all dimensions except dimension 0.

##### Raises:


*  <b>`ValueError`</b>: If `predicted` and `target` shapes do not match.


- - -

### `tf.contrib.layers.reduce_batch_mean(x, name=None)` {#reduce_batch_mean}

Given a tensor `x`, returns the mean across all dimensions except dim 0.

Given a tensor with the number of dimensions > 1, reduce_batch_mean
will calculate the mean across all dimensions except for dimension
0. This function is useful for calculating the mean loss (error)
across all examples in a batch when training. As an example, given a
tensor of shape [batch_size, d1, d2], this function will calculate
the mean across dimensions d1 and d2, returning a tensor of shape
[batch_size].

Tensors of dimension 1 are returned as-is.

##### Args:


*  <b>`x`</b>: A `Tensor` with dimension > 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with values averaged across all dimensions > 0.

##### Raises:


*  <b>`ValueError`</b>: If `x` has dimension 0.


- - -

### `tf.contrib.layers.reduce_batch_sum(x, name=None)` {#reduce_batch_sum}

Given a tensor `x`, sums across all dimensions except dimension 0.

Given a tensor with the number of dimensions > 1, reduce_batch_sum
will sum across all dimensions except for dimension 0. This function
is useful for summing the loss (error) across all examples in a
batch when training. As an example, given a tensor of shape
[batch_size, d1, d2], this function will sum across dimensions d1
and d2, returning a tensor of shape [batch_size].

Tensors of dimension 1 are returned as-is, while tensors of dimension 0
raise a ValueError.

##### Args:


*  <b>`x`</b>: A `Tensor` with dimension > 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` with values summed across all dimensions > 0.

##### Raises:


*  <b>`ValueError`</b>: If `x` has dimension 0.


- - -

### `tf.contrib.layers.root_mean_squared_loss(predicted, target, name=None)` {#root_mean_squared_loss}

Calculates the root mean squared loss across batches.

Computes the root mean squared loss between the target and predicted
tensors, which is the square root of the mean squared differences
between the predicted and target tensors:

      losses = sqrt(mean_squared_loss(predicted, target))

where `losses` is a tensor with dimensions [batch_size].

The tensors must have the same shape.

##### Args:


*  <b>`predicted`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
    of predicted values.
*  <b>`target`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
    target values. The shape of the target tensor should match the
    `predicted` tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `[batch_size]` tensor of the root mean squared differences.

##### Raises:


*  <b>`ValueError`</b>: If `predicted` and `target` shapes do not match.


- - -

### `tf.contrib.layers.squared_loss(predicted, target, name=None)` {#squared_loss}

Computes and returns the per-example squared loss.

Computes the per-example squared difference between the target and
predicted tensors. The tensors must have the same shape.

##### Args:


*  <b>`predicted`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
    of predicted values.
*  <b>`target`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
    target values. The shape of the target tensor should match the
    `predicted` tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `[batch_size, dim_1, ..., dim_n]` tensor of per-example squared losses.

##### Raises:


*  <b>`ValueError`</b>: If `predicted` and `target` shapes do not match.


- - -

### `tf.contrib.layers.sum_squared_loss(predicted, target, name=None)` {#sum_squared_loss}

Calculates 1/2 the sum of the squared loss across batches.

Computes the squared difference between the target and predicted
tensors, sums across all dimensions except dimension 0, and divides
by 2:

    losses = reduce_batch_sum(squared_loss(predicted, target)) / 2.0

where `losses` is a tensor with dimensions [batch_size].

The tensors must have the same shape.

This function is equivalent to typical formulations of L2 loss, and similar
to TensorFlow's l2_loss function. It differs from the l2_loss function
by allowing the caller to specify both the predicted and target tensors.

##### Args:


*  <b>`predicted`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]`
    of predicted values.
*  <b>`target`</b>: A `Tensor` of shape `[batch_size, dim_1, ..., dim_n]` of
    target values. The shape of the target tensor should match the
    `predicted` tensor.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `[batch_size]` tensor of squared losses summed across all dimensions
  except dimension 0, divided by 2.

##### Raises:


*  <b>`ValueError`</b>: If `predicted` and `target` shapes do not match.


