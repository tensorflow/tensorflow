<!-- This file is machine generated: DO NOT EDIT! -->

# Layers (contrib)
[TOC]

Ops for building neural network layers, regularizers, summaries, etc.

## Higher level ops for building neural network layers.

This package provides several ops that take care of creating variables that are
used internally in a consistent way and provide the building blocks for many
common machine learning algorithms.

- - -

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


- - -

### `tf.contrib.layers.fully_connected(x, num_output_units, activation_fn=None, weight_init=_initializer, bias_init=_initializer, name=None, weight_collections=('weights',), bias_collections=('biases',), output_collections=('activations',), trainable=True, weight_regularizer=None, bias_regularizer=None)` {#fully_connected}

Adds the parameters for a fully connected layer and returns the output.

A fully connected layer is generally defined as a matrix multiply:
`y = f(w * x + b)` where `f` is given by `activation_fn`. If
`activation_fn` is `None`, the result of `y = w * x + b` is
returned.

If `x` has shape [\\\(\\text{dim}_0, \\text{dim}_1, ..., \\text{dim}_n\\\)]
with more than 2 dimensions (\\\(n > 1\\\)), then we repeat the matrix
multiply along the first dimensions. The result r is a tensor of shape
[\\\(\\text{dim}_0, ..., \\text{dim}_{n-1},\\\) `num_output_units`],
where \\\( r_{i_0, ..., i_{n-1}, k} =
\\sum_{0 \\leq j < \\text{dim}_n} x_{i_0, ... i_{n-1}, j} \cdot w_{j, k}\\\).
This is accomplished by reshaping `x` to 2-D
[\\\(\\text{dim}_0 \\cdot ... \\cdot \\text{dim}_{n-1}, \\text{dim}_n\\\)]
before the matrix multiply and afterwards reshaping it to
[\\\(\\text{dim}_0, ..., \\text{dim}_{n-1},\\\) `num_output_units`].

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
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`weight_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for weights.
*  <b>`bias_regularizer`</b>: A regularizer like the result of
    `l1_regularizer` or `l2_regularizer`. Used for biases.

##### Returns:

  The output of the fully connected layer.

##### Raises:


*  <b>`ValueError`</b>: if x has rank less than 2 or if its last dimension is not set.



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


- - -

### `tf.contrib.layers.sum_regularizer(regularizer_list)` {#sum_regularizer}

Returns a function that applies the sum of multiple regularizers.

##### Args:


*  <b>`regularizer_list`</b>: A list of regularizers to apply.

##### Returns:

  A function with signature `sum_reg(weights, name=None)` that applies the
  sum of all the input regularizers.



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



## Other Functions and Classes
- - -

### `tf.contrib.layers.apply_regularization(regularizer, weights_list=None)` {#apply_regularization}

Returns the summed penalty by applying `regularizer` to the `weights_list`.

Adding a regularization penalty over the layer weights and embedding weights
can help prevent overfitting the training data. Regularization over layer
biases is less common/useful, but assuming proper data preprocessing/mean
subtraction, it usually shouldn't hurt much either.

##### Args:


*  <b>`regularizer`</b>: A function that takes a single `Tensor` argument and returns
    a scalar `Tensor` output.
*  <b>`weights_list`</b>: List of weights `Tensors` or `Variables` to apply
    `regularizer` over. Defaults to the `GraphKeys.WEIGHTS` collection if
    `None`.

##### Returns:

  A scalar representing the overall regularization penalty.

##### Raises:


*  <b>`ValueError`</b>: If `regularizer` does not return a scalar output.


- - -

### `tf.contrib.layers.make_all(module_name, doc_string_modules=None)` {#make_all}

Generate `__all__` from the docstring of one or more modules.

Usage: `make_all(__name__)` or
`make_all(__name__, [sys.modules(__name__), other_module])`. The doc string
modules must each a docstring, and `__all__` will contain all symbols with
`@@` references, where that symbol currently exists in the module named
`module_name`.

##### Args:


*  <b>`module_name`</b>: The name of the module (usually `__name__`).
*  <b>`doc_string_modules`</b>: a list of modules from which to take docstring.
  If None, then a list containing only the module named `module_name` is used.

##### Returns:

  A list suitable for use as `__all__`.


- - -

### `tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, clip_gradients=None, moving_average_decay=0.9, learning_rate_decay_fn=None, variables=None)` {#optimize_loss}

Given loss and parameters for optimizer, returns a training op.

##### Args:


*  <b>`loss`</b>: Tensor, 0 dimensional.
*  <b>`global_step`</b>: Tensor, step counter for each update.
*  <b>`learning_rate`</b>: float or Tensor, magnitude of update per each training step.
*  <b>`optimizer`</b>: string, class or optimizer instance, used as trainer.
             string should be name of optimizer, like 'SGD',
               'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
             class should be sub-class of tf.Optimizer that implements
               `compute_gradients` and `apply_gradients` functions.
             optimizer instance should be instantion of tf.Optimizer sub-class
               and have `compute_gradients` and `apply_gradients` functions.
*  <b>`clip_gradients`</b>: float or None, clips gradients by this value.
*  <b>`moving_average_decay`</b>: float or None, takes into account previous loss
                        to make learning smoother due to outliers.
*  <b>`learning_rate_decay_fn`</b>: function, takes learning_rate and global_step
                          Tensors, returns Tensor. Can be used to implement
                          any learning rate decay funcitons.
                          For example: tf.train.exponential_decay.
*  <b>`variables`</b>: list of variables to optimizer or none.

##### Returns:

  Training op.

##### Raises:


*  <b>`ValueError`</b>: if optimizer is wrong type.


