<!-- This file is machine generated: DO NOT EDIT! -->

# Layers (contrib)
[TOC]

Ops for building neural network layers, regularizers, summaries, etc.

## Higher level ops for building neural network layers.

This package provides several ops that take care of creating variables that are
used internally in a consistent way and provide the building blocks for many
common machine learning algorithms.

- - -

### `tf.contrib.layers.convolution2d(*args, **kwargs)` {#convolution2d}

Adds a 2D convolution followed by an optional batch_norm layer.

`convolution2d` creates a variable called `weights`, representing the
convolutional kernel, that is convolved with the `inputs` to produce a
`Tensor` of activations. If a `normalizer_fn` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
None and a `biases_initializer` is provided then a `biases` variable would be
created and added the activations. Finally, if `activation_fn` is not `None`,
it is applied to the activations as well.

##### Args:


*  <b>`inputs`</b>: a 4-D tensor  `[batch_size, height, width, channels]`.
*  <b>`num_outputs`</b>: integer, the number of output filters.
*  <b>`kernel_size`</b>: a list of length 2 `[kernel_height, kernel_width]` of
    of the filters. Can be an int if both values are the same.
*  <b>`stride`</b>: a list of length 2 `[stride_height, stride_width]`.
    Can be an int if both strides are the same. Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: one of `VALID` or `SAME`.
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


- - -

### `tf.contrib.layers.fully_connected(*args, **kwargs)` {#fully_connected}

Adds a fully connected layer.

`fully_connected` creates a variable called `weights`, representing a fully
connected weight matrix, which is multiplied by the `inputs` to produce a
`Tensor` of hidden units. If a `normalizer_fn` is provided (such as
`batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
None and a `biases_initializer` is provided then a `biases` variable would be
created and added the hidden units. Finally, if `activation_fn` is not `None`,
it is applied to the hidden units as well.

Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
prior to the initial matrix multiply by `weights`.

##### Args:


*  <b>`inputs`</b>: A tensor of with at least rank 2 and value for the last dimension,
    i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
*  <b>`num_outputs`</b>: Integer, the number of output units in the layer.
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
*  <b>`variables_collections`</b>: Optional list of collections for all the variables or
    a dictionary containing a different list of collections per variable.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for variable_op_scope.

##### Returns:

   the tensor variable representing the result of the series of operations.

##### Raises:


*  <b>`ValueError`</b>: if x has rank less than 2 or if its last dimension is not set.



Aliases for fully_connected which set a default activation function are
available: `relu`, `relu6` and `linear`.

## Regularizers

Regularization can help prevent overfitting. These have the signature
`fn(weights)`. The loss is typically added to `tf.GraphKeys.REGULARIZATION_LOSS`

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

##### Args:


*  <b>`uniform`</b>: Whether to use uniform or normal distributed random initialization.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer for a weight matrix.


- - -

### `tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)` {#xavier_initializer_conv2d}

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

##### Args:


*  <b>`uniform`</b>: Whether to use uniform or normal distributed random initialization.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer for a weight matrix.


- - -

### `tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)` {#variance_scaling_initializer}

Returns an initializer that generates tensors without scaling variance.

When initializing a deep network, it is in principle advantageous to keep
the scale of the input variance constant, so it does not explode or diminish
by reaching the final layer. This initializer use the following formula:
  if mode='FAN_IN': # Count only number of input connections.
    n = fan_in
  elif mode='FAN_OUT': # Count only number of output connections.
    n = fan_out
  elif mode='FAN_AVG': # Average number of inputs and output connections.
    n = (fan_in + fan_out)/2.0

    truncated_normal(shape, 0.0, stddev=sqrt(factor / n))

To get http://arxiv.org/pdf/1502.01852v1.pdf use (Default):
  - factor=2.0 mode='FAN_IN' uniform=False
To get http://arxiv.org/abs/1408.5093 use:
  - factor=1.0 mode='FAN_IN' uniform=True
To get http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf use:
  - factor=1.0 mode='FAN_AVG' uniform=True.
To get xavier_initializer use either:
  - factor=1.0 mode='FAN_AVG' uniform=True.
  - factor=1.0 mode='FAN_AVG' uniform=False.

##### Args:


*  <b>`factor`</b>: Float.  A multiplicative factor.
*  <b>`mode`</b>: String.  'FAN_IN', 'FAN_OUT', 'FAN_AVG'.
*  <b>`uniform`</b>: Whether to use uniform or normal distributed random initialization.
*  <b>`seed`</b>: A Python integer. Used to create random seeds. See
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
    for behavior.
*  <b>`dtype`</b>: The data type. Only floating point types are supported.

##### Returns:

  An initializer that generates tensors with unit variance.

##### Raises:


*  <b>`ValueError`</b>: if `dtype` is not a floating point type.
*  <b>`TypeError`</b>: if `mode` is not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG'].



## Optimization

Optimize weights given a loss.

- - -

### `tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, gradient_noise_scale=None, gradient_multipliers=None, clip_gradients=None, moving_average_decay=0.9, learning_rate_decay_fn=None, update_ops=None, variables=None, name=None)` {#optimize_loss}

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
*  <b>`gradient_noise_scale`</b>: float or None, adds 0-mean normal noise scaled by this
                        value.
*  <b>`gradient_multipliers`</b>: dict of variables or variable names to floats.
                        If present, gradients for specified
                        variables will be multiplied by given constant.
*  <b>`clip_gradients`</b>: float or `None`, clips gradients by this value.
*  <b>`moving_average_decay`</b>: float or None, takes into account previous loss
                        to make learning smoother due to outliers.
*  <b>`learning_rate_decay_fn`</b>: function, takes `learning_rate` and `global_step`
                          `Tensor`s, returns `Tensor`.
                          Can be used to implement any learning rate decay
                          functions.
                          For example: tf.train.exponential_decay.
*  <b>`update_ops`</b>: list of update `Operation`s to execute at each step. If `None`,
              uses elements of UPDATE_OPS collection.
*  <b>`variables`</b>: list of variables to optimize or
             `None` to use all trainable variables.
*  <b>`name`</b>: The name for this operation is used to scope operations and summaries.

##### Returns:

  Training op.

##### Raises:


*  <b>`ValueError`</b>: if optimizer is wrong type.



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

  The summary op created or None for string tensors.


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


