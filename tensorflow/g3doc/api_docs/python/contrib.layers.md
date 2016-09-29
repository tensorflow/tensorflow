<!-- This file is machine generated: DO NOT EDIT! -->

# Layers (contrib)
[TOC]

Ops for building neural network layers, regularizers, summaries, etc.

## Higher level ops for building neural network layers.

This package provides several ops that take care of creating variables that are
used internally in a consistent way and provide the building blocks for many
common machine learning algorithms.

- - -

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


- - -

### `tf.contrib.layers.batch_norm(*args, **kwargs)` {#batch_norm}

Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift"

  Sergey Ioffe, Christian Szegedy

Can be used as a normalizer function for conv2d and fully_connected.

Note: When is_training is True the moving_mean and moving_variance need to be
updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
they need to be added as a dependency to the `train_op`, example:

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if update_ops:
    updates = tf.group(*update_ops)
    total_loss = control_flow_ops.with_dependencies([updates], total_loss)

One can set update_collections=None to force the updates in place, but that
can have speed penalty, specially in distributed settings.

##### Args:


*  <b>`inputs`</b>: a tensor with 2 or more dimensions, where the first dimension has
    `batch_size`. The normalization is over all but the last dimension.
*  <b>`decay`</b>: decay for the moving average.
*  <b>`center`</b>: If True, subtract `beta`. If False, `beta` is ignored.
*  <b>`scale`</b>: If True, multiply by `gamma`. If False, `gamma` is
    not used. When the next layer is linear (also e.g. `nn.relu`), this can be
    disabled since the scaling can be done by the next layer.
*  <b>`epsilon`</b>: small float added to variance to avoid dividing by zero.
*  <b>`activation_fn`</b>: activation function, default set to None to skip it and
    maintain a linear activation.
*  <b>`updates_collections`</b>: collections to collect the update ops for computation.
    The updates_ops need to be executed with the train_op.
    If None, a control dependency would be added to make sure the updates are
    computed in place.
*  <b>`is_training`</b>: whether or not the layer is in training mode. In training mode
    it would accumulate the statistics of the moments into `moving_mean` and
    `moving_variance` using an exponential moving average with the given
    `decay`. When it is not in training mode then it would use the values of
    the `moving_mean` and the `moving_variance`.
*  <b>`reuse`</b>: whether or not the layer and its variables should be reused. To be
    able to reuse the layer scope must be given.
*  <b>`variables_collections`</b>: optional collections for the variables.
*  <b>`outputs_collections`</b>: collections to add the outputs.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
*  <b>`batch_weights`</b>: An optional tensor of shape `[batch_size]`,
    containing a frequency weight for each batch item. If present,
    then the batch normalization uses weighted mean and
    variance. (This can be used to correct for bias in training
    example selection.)
*  <b>`scope`</b>: Optional scope for `variable_scope`.

##### Returns:

  A `Tensor` representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if rank or last dimension of `inputs` is undefined.


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


*  <b>`ValueError`</b>: if both 'rate' and `stride` are larger than one.


- - -

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


- - -

### `tf.contrib.layers.convolution2d_transpose(*args, **kwargs)` {#convolution2d_transpose}

Adds a convolution2d_transpose with an optional batch normalization layer.

The function creates a variable called `weights`, representing the
kernel, that is convolved with the input. If `batch_norm_params` is `None`, a
second variable called 'biases' is added to the result of the operation.

##### Args:


*  <b>`inputs`</b>: a tensor of size [batch_size, height, width, channels].
*  <b>`num_outputs`</b>: integer, the number of output filters.
*  <b>`kernel_size`</b>: a list of length 2 holding the [kernel_height, kernel_width] of
    of the filters. Can be an int if both values are the same.
*  <b>`stride`</b>: a list of length 2: [stride_height, stride_width].
    Can be an int if both strides are the same.  Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: one of 'VALID' or 'SAME'.
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
*  <b>`trainable`</b>: whether or not the variables should be trainable or not.
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  a tensor representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if 'kernel_size' is not a list of length 2.


- - -

### `tf.contrib.layers.flatten(*args, **kwargs)` {#flatten}

Flattens the input while maintaining the batch_size.

  Assumes that the first dimension represents the batch.

##### Args:


*  <b>`inputs`</b>: a tensor of size [batch_size, ...].
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  a flattened tensor with shape [batch_size, k].

##### Raises:


*  <b>`ValueError`</b>: if inputs.shape is wrong.


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
*  <b>`num_outputs`</b>: Integer or long, the number of output units in the layer.
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
*  <b>`variables_collections`</b>: Optional list of collections for all the variables or
    a dictionary containing a different list of collections per variable.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

   the tensor variable representing the result of the series of operations.

##### Raises:


*  <b>`ValueError`</b>: if x has rank less than 2 or if its last dimension is not set.


- - -

### `tf.contrib.layers.layer_norm(*args, **kwargs)` {#layer_norm}

Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.

  "Layer Normalization"

  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

Can be used as a normalizer function for conv2d and fully_connected.

##### Args:


*  <b>`inputs`</b>: a tensor with 2 or more dimensions. The normalization
          occurs over all but the first dimension.
*  <b>`center`</b>: If True, subtract `beta`. If False, `beta` is ignored.
*  <b>`scale`</b>: If True, multiply by `gamma`. If False, `gamma` is
    not used. When the next layer is linear (also e.g. `nn.relu`), this can be
    disabled since the scaling can be done by the next layer.
*  <b>`activation_fn`</b>: activation function, default set to None to skip it and
    maintain a linear activation.
*  <b>`reuse`</b>: whether or not the layer and its variables should be reused. To be
    able to reuse the layer scope must be given.
*  <b>`variables_collections`</b>: optional collections for the variables.
*  <b>`outputs_collections`</b>: collections to add the outputs.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for `variable_op_scope`.

##### Returns:

  A `Tensor` representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if rank or last dimension of `inputs` is undefined.


- - -

### `tf.contrib.layers.max_pool2d(*args, **kwargs)` {#max_pool2d}

Adds a 2D Max Pooling op.

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

##### Raises:


*  <b>`ValueError`</b>: If 'kernel_size' is not a 2-D list


- - -

### `tf.contrib.layers.one_hot_encoding(*args, **kwargs)` {#one_hot_encoding}

Transform numeric labels into onehot_labels using `tf.one_hot`.

##### Args:


*  <b>`labels`</b>: [batch_size] target labels.
*  <b>`num_classes`</b>: total number of classes.
*  <b>`on_value`</b>: A scalar defining the on-value.
*  <b>`off_value`</b>: A scalar defining the off-value.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  one hot encoding of the labels.


- - -

### `tf.contrib.layers.repeat(inputs, repetitions, layer, *args, **kwargs)` {#repeat}

Applies the same layer with the same arguments repeatedly.

```python
  y = repeat(x, 3, conv2d, 64, [3, 3], scope='conv1')
  # It is equivalent to:

  x = conv2d(x, 64, [3, 3], scope='conv1/conv1_1')
  x = conv2d(x, 64, [3, 3], scope='conv1/conv1_2')
  y = conv2d(x, 64, [3, 3], scope='conv1/conv1_3')
```

If the `scope` argument is not given in `kwargs`, it is set to
`layer.__name__`, or `layer.func.__name__` (for `functools.partial`
objects). If neither `__name__` nor `func.__name__` is available, the
layers are called with `scope='stack'`.

##### Args:


*  <b>`inputs`</b>: A `Tensor` suitable for layer.
*  <b>`repetitions`</b>: Int, number of repetitions.
*  <b>`layer`</b>: A layer with arguments `(inputs, *args, **kwargs)`
*  <b>`*args`</b>: Extra args for the layer.
*  <b>`**kwargs`</b>: Extra kwargs for the layer.

##### Returns:

  a tensor result of applying the layer, repetitions times.

##### Raises:


*  <b>`ValueError`</b>: if the op is unknown or wrong.


- - -

### `tf.contrib.layers.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=None, combiner=None, default_id=None, name=None, partition_strategy='div')` {#safe_embedding_lookup_sparse}

Lookup embedding results, accounting for invalid IDs and empty features.

The partitioned embedding in `embedding_weights` must all be the same shape
except for the first dimension. The first dimension is allowed to vary as the
vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
may be a `PartitionedVariable` as returned by using `tf.get_variable()` with a
partitioner.

Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
with non-positive weight. For an entry with no features, the embedding vector
for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

The ids and weights may be multi-dimensional. Embeddings are always aggregated
along the last dimension.

##### Args:


*  <b>`embedding_weights`</b>: A list of `P` float tensors or values representing
      partitioned embedding tensors.  Alternatively, a `PartitionedVariable`,
      created by partitioning along dimension 0.  The total unpartitioned
      shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the
      vocab size and `e_1, ..., e_m` are the embedding dimensions.
*  <b>`sparse_ids`</b>: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids. `d_0` is typically batch size.
*  <b>`sparse_weights`</b>: `SparseTensor` of same shape as `sparse_ids`, containing
      float weights corresponding to `sparse_ids`, or `None` if all weights
      are be assumed to be 1.0.
*  <b>`combiner`</b>: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
      the default.
*  <b>`default_id`</b>: The id to use for an entry with no features.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`partition_strategy`</b>: A string specifying the partitioning strategy.
      Currently `"div"` and `"mod"` are supported. Default is `"div"`.


##### Returns:

  Dense tensor of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

##### Raises:


*  <b>`ValueError`</b>: if `embedding_weights` is empty.


- - -

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
    a dictionay containing a different list of collection per variable.
*  <b>`outputs_collections`</b>: collection to add the outputs.
*  <b>`trainable`</b>: whether or not the variables should be trainable or not.
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  A `Tensor` representing the output of the operation.


- - -

### `tf.contrib.layers.stack(inputs, layer, stack_args, **kwargs)` {#stack}

Builds a stack of layers by applying layer repeatedly using stack_args.

`stack` allows you to repeatedly apply the same operation with different
arguments `stack_args[i]`. For each application of the layer, `stack` creates
a new scope appended with an increasing number. For example:

```python
  y = stack(x, fully_connected, [32, 64, 128], scope='fc')
  # It is equivalent to:

  x = fully_connected(x, 32, scope='fc/fc_1')
  x = fully_connected(x, 64, scope='fc/fc_2')
  y = fully_connected(x, 128, scope='fc/fc_3')
```

If the `scope` argument is not given in `kwargs`, it is set to
`layer.__name__`, or `layer.func.__name__` (for `functools.partial`
objects). If neither `__name__` nor `func.__name__` is available, the
layers are called with `scope='stack'`.

##### Args:


*  <b>`inputs`</b>: A `Tensor` suitable for layer.
*  <b>`layer`</b>: A layer with arguments `(inputs, *args, **kwargs)`
*  <b>`stack_args`</b>: A list/tuple of parameters for each call of layer.
*  <b>`**kwargs`</b>: Extra kwargs for the layer.

##### Returns:

  a `Tensor` result of applying the stacked layers.

##### Raises:


*  <b>`ValueError`</b>: if the op is unknown or wrong.


- - -

### `tf.contrib.layers.unit_norm(*args, **kwargs)` {#unit_norm}

Normalizes the given input across the specified dimension to unit length.

Note that the rank of `input` must be known.

##### Args:


*  <b>`inputs`</b>: A `Tensor` of arbitrary size.
*  <b>`dim`</b>: The dimension along which the input is normalized.
*  <b>`epsilon`</b>: A small value to add to the inputs to avoid dividing by zero.
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  The normalized `Tensor`.

##### Raises:


*  <b>`ValueError`</b>: If dim is smaller than the number of dimensions in 'inputs'.



Aliases for fully_connected which set a default activation function are
available: `relu`, `relu6` and `linear`.

## Regularizers

Regularization can help prevent overfitting. These have the signature
`fn(weights)`. The loss is typically added to
`tf.GraphKeys.REGULARIZATION_LOSSES`.

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


*  <b>`ValueError`</b>: If `regularizer` does not return a scalar output, or if we find
      no weights.


- - -

### `tf.contrib.layers.l1_regularizer(scale, scope=None)` {#l1_regularizer}

Returns a function that can be used to apply L1 regularization to weights.

L1 regularization encourages sparsity.

##### Args:


*  <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
*  <b>`scope`</b>: An optional scope name.

##### Returns:

  A function with signature `l1(weights)` that apply L1 regularization.

##### Raises:


*  <b>`ValueError`</b>: If scale is negative or if scale is not a float.


- - -

### `tf.contrib.layers.l2_regularizer(scale, scope=None)` {#l2_regularizer}

Returns a function that can be used to apply L2 regularization to weights.

Small values of L2 can help prevent overfitting the training data.

##### Args:


*  <b>`scale`</b>: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
*  <b>`scope`</b>: An optional scope name.

##### Returns:

  A function with signature `l2(weights)` that applies L2 regularization.

##### Raises:


*  <b>`ValueError`</b>: If scale is negative or if scale is not a float.


- - -

### `tf.contrib.layers.sum_regularizer(regularizer_list, scope=None)` {#sum_regularizer}

Returns a function that applies the sum of multiple regularizers.

##### Args:


*  <b>`regularizer_list`</b>: A list of regularizers to apply.
*  <b>`scope`</b>: An optional scope name

##### Returns:

  A function with signature `sum_reg(weights)` that applies the
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

```python
  if mode='FAN_IN': # Count only number of input connections.
    n = fan_in
  elif mode='FAN_OUT': # Count only number of output connections.
    n = fan_out
  elif mode='FAN_AVG': # Average number of inputs and output connections.
    n = (fan_in + fan_out)/2.0

    truncated_normal(shape, 0.0, stddev=sqrt(factor / n))
```

* To get [Delving Deep into Rectifiers](
   http://arxiv.org/pdf/1502.01852v1.pdf), use (Default):<br/>
  `factor=2.0 mode='FAN_IN' uniform=False`
* To get [Convolutional Architecture for Fast Feature Embedding](
   http://arxiv.org/abs/1408.5093), use:<br/>
  `factor=1.0 mode='FAN_IN' uniform=True`
* To get [Understanding the difficulty of training deep feedforward neural
  networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf),
  use:<br/>
  `factor=1.0 mode='FAN_AVG' uniform=True.`
* To get `xavier_initializer` use either:<br/>
  `factor=1.0 mode='FAN_AVG' uniform=True`, or<br/>
  `factor=1.0 mode='FAN_AVG' uniform=False`.

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

### `tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, gradient_noise_scale=None, gradient_multipliers=None, clip_gradients=None, learning_rate_decay_fn=None, update_ops=None, variables=None, name=None, summaries=None, colocate_gradients_with_ops=False)` {#optimize_loss}

Given loss and parameters for optimizer, returns a training op.

Various ways of passing optimizers, include:

- string, name of the optimizer like 'SGD', 'Adam', see OPTIMIZER_CLS_NAMES
    for full list. E.g. `optimize_loss(..., optimizer='Adam')`.
- function, takes learning rate `Tensor` as argument and must return
    `Optimizer` instance. E.g. `optimize_loss(...,
    optimizer=lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.5))`.
  Alternatively, if `learning_rate` is `None`, the function takes no
  arguments. E.g. `optimize_loss(..., learning_rate=None,
    optimizer=lambda: tf.train.MomentumOptimizer(0.5, momentum=0.5))`.
- class, subclass of `Optimizer` that takes only one required argument -
    learning rate, such as AdamOptimizer, AdagradOptimizer.
    E.g. `optimize_loss(..., optimizer=tf.train.AdagradOptimizer)`.
- object, instance of subclass of `Optimizer`.
    E.g., `optimizer_loss(..., optimizer=tf.train.AdagradOptimizer(0.5))`.

##### Args:


*  <b>`loss`</b>: Tensor, 0 dimensional.
*  <b>`global_step`</b>: Tensor, step counter for each update.
*  <b>`learning_rate`</b>: float or Tensor, magnitude of update per each training step.
*  <b>`optimizer`</b>: string, class or optimizer instance, used as trainer.
             string should be name of optimizer, like 'SGD',
               'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
             class should be sub-class of `tf.Optimizer` that implements
               `compute_gradients` and `apply_gradients` functions.
             optimizer instance should be instantiation of `tf.Optimizer`
               sub-class and have `compute_gradients` and `apply_gradients`
               functions.
*  <b>`gradient_noise_scale`</b>: float or None, adds 0-mean normal noise scaled by this
                        value.
*  <b>`gradient_multipliers`</b>: dict of variables or variable names to floats.
                        If present, gradients for specified
                        variables will be multiplied by given constant.
*  <b>`clip_gradients`</b>: float or `None`, clips gradients by this value.
*  <b>`learning_rate_decay_fn`</b>: function, takes `learning_rate` and `global_step`
                          `Tensor`s, returns `Tensor`.
                          Can be used to implement any learning rate decay
                          functions.
                          For example: `tf.train.exponential_decay`.
*  <b>`update_ops`</b>: list of update `Operation`s to execute at each step. If `None`,
              uses elements of UPDATE_OPS collection. The order of execution
              between `update_ops` and `loss` is non-deterministic.
*  <b>`variables`</b>: list of variables to optimize or
             `None` to use all trainable variables.
*  <b>`name`</b>: The name for this operation is used to scope operations and summaries.
*  <b>`summaries`</b>: List of internal quantities to visualize on tensorboard. If not
             set only the loss and the learning rate will be reported. The
             complete list is in OPTIMIZER_SUMMARIES.
*  <b>`colocate_gradients_with_ops`</b>: If True, try colocating gradients with the
                               corresponding op.

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


