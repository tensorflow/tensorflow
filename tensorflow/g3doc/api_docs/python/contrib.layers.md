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


*  <b>`inputs`</b>: A 4-D tensor of shape `[batch_size, height, width, channels]` if
    `data_format` is `NHWC`, and `[batch_size, channels, height, width]` if
    `data_format` is `NCHW`.
*  <b>`kernel_size`</b>: A list of length 2: [kernel_height, kernel_width] of the
    pooling kernel over which the op is computed. Can be an int if both
    values are the same.
*  <b>`stride`</b>: A list of length 2: [stride_height, stride_width].
    Can be an int if both strides are the same. Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: The padding method, either 'VALID' or 'SAME'.
*  <b>`data_format`</b>: A string. `NHWC` (default) and `NCHW` are supported.
*  <b>`outputs_collections`</b>: The collections to which the outputs are added.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  A `Tensor` representing the results of the pooling operation.

##### Raises:


*  <b>`ValueError`</b>: if `data_format` is neither `NHWC` nor `NCHW`.


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

One can set updates_collections=None to force the updates in place, but that
can have speed penalty, specially in distributed settings.

##### Args:


*  <b>`inputs`</b>: a tensor with 2 or more dimensions, where the first dimension has
    `batch_size`. The normalization is over all but the last dimension if
    `data_format` is `NHWC` and the second dimension if `data_format` is
    `NCHW`.
*  <b>`decay`</b>: decay for the moving average. Reasonable values for `decay` are close
    to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc. Lower
    `decay` value (recommend trying `decay`=0.9) if model experiences reasonably
    good training performance but poor validation and/or test performance.
*  <b>`center`</b>: If True, subtract `beta`. If False, `beta` is ignored.
*  <b>`scale`</b>: If True, multiply by `gamma`. If False, `gamma` is
    not used. When the next layer is linear (also e.g. `nn.relu`), this can be
    disabled since the scaling can be done by the next layer.
*  <b>`epsilon`</b>: small float added to variance to avoid dividing by zero.
*  <b>`activation_fn`</b>: activation function, default set to None to skip it and
    maintain a linear activation.
*  <b>`param_initializers`</b>: optional initializers for beta, gamma, moving mean and
    moving variance.
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
*  <b>`fused`</b>: Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
*  <b>`data_format`</b>: A string. `NHWC` (default) and `NCHW` are supported.
*  <b>`scope`</b>: Optional scope for `variable_scope`.

##### Returns:

  A `Tensor` representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if `batch_weights` is not None and `fused` is True.
*  <b>`ValueError`</b>: if `data_format` is neither `NHWC` nor `NCHW`.
*  <b>`ValueError`</b>: if the rank of `inputs` is undefined.
*  <b>`ValueError`</b>: if rank or channels dimension of `inputs` is undefined.


- - -

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


*  <b>`inputs`</b>: A 4-D `Tensor` of type `float` and shape
    `[batch, height, width, in_channels]` for `NHWC` data format or
    `[batch, in_channels, height, width]` for `NCHW` data format.
*  <b>`num_outputs`</b>: integer, the number of output filters.
*  <b>`kernel_size`</b>: a list of length 2 holding the [kernel_height, kernel_width] of
    of the filters. Can be an int if both values are the same.
*  <b>`stride`</b>: a list of length 2: [stride_height, stride_width].
    Can be an int if both strides are the same.  Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: one of 'VALID' or 'SAME'.
*  <b>`data_format`</b>: A string. `NHWC` (default) and `NCHW` are supported.
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
*  <b>`ValueError`</b>: if `data_format` is neither `NHWC` nor `NCHW`.
*  <b>`ValueError`</b>: if `C` dimension of `inputs` is None.


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

   The tensor variable representing the result of the series of operations.

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
*  <b>`scope`</b>: Optional scope for `variable_scope`.

##### Returns:

  A `Tensor` representing the output of the operation.

##### Raises:


*  <b>`ValueError`</b>: if rank or last dimension of `inputs` is undefined.


- - -

### `tf.contrib.layers.max_pool2d(*args, **kwargs)` {#max_pool2d}

Adds a 2D Max Pooling op.

It is assumed that the pooling is done per image but not in batch or channels.

##### Args:


*  <b>`inputs`</b>: A 4-D tensor of shape `[batch_size, height, width, channels]` if
    `data_format` is `NHWC`, and `[batch_size, channels, height, width]` if
    `data_format` is `NCHW`.
*  <b>`kernel_size`</b>: A list of length 2: [kernel_height, kernel_width] of the
    pooling kernel over which the op is computed. Can be an int if both
    values are the same.
*  <b>`stride`</b>: A list of length 2: [stride_height, stride_width].
    Can be an int if both strides are the same. Note that presently
    both strides must have the same value.
*  <b>`padding`</b>: The padding method, either 'VALID' or 'SAME'.
*  <b>`data_format`</b>: A string. `NHWC` (default) and `NCHW` are supported.
*  <b>`outputs_collections`</b>: The collections to which the outputs are added.
*  <b>`scope`</b>: Optional scope for name_scope.

##### Returns:

  A `Tensor` representing the results of the pooling operation.

##### Raises:


*  <b>`ValueError`</b>: if `data_format` is neither `NHWC` nor `NCHW`.
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

### `tf.contrib.layers.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=None, combiner=None, default_id=None, name=None, partition_strategy='div', max_norm=None)` {#safe_embedding_lookup_sparse}

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
*  <b>`max_norm`</b>: If not None, all embeddings are l2-normalized to max_norm before
      combining.


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

### `tf.stack(values, axis=0, name='stack')` {#stack}

Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.

Packs the list of tensors in `values` into a tensor with rank one higher than
each tensor in `values`, by packing them along the `axis` dimension.
Given a list of length `N` of tensors of shape `(A, B, C)`;

if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
Etc.

For example:

```prettyprint
# 'x' is [1, 4]
# 'y' is [2, 5]
# 'z' is [3, 6]
stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
stack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
```

This is the opposite of unstack.  The numpy equivalent is

    tf.stack([x, y, z]) = np.asarray([x, y, z])

##### Args:


*  <b>`values`</b>: A list of `Tensor` objects with the same shape and type.
*  <b>`axis`</b>: An `int`. The axis to stack along. Defaults to the first dimension.
    Supports negative indexes.
*  <b>`name`</b>: A name for this operation (optional).

##### Returns:


*  <b>`output`</b>: A stacked `Tensor` with the same type as `values`.

##### Raises:


*  <b>`ValueError`</b>: If `axis` is out of the range [-(R+1), R+1).


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


*  <b>`loss`</b>: Scalar `Tensor`.
*  <b>`global_step`</b>: Scalar int `Tensor`, step counter for each update. If not
               supplied, it will be fetched from the default graph (see
               `tf.contrib.framework.get_global_step` for details). If it's
               not been created, no step will be incremented with each weight
               update. `learning_rate_decay_fn` requires `global_step`.
*  <b>`learning_rate`</b>: float or `Tensor`, magnitude of update per each training
                 step. Can be `None`.
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
*  <b>`clip_gradients`</b>: float, callable or `None`. If float, is provided, a global
    clipping is applied to prevent the norm of the gradient to exceed this
    value. Alternatively, a callable can be provided e.g.: adaptive_clipping.
    This callable takes a `list` of `(gradients, variables)` `tuple`s and
    returns the same thing with the gradients modified.
*  <b>`learning_rate_decay_fn`</b>: function, takes `learning_rate` and `global_step`
                          `Tensor`s, returns `Tensor`.
                          Can be used to implement any learning rate decay
                          functions.
                          For example: `tf.train.exponential_decay`.
                          Ignored if `learning_rate` is not supplied.
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


*  <b>`ValueError`</b>: if:
      * `loss` is an invalid type or shape.
      * `global_step` is an invalid type or shape.
      * `learning_rate` is an invalid type or value.
      * `optimizer` is wrong type.
      * `clip_gradients` is not float or callable.
      * `learning_rate` and `learning_rate_decay_fn` are supplied, but no
        `global_step` is available.



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



## Feature columns

Feature columns provide a mechanism to map data to a model.

- - -

### `tf.contrib.layers.bucketized_column(source_column, boundaries)` {#bucketized_column}

Creates a _BucketizedColumn for discretizing dense input.

##### Args:


*  <b>`source_column`</b>: A _RealValuedColumn defining dense column.
*  <b>`boundaries`</b>: A list of floats specifying the boundaries. It has to be sorted.

##### Returns:

  A _BucketizedColumn.

##### Raises:


*  <b>`ValueError`</b>: if 'boundaries' is empty or not sorted.


- - -

### `tf.contrib.layers.check_feature_columns(feature_columns)` {#check_feature_columns}

Checks the validity of the set of FeatureColumns.

##### Args:


*  <b>`feature_columns`</b>: A set of instances or subclasses of FeatureColumn.

##### Raises:


*  <b>`ValueError`</b>: If there are duplicate feature column keys.


- - -

### `tf.contrib.layers.create_feature_spec_for_parsing(feature_columns)` {#create_feature_spec_for_parsing}

Helper that prepares features config from input feature_columns.

The returned feature config can be used as arg 'features' in tf.parse_example.

Typical usage example:

```python
# Define features and transformations
feature_a = sparse_column_with_vocabulary_file(...)
feature_b = real_valued_column(...)
feature_c_bucketized = bucketized_column(real_valued_column("feature_c"), ...)
feature_a_x_feature_c = crossed_column(
  columns=[feature_a, feature_c_bucketized], ...)

feature_columns = set(
  [feature_b, feature_c_bucketized, feature_a_x_feature_c])
batch_examples = tf.parse_example(
    serialized=serialized_examples,
    features=create_feature_spec_for_parsing(feature_columns))
```

For the above example, create_feature_spec_for_parsing would return the dict:
{
  "feature_a": parsing_ops.VarLenFeature(tf.string),
  "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
  "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
}

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns. All items
    should be instances of classes derived from _FeatureColumn, unless
    feature_columns is a dict -- in which case, this should be true of all
    values in the dict.

##### Returns:

  A dict mapping feature keys to FixedLenFeature or VarLenFeature values.


- - -

### `tf.contrib.layers.crossed_column(columns, hash_bucket_size, combiner=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, hash_key=None)` {#crossed_column}

Creates a _CrossedColumn for performing feature crosses.

##### Args:


*  <b>`columns`</b>: An iterable of _FeatureColumn. Items can be an instance of
    _SparseColumn, _CrossedColumn, or _BucketizedColumn.
*  <b>`hash_bucket_size`</b>: An int that is > 1. The number of buckets.
*  <b>`combiner`</b>: A combiner string, supports sum, mean, sqrtn.
*  <b>`ckpt_to_load_from`</b>: (Optional). String representing checkpoint name/pattern
    to restore the column weights. Required if `tensor_name_in_ckpt` is not
    None.
*  <b>`tensor_name_in_ckpt`</b>: (Optional). Name of the `Tensor` in the provided
    checkpoint from which to restore the column weights. Required if
    `ckpt_to_load_from` is not None.
*  <b>`hash_key`</b>: Specify the hash_key that will be used by the `FingerprintCat64`
    function to combine the crosses fingerprints on SparseFeatureCrossOp
    (optional).

##### Returns:

  A _CrossedColumn.

##### Raises:


*  <b>`TypeError`</b>: if any item in columns is not an instance of _SparseColumn,
    _CrossedColumn, or _BucketizedColumn, or
    hash_bucket_size is not an int.
*  <b>`ValueError`</b>: if hash_bucket_size is not > 1 or
    len(columns) is not > 1.


- - -

### `tf.contrib.layers.embedding_column(sparse_id_column, dimension, combiner=None, initializer=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None)` {#embedding_column}

Creates an `_EmbeddingColumn` for feeding sparse data into a DNN.

##### Args:


*  <b>`sparse_id_column`</b>: A `_SparseColumn` which is created by for example
    `sparse_column_with_*` or crossed_column functions. Note that `combiner`
    defined in `sparse_id_column` is ignored.
*  <b>`dimension`</b>: An integer specifying dimension of the embedding.
*  <b>`combiner`</b>: A string specifying how to reduce if there are multiple entries
    in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
    of this can be considered an example level normalization on the column:
      * "sum": do not normalize
      * "mean": do l1 normalization
      * "sqrtn": do l2 normalization
    For more information: `tf.embedding_lookup_sparse`.
*  <b>`initializer`</b>: A variable initializer function to be used in embedding
    variable initialization. If not specified, defaults to
    `tf.truncated_normal_initializer` with mean 0.0 and standard deviation
    1/sqrt(sparse_id_column.length).
*  <b>`ckpt_to_load_from`</b>: (Optional). String representing checkpoint name/pattern
    to restore the column weights. Required if `tensor_name_in_ckpt` is not
    None.
*  <b>`tensor_name_in_ckpt`</b>: (Optional). Name of the `Tensor` in the provided
    checkpoint from which to restore the column weights. Required if
    `ckpt_to_load_from` is not None.

##### Returns:

  An `_EmbeddingColumn`.


- - -

### `tf.contrib.layers.scattered_embedding_column(column_name, size, dimension, hash_key, combiner=None, initializer=None)` {#scattered_embedding_column}

Creates an embedding column of a sparse feature using parameter hashing.

The i-th embedding component of a value v is found by retrieving an
embedding weight whose index is a fingerprint of the pair (v,i).

An embedding column with sparse_column_with_hash_bucket such as
  embedding_column(
      sparse_column_with_hash_bucket(column_name, bucket_size),
      dimension)

could be replaced by
  scattered_embedding_column(
      column_name, size=bucket_size * dimension, dimension=dimension,
      hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY)

for the same number of embedding parameters and hopefully reduced impact of
collisions with a cost of slowing down training.

##### Args:


*  <b>`column_name`</b>: A string defining sparse column name.
*  <b>`size`</b>: An integer specifying the number of parameters in the embedding layer.
*  <b>`dimension`</b>: An integer specifying dimension of the embedding.
*  <b>`hash_key`</b>: Specify the hash_key that will be used by the `FingerprintCat64`
    function to combine the crosses fingerprints on SparseFeatureCrossOp.
*  <b>`combiner`</b>: A string specifying how to reduce if there are multiple entries
    in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
    of this can be thought as example level normalizations on the column:
      * "sum": do not normalize features in the column
      * "mean": do l1 normalization on features in the column
      * "sqrtn": do l2 normalization on features in the column
    For more information: `tf.embedding_lookup_sparse`.
*  <b>`initializer`</b>: A variable initializer function to be used in embedding
    variable initialization. If not specified, defaults to
    `tf.truncated_normal_initializer` with mean 0 and standard deviation 0.1.

##### Returns:

  A _ScatteredEmbeddingColumn.

##### Raises:


*  <b>`ValueError`</b>: if dimension or size is not a positive integer; or if combiner
    is not supported.


- - -

### `tf.contrib.layers.input_from_feature_columns(columns_to_tensors, feature_columns, weight_collections=None, trainable=True, scope=None)` {#input_from_feature_columns}

A tf.contrib.layer style input layer builder based on FeatureColumns.

Generally a single example in training data is described with feature columns.
At the first layer of the model, this column oriented data should be converted
to a single tensor. Each feature column needs a different kind of operation
during this conversion. For example sparse features need a totally different
handling than continuous features.

Example:

```python
  # Building model for training
  columns_to_tensor = tf.parse_example(...)
  first_layer = input_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=feature_columns)
  second_layer = fully_connected(inputs=first_layer, ...)
  ...
```

where feature_columns can be defined as follows:

```python
  sparse_feature = sparse_column_with_hash_bucket(
      column_name="sparse_col", ...)
  sparse_feature_emb = embedding_column(sparse_id_column=sparse_feature, ...)
  real_valued_feature = real_valued_column(...)
  real_valued_buckets = bucketized_column(
      source_column=real_valued_feature, ...)

  feature_columns=[sparse_feature_emb, real_valued_buckets]
```

##### Args:


*  <b>`columns_to_tensors`</b>: A mapping from feature column to tensors. 'string' key
    means a base feature (not-transformed). It can have FeatureColumn as a
    key too. That means that FeatureColumn is already transformed by input
    pipeline. For example, `inflow` may have handled transformations.
*  <b>`feature_columns`</b>: A set containing all the feature columns. All items in the
    set should be instances of classes derived by FeatureColumn.
*  <b>`weight_collections`</b>: List of graph collections to which weights are added.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  A Tensor which can be consumed by hidden layers in the neural network.

##### Raises:


*  <b>`ValueError`</b>: if FeatureColumn cannot be consumed by a neural network.


- - -

### `tf.contrib.layers.joint_weighted_sum_from_feature_columns(columns_to_tensors, feature_columns, num_outputs, weight_collections=None, trainable=True, scope=None)` {#joint_weighted_sum_from_feature_columns}

A restricted linear prediction builder based on FeatureColumns.

As long as all feature columns are unweighted sparse columns this computes the
prediction of a linear model which stores all weights in a single variable.

##### Args:


*  <b>`columns_to_tensors`</b>: A mapping from feature column to tensors. 'string' key
    means a base feature (not-transformed). It can have FeatureColumn as a
    key too. That means that FeatureColumn is already transformed by input
    pipeline. For example, `inflow` may have handled transformations.
*  <b>`feature_columns`</b>: A set containing all the feature columns. All items in the
    set should be instances of classes derived from FeatureColumn.
*  <b>`num_outputs`</b>: An integer specifying number of outputs. Default value is 1.
*  <b>`weight_collections`</b>: List of graph collections to which weights are added.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  A tuple containing:

    * A Tensor which represents predictions of a linear model.
    * A list of Variables storing the weights.
    * A Variable which is used for bias.

##### Raises:


*  <b>`ValueError`</b>: if FeatureColumn cannot be used for linear predictions.


- - -

### `tf.contrib.layers.make_place_holder_tensors_for_base_features(feature_columns)` {#make_place_holder_tensors_for_base_features}

Returns placeholder tensors for inference.

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns. All items
    should be instances of classes derived from _FeatureColumn.

##### Returns:

  A dict mapping feature keys to SparseTensors (sparse columns) or
  placeholder Tensors (dense columns).


- - -

### `tf.contrib.layers.one_hot_column(sparse_id_column)` {#one_hot_column}

Creates an `_OneHotColumn` for a one-hot or multi-hot repr in a DNN.

##### Args:


*  <b>`sparse_id_column`</b>: A _SparseColumn which is created by
      `sparse_column_with_*`
      or crossed_column functions. Note that `combiner` defined in
      `sparse_id_column` is ignored.

##### Returns:

  An _OneHotColumn.


- - -

### `tf.contrib.layers.parse_feature_columns_from_examples(serialized, feature_columns, name=None, example_names=None)` {#parse_feature_columns_from_examples}

Parses tf.Examples to extract tensors for given feature_columns.

This is a wrapper of 'tf.parse_example'.

Example:

```python
columns_to_tensor = parse_feature_columns_from_examples(
    serialized=my_data,
    feature_columns=my_features)

# Where my_features are:
# Define features and transformations
sparse_feature_a = sparse_column_with_keys(
    column_name="sparse_feature_a", keys=["AB", "CD", ...])

embedding_feature_a = embedding_column(
    sparse_id_column=sparse_feature_a, dimension=3, combiner="sum")

sparse_feature_b = sparse_column_with_hash_bucket(
    column_name="sparse_feature_b", hash_bucket_size=1000)

embedding_feature_b = embedding_column(
    sparse_id_column=sparse_feature_b, dimension=16, combiner="sum")

crossed_feature_a_x_b = crossed_column(
    columns=[sparse_feature_a, sparse_feature_b], hash_bucket_size=10000)

real_feature = real_valued_column("real_feature")
real_feature_buckets = bucketized_column(
    source_column=real_feature, boundaries=[...])

my_features = [embedding_feature_b, real_feature_buckets, embedding_feature_a]
```

##### Args:


*  <b>`serialized`</b>: A vector (1-D Tensor) of strings, a batch of binary
    serialized `Example` protos.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns. All items
    should be instances of classes derived from _FeatureColumn.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`example_names`</b>: A vector (1-D Tensor) of strings (optional), the names of
    the serialized protos in the batch.

##### Returns:

  A `dict` mapping FeatureColumn to `Tensor` and `SparseTensor` values.


- - -

### `tf.contrib.layers.parse_feature_columns_from_sequence_examples(serialized, context_feature_columns, sequence_feature_columns, name=None, example_name=None)` {#parse_feature_columns_from_sequence_examples}

Parses tf.SequenceExamples to extract tensors for given `FeatureColumn`s.

##### Args:


*  <b>`serialized`</b>: A scalar (0-D Tensor) of type string, a single serialized
    `SequenceExample` proto.
*  <b>`context_feature_columns`</b>: An iterable containing the feature columns for
    context features. All items should be instances of classes derived from
    `_FeatureColumn`. Can be `None`.
*  <b>`sequence_feature_columns`</b>: An iterable containing the feature columns for
    sequence features. All items should be instances of classes derived from
    `_FeatureColumn`. Can be `None`.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`example_name`</b>: A scalar (0-D Tensor) of type string (optional), the names of
    the serialized proto.

##### Returns:

  A tuple consisting of:

*  <b>`context_features`</b>: a dict mapping `FeatureColumns` from
    `context_feature_columns` to their parsed `Tensors`/`SparseTensor`s.
*  <b>`sequence_features`</b>: a dict mapping `FeatureColumns` from
    `sequence_feature_columns` to their parsed `Tensors`/`SparseTensor`s.


- - -

### `tf.contrib.layers.real_valued_column(column_name, dimension=1, default_value=None, dtype=tf.float32, normalizer=None)` {#real_valued_column}

Creates a `_RealValuedColumn` for dense numeric data.

##### Args:


*  <b>`column_name`</b>: A string defining real valued column name.
*  <b>`dimension`</b>: An integer specifying dimension of the real valued column.
    The default is 1. The Tensor representing the _RealValuedColumn
    will have the shape of [batch_size, dimension].
*  <b>`default_value`</b>: A single value compatible with dtype or a list of values
    compatible with dtype which the column takes on during tf.Example parsing
    if data is missing. If None, then tf.parse_example will fail if an example
    does not contain this column. If a single value is provided, the same
    value will be applied as the default value for every dimension. If a
    list of values is provided, the length of the list should be equal to the
    value of `dimension`.
*  <b>`dtype`</b>: defines the type of values. Default value is tf.float32. Must be a
    non-quantized, real integer or floating point type.
*  <b>`normalizer`</b>: If not None, a function that can be used to normalize the value
    of the real valued column after default_value is applied for parsing.
    Normalizer function takes the input tensor as its argument, and returns
    the output tensor. (e.g. lambda x: (x - 3.0) / 4.2).

##### Returns:

  A _RealValuedColumn.

##### Raises:


*  <b>`TypeError`</b>: if dimension is not an int
*  <b>`ValueError`</b>: if dimension is not a positive integer
*  <b>`TypeError`</b>: if default_value is a list but its length is not equal to the
    value of `dimension`.
*  <b>`TypeError`</b>: if default_value is not compatible with dtype.
*  <b>`ValueError`</b>: if dtype is not convertable to tf.float32.


- - -

### `tf.contrib.layers.shared_embedding_columns(sparse_id_columns, dimension, combiner=None, shared_embedding_name=None, initializer=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None)` {#shared_embedding_columns}

Creates a list of `_EmbeddingColumn` sharing the same embedding.

##### Args:


*  <b>`sparse_id_columns`</b>: An iterable of `_SparseColumn`, such as those created by
    `sparse_column_with_*` or crossed_column functions. Note that `combiner`
    defined in each sparse_id_column is ignored.
*  <b>`dimension`</b>: An integer specifying dimension of the embedding.
*  <b>`combiner`</b>: A string specifying how to reduce if there are multiple entries
    in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
    of this can be considered an example level normalization on the column:
      * "sum": do not normalize
      * "mean": do l1 normalization
      * "sqrtn": do l2 normalization
    For more information: `tf.embedding_lookup_sparse`.
*  <b>`shared_embedding_name`</b>: (Optional). A string specifying the name of shared
    embedding weights. This will be needed if you want to reference the shared
    embedding separately from the generated `_EmbeddingColumn`.
*  <b>`initializer`</b>: A variable initializer function to be used in embedding
    variable initialization. If not specified, defaults to
    `tf.truncated_normal_initializer` with mean 0.0 and standard deviation
    1/sqrt(sparse_id_columns[0].length).
*  <b>`ckpt_to_load_from`</b>: (Optional). String representing checkpoint name/pattern
    to restore the column weights. Required if `tensor_name_in_ckpt` is not
    None.
*  <b>`tensor_name_in_ckpt`</b>: (Optional). Name of the `Tensor` in the provided
    checkpoint from which to restore the column weights. Required if
    `ckpt_to_load_from` is not None.

##### Returns:

  A tuple of `_EmbeddingColumn` with shared embedding space.

##### Raises:


*  <b>`ValueError`</b>: if sparse_id_columns is empty, or its elements are not
    compatible with each other.
*  <b>`TypeError`</b>: if `sparse_id_columns` is not a sequence or is a string. If at
    least one element of `sparse_id_columns` is not a `SparseTensor`.


- - -

### `tf.contrib.layers.sparse_column_with_hash_bucket(column_name, hash_bucket_size, combiner=None, dtype=tf.string)` {#sparse_column_with_hash_bucket}

Creates a _SparseColumn with hashed bucket configuration.

Use this when your sparse features are in string or integer format, but you
don't have a vocab file that maps each value to an integer ID.
output_id = Hash(input_feature_string) % bucket_size

##### Args:


*  <b>`column_name`</b>: A string defining sparse column name.
*  <b>`hash_bucket_size`</b>: An int that is > 1. The number of buckets.
*  <b>`combiner`</b>: A string specifying how to reduce if the sparse column is
    multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
    "sum" the default:
      * "sum": do not normalize features in the column
      * "mean": do l1 normalization on features in the column
      * "sqrtn": do l2 normalization on features in the column
    For more information: `tf.embedding_lookup_sparse`.
*  <b>`dtype`</b>: The type of features. Only string and integer types are supported.

##### Returns:

  A _SparseColumn with hashed bucket configuration

##### Raises:


*  <b>`ValueError`</b>: hash_bucket_size is not greater than 2.
*  <b>`ValueError`</b>: dtype is neither string nor integer.


- - -

### `tf.contrib.layers.sparse_column_with_integerized_feature(column_name, bucket_size, combiner=None, dtype=tf.int64)` {#sparse_column_with_integerized_feature}

Creates an integerized _SparseColumn.

Use this when your features are already pre-integerized into int64 IDs.
output_id = input_feature

##### Args:


*  <b>`column_name`</b>: A string defining sparse column name.
*  <b>`bucket_size`</b>: An int that is > 1. The number of buckets. It should be bigger
    than maximum feature. In other words features in this column should be an
    int64 in range [0, bucket_size)
*  <b>`combiner`</b>: A string specifying how to reduce if the sparse column is
    multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
    "sum" the default:
      * "sum": do not normalize features in the column
      * "mean": do l1 normalization on features in the column
      * "sqrtn": do l2 normalization on features in the column
    For more information: `tf.embedding_lookup_sparse`.
*  <b>`dtype`</b>: Type of features. It should be an integer type. Default value is
    dtypes.int64.

##### Returns:

  An integerized _SparseColumn definition.

##### Raises:


*  <b>`ValueError`</b>: bucket_size is not greater than 1.
*  <b>`ValueError`</b>: dtype is not integer.


- - -

### `tf.contrib.layers.sparse_column_with_keys(column_name, keys, default_value=-1, combiner=None)` {#sparse_column_with_keys}

Creates a _SparseColumn with keys.

Look up logic is as follows:
lookup_id = index_of_feature_in_keys if feature in keys else default_value

##### Args:


*  <b>`column_name`</b>: A string defining sparse column name.
*  <b>`keys`</b>: a string list defining vocabulary.
*  <b>`default_value`</b>: The value to use for out-of-vocabulary feature values.
    Default is -1.
*  <b>`combiner`</b>: A string specifying how to reduce if the sparse column is
    multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
    "sum" the default:
      * "sum": do not normalize features in the column
      * "mean": do l1 normalization on features in the column
      * "sqrtn": do l2 normalization on features in the column
    For more information: `tf.embedding_lookup_sparse`.

##### Returns:

  A _SparseColumnKeys with keys configuration.


- - -

### `tf.contrib.layers.weighted_sparse_column(sparse_id_column, weight_column_name, dtype=tf.float32)` {#weighted_sparse_column}

Creates a _SparseColumn by combining sparse_id_column with a weight column.

Example:

  ```python
  sparse_feature = sparse_column_with_hash_bucket(column_name="sparse_col",
                                                  hash_bucket_size=1000)
  weighted_feature = weighted_sparse_column(sparse_id_column=sparse_feature,
                                            weight_column_name="weights_col")
  ```

  This configuration assumes that input dictionary of model contains the
  following two items:
    * (key="sparse_col", value=sparse_tensor) where sparse_tensor is
      a SparseTensor.
    * (key="weights_col", value=weights_tensor) where weights_tensor
      is a SparseTensor.
   Following are assumed to be true:
     * sparse_tensor.indices = weights_tensor.indices
     * sparse_tensor.shape = weights_tensor.shape

##### Args:


*  <b>`sparse_id_column`</b>: A `_SparseColumn` which is created by
    `sparse_column_with_*` functions.
*  <b>`weight_column_name`</b>: A string defining a sparse column name which represents
    weight or value of the corresponding sparse id feature.
*  <b>`dtype`</b>: Type of weights, such as `tf.float32`

##### Returns:

  A _WeightedSparseColumn composed of two sparse features: one represents id,
  the other represents weight (value) of the id feature in that example.

##### Raises:


*  <b>`ValueError`</b>: if dtype is not convertible to float.


- - -

### `tf.contrib.layers.weighted_sum_from_feature_columns(columns_to_tensors, feature_columns, num_outputs, weight_collections=None, trainable=True, scope=None)` {#weighted_sum_from_feature_columns}

A tf.contrib.layer style linear prediction builder based on FeatureColumns.

Generally a single example in training data is described with feature columns.
This function generates weighted sum for each num_outputs. Weighted sum refers
to logits in classification problems. It refers to prediction itself for
linear regression problems.

Example:

  ```
  # Building model for training
  feature_columns = (
      real_valued_column("my_feature1"),
      ...
  )
  columns_to_tensor = tf.parse_example(...)
  logits = weighted_sum_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=feature_columns,
      num_outputs=1)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
  ```

##### Args:


*  <b>`columns_to_tensors`</b>: A mapping from feature column to tensors. 'string' key
    means a base feature (not-transformed). It can have FeatureColumn as a
    key too. That means that FeatureColumn is already transformed by input
    pipeline. For example, `inflow` may have handled transformations.
*  <b>`feature_columns`</b>: A set containing all the feature columns. All items in the
    set should be instances of classes derived from FeatureColumn.
*  <b>`num_outputs`</b>: An integer specifying number of outputs. Default value is 1.
*  <b>`weight_collections`</b>: List of graph collections to which weights are added.
*  <b>`trainable`</b>: If `True` also add variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
*  <b>`scope`</b>: Optional scope for variable_scope.

##### Returns:

  A tuple containing:

    * A Tensor which represents predictions of a linear model.
    * A dictionary which maps feature_column to corresponding Variable.
    * A Variable which is used for bias.

##### Raises:


*  <b>`ValueError`</b>: if FeatureColumn cannot be used for linear predictions.


