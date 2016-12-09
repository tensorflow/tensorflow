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

