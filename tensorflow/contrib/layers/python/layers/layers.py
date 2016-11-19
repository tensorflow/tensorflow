# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-short-docstring-punctuation
"""Higher level ops for building layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

# TODO(b/28426988): Replace legacy_* fns migrated from slim.
# TODO(b/28426988): Remove legacy_* when all uses have migrated to new API.
__all__ = ['avg_pool2d',
           'batch_norm',
           'bias_add',
           'conv2d',
           'conv2d_in_plane',
           'conv2d_transpose',
           'convolution',
           'convolution2d',
           'convolution2d_in_plane',
           'convolution2d_transpose',
           'dropout',
           'flatten',
           'fully_connected',
           'layer_norm',
           'linear',
           'pool',
           'max_pool2d',
           'one_hot_encoding',
           'relu',
           'relu6',
           'repeat',
           'separable_conv2d',
           'separable_convolution2d',
           'softmax',
           'stack',
           'unit_norm',
           'legacy_fully_connected',
           'legacy_linear',
           'legacy_relu']

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


@add_arg_scope
def avg_pool2d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NHWC,
               outputs_collections=None,
               scope=None):
  """Adds a 2D average pooling op.

  It is assumed that the pooling is done per image but not in batch or channels.

  Args:
    inputs: A 4-D tensor of shape `[batch_size, height, width, channels]` if
      `data_format` is `NHWC`, and `[batch_size, channels, height, width]` if
      `data_format` is `NCHW`.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: A list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: The padding method, either 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.

  Returns:
    A `Tensor` representing the results of the pooling operation.

  Raises:
    ValueError: if `data_format` is neither `NHWC` nor `NCHW`.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with ops.name_scope(scope, 'AvgPool2D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
    stride_h, stride_w = utils.two_element_tuple(stride)
    if data_format == DATA_FORMAT_NHWC:
      ksize = [1, kernel_h, kernel_w, 1]
      strides = [1, stride_h, stride_w, 1]
    else:
      ksize = [1, 1, kernel_h, kernel_w]
      strides = [1, 1, stride_h, stride_w]
    outputs = nn.avg_pool(inputs,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          data_format=data_format)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


def _fused_batch_norm(
    inputs,
    decay=0.999,
    center=True,
    scale=False,
    epsilon=0.001,
    activation_fn=None,
    param_initializers=None,
    updates_collections=ops.GraphKeys.UPDATE_OPS,
    is_training=True,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    data_format=DATA_FORMAT_NHWC,
    scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

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

  Args:
    inputs: a tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: decay for the moving average. Reasonable values for `decay` are close 
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc. Lower 
      `decay` value (recommend trying `decay`=0.9) if model experiences reasonably 
      good training performance but poor validation and/or test performance.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: optional initializers for beta, gamma, moving mean and
      moving variance.
    updates_collections: collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: if the rank of `inputs` is undefined.
    ValueError: if the rank of `inputs` is neither 2 or 4.
    ValueError: if rank or `C` dimension of `inputs` is undefined.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    original_shape = inputs.get_shape()
    original_rank = original_shape.ndims
    if original_rank is None:
      raise ValueError('Inputs %s has undefined rank' % inputs.name)
    elif original_rank not in [2, 4]:
      raise ValueError('Inputs %s has unsupported rank. \
          Expected 2 or 4 but got %d' % (inputs.name, original_rank))
    if original_rank == 2:
      channels = inputs.get_shape()[-1].value
      if channels is None:
        raise ValueError('`C` dimension must be known but is None')
      new_shape = [-1, channels, 1, 1] if data_format == DATA_FORMAT_NCHW else \
          [-1, 1, 1, channels]
      inputs = array_ops.reshape(inputs, new_shape)
    inputs_shape = inputs.get_shape()
    dtype = inputs.dtype.base_dtype
    if data_format == DATA_FORMAT_NHWC:
      params_shape = inputs_shape[-1:]
    else:
      params_shape = inputs_shape[1:2]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    trainable_beta = trainable and center
    beta_collections = utils.get_variable_collections(variables_collections,
                                                      'beta')
    if not param_initializers:
      param_initializers = {}
    beta_initializer = param_initializers.get('beta',
                                              init_ops.zeros_initializer)
    beta = variables.model_variable(
        'beta',
        shape=params_shape,
        dtype=dtype,
        initializer=beta_initializer,
        collections=beta_collections,
        trainable=trainable_beta)
    trainable_gamma = trainable and scale
    gamma_collections = utils.get_variable_collections(variables_collections,
                                                       'gamma')
    gamma_initializer = param_initializers.get('gamma',
                                               init_ops.ones_initializer())
    gamma = variables.model_variable(
        'gamma',
        shape=params_shape,
        dtype=dtype,
        initializer=gamma_initializer,
        collections=gamma_collections,
        trainable=trainable_gamma)

    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections.
    moving_mean_collections = utils.get_variable_collections(
        variables_collections, 'moving_mean')
    moving_mean_initializer = param_initializers.get('moving_mean',
                                                     init_ops.zeros_initializer)
    moving_mean = variables.model_variable(
        'moving_mean',
        shape=params_shape,
        dtype=dtype,
        initializer=moving_mean_initializer,
        trainable=False,
        collections=moving_mean_collections)
    moving_variance_collections = utils.get_variable_collections(
        variables_collections, 'moving_variance')
    moving_variance_initializer = param_initializers.get(
        'moving_variance', init_ops.ones_initializer())
    moving_variance = variables.model_variable(
        'moving_variance',
        shape=params_shape,
        dtype=dtype,
        initializer=moving_variance_initializer,
        trainable=False,
        collections=moving_variance_collections)

    def _fused_batch_norm_training():
      return nn.fused_batch_norm(
          inputs, gamma, beta, epsilon=epsilon, data_format=data_format)
    def _fused_batch_norm_inference():
      return nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=moving_mean,
          variance=moving_variance,
          epsilon=epsilon,
          is_training=False,
          data_format=data_format)
    outputs, mean, variance = utils.smart_cond(is_training,
                                               _fused_batch_norm_training,
                                               _fused_batch_norm_inference)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `need_updates` will be true.
    is_training_value = utils.constant_value(is_training)
    need_updates = is_training_value is None or is_training_value
    if need_updates:
      if updates_collections is None:
        _no_updates = lambda: outputs
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=False)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies(
              [update_moving_mean, update_moving_variance]):
            return array_ops.identity(outputs)
        outputs = utils.smart_cond(is_training, _force_updates, _no_updates)
      else:
        moving_vars_fn = lambda: (moving_mean, moving_variance)
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=False)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          return update_moving_mean, update_moving_variance
        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)

    outputs.set_shape(inputs_shape)
    if original_shape.ndims == 2:
      outputs = array_ops.reshape(outputs, original_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


@add_arg_scope
def batch_norm(
    inputs,
    decay=0.999,
    center=True,
    scale=False,
    epsilon=0.001,
    activation_fn=None,
    param_initializers=None,
    updates_collections=ops.GraphKeys.UPDATE_OPS,
    is_training=True,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    batch_weights=None,
    fused=False,
    data_format=DATA_FORMAT_NHWC,
    scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

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

  Args:
    inputs: a tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: decay for the moving average. Reasonable values for `decay` are close 
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc. Lower 
      `decay` value (recommend trying `decay`=0.9) if model experiences reasonably 
      good training performance but poor validation and/or test performance.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: optional initializers for beta, gamma, moving mean and
      moving variance.
    updates_collections: collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused:  Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if `batch_weights` is not None and `fused` is True.
    ValueError: if `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: if the rank of `inputs` is undefined.
    ValueError: if rank or channels dimension of `inputs` is undefined.
  """
  if fused:
    if batch_weights is not None:
      raise ValueError('Weighted mean and variance is not currently '
                       'supported for fused batch norm.')
    return _fused_batch_norm(
        inputs,
        decay=decay,
        center=center,
        scale=scale,
        epsilon=epsilon,
        activation_fn=activation_fn,
        param_initializers=param_initializers,
        updates_collections=updates_collections,
        is_training=is_training,
        reuse=reuse,
        variables_collections=variables_collections,
        outputs_collections=outputs_collections,
        trainable=trainable,
        data_format=data_format,
        scope=scope)

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  with variable_scope.variable_scope(scope, 'BatchNorm', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if batch_weights is not None:
      batch_weights = ops.convert_to_tensor(batch_weights)
      inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
      # Reshape batch weight values so they broadcast across inputs.
      nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
      batch_weights = array_ops.reshape(batch_weights, nshape)

    if data_format == DATA_FORMAT_NCHW:
      moments_axes = [0] + list(range(2, inputs_rank))
      params_shape = inputs_shape[1:2]
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      moments_axes = list(range(inputs_rank - 1))
      params_shape = inputs_shape[-1:]
      params_shape_broadcast = None
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if not param_initializers:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer)
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)

    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables.
    partitioner = variable_scope.get_variable_scope().partitioner
    try:
      variable_scope.get_variable_scope().set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer)
      moving_mean = variables.model_variable(
          'moving_mean',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_mean_initializer,
          trainable=False,
          collections=moving_mean_collections)
      moving_variance_collections = utils.get_variable_collections(
          variables_collections, 'moving_variance')
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      moving_variance = variables.model_variable(
          'moving_variance',
          shape=params_shape,
          dtype=dtype,
          initializer=moving_variance_initializer,
          trainable=False,
          collections=moving_variance_collections)
    finally:
      variable_scope.get_variable_scope().set_partitioner(partitioner)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
      # Calculate the moments based on the individual batch.
      if batch_weights is None:
        # Use a copy of moving_mean as a shift to compute more reliable moments.
        shift = math_ops.add(moving_mean, 0)
        if data_format == DATA_FORMAT_NCHW:
          shift = array_ops.reshape(shift, params_shape_broadcast)
          mean, variance = nn.moments(inputs, moments_axes, shift=shift,
                                      keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, variance = nn.moments(inputs, moments_axes, shift=shift)
      else:
        if data_format == DATA_FORMAT_NCHW:
          mean, variance = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, variance = nn.weighted_moments(inputs, moments_axes,
                                               batch_weights)

      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=False)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            return array_ops.identity(mean), array_ops.identity(variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=False)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        vars_fn = lambda: (mean, variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    if data_format == DATA_FORMAT_NCHW:
      mean = array_ops.reshape(mean, params_shape_broadcast)
      variance = array_ops.reshape(variance, params_shape_broadcast)
      beta = array_ops.reshape(beta, params_shape_broadcast)
      if gamma is not None:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Compute batch_normalization.
    outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                     epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


@add_arg_scope
def bias_add(inputs,
             activation_fn=None,
             initializer=init_ops.zeros_initializer,
             regularizer=None,
             reuse=None,
             variables_collections=None,
             outputs_collections=None,
             trainable=True,
             data_format=DATA_FORMAT_NHWC,
             scope=None):
  """Adds a bias to the inputs.

  Can be used as a normalizer function for conv2d and fully_connected.

  Args:
    inputs: a tensor of with at least rank 2 and value for the last dimension,
      e.g. `[batch_size, depth]`, `[None, None, None, depth]`.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    initializer: An initializer for the bias, defaults to 0.
    regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    scope: Optional scope for variable_scope.

  Returns:
    a tensor representing the result of adding biases to the inputs.

  Raises:
    ValueError: if `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: if `data_format` is `NCHW` and rank of `inputs` is not 4.
    ValueError: if the rank of `inputs` is undefined.
    ValueError: if rank or `C` dimension of `inputs` is undefined.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with variable_scope.variable_scope(scope, 'BiasAdd', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Dims of shape must be known but is None')
    elif inputs_rank != 4 and data_format == DATA_FORMAT_NCHW:
      raise ValueError('Data format NCHW only supports 4D Tensor')
    axis = 1 if data_format==DATA_FORMAT_NCHW else -1
    num_features = inputs_shape[axis].value
    if num_features is None:
      raise ValueError('`C` dimension must be known but is None')
    biases_collections = utils.get_variable_collections(variables_collections,
                                                        'biases')
    biases = variables.model_variable('biases',
                                      shape=[num_features,],
                                      dtype=dtype,
                                      initializer=initializer,
                                      regularizer=regularizer,
                                      collections=biases_collections,
                                      trainable=trainable)
    outputs = nn.bias_add(inputs, biases, data_format=data_format)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


# TODO(jbms): change `rate` parameter to `dilation_rate` for consistency with
# underlying op.
@add_arg_scope
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer,
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
  """Adds an N-D convolution followed by an optional batch_norm layer.

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

  Args:
    inputs: a Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    num_outputs: integer, the number of output filters.
    kernel_size: a sequence of N positive integers specifying the spatial
      dimensions of of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: a sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: one of `"VALID"` or `"SAME"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".  For
      N=3, currently the only valid value is "NDHWC".
    rate: a sequence of N positive integers specifying the dilation rate to use
      for a'trous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    a tensor representing the output of the operation.

  Raises:
    ValueError: if `data_format` is invalid.
    ValueError: both 'rate' and `stride` are not uniformly 1.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC']:
    raise ValueError('Invalid data_format: %r' % (data_format,))
  with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    input_rank = inputs.get_shape().ndims
    if input_rank is None:
      raise ValueError('Rank of inputs must be known')
    if input_rank < 3 or input_rank > 5:
      raise ValueError('Rank of inputs is %d, which is not >= 3 and <= 5' %
                       input_rank)
    conv_dims = input_rank - 2
    kernel_size = utils.n_positive_integers(conv_dims, kernel_size)
    stride = utils.n_positive_integers(conv_dims, stride)
    rate = utils.n_positive_integers(conv_dims, rate)

    if data_format is None or data_format.endswith('C'):
      num_input_channels = inputs.get_shape()[input_rank - 1].value
    elif data_format.startswith('NC'):
      num_input_channels = inputs.get_shape()[1].value
    else:
      raise ValueError('Invalid data_format')

    if num_input_channels is None:
      raise ValueError('Number of in_channels must be known.')

    weights_shape = (
        list(kernel_size) + [num_input_channels, num_outputs])
    weights_collections = utils.get_variable_collections(variables_collections,
                                                         'weights')
    weights = variables.model_variable('weights',
                                       shape=weights_shape,
                                       dtype=dtype,
                                       initializer=weights_initializer,
                                       regularizer=weights_regularizer,
                                       collections=weights_collections,
                                       trainable=trainable)
    outputs = nn.convolution(input=inputs,
                             filter=weights,
                             dilation_rate=rate,
                             strides=stride,
                             padding=padding,
                             data_format=data_format)
    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections,
                                          trainable=trainable)
        outputs = nn.bias_add(outputs, biases, data_format=data_format)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


convolution2d = convolution


@add_arg_scope
def convolution2d_in_plane(
    inputs,
    kernel_size,
    stride=1,
    padding='SAME',
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer,
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Performs the same in-plane convolution to each channel independently.

  This is useful for performing various simple channel-independent convolution
  operations such as image gradients:

    image = tf.constant(..., shape=(16, 240, 320, 3))
    vert_gradients = layers.conv2d_in_plane(image,
                                            kernel=[1, -1],
                                            kernel_size=[2, 1])
    horz_gradients = layers.conv2d_in_plane(image,
                                            kernel=[1, -1],
                                            kernel_size=[1, 2])

  Args:
    inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
    kernel_size: a list of length 2 holding the [kernel_height, kernel_width] of
      of the pooling. Can be an int if both values are the same.
    stride: a list of length 2 `[stride_height, stride_width]`.
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: the padding type to use, either 'SAME' or 'VALID'.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.
  """
  with variable_scope.variable_scope(
      scope, 'ConvInPlane', [inputs], reuse=reuse) as sc:
    dtype = inputs.dtype.base_dtype
    kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
    stride_h, stride_w = utils.two_element_tuple(stride)
    num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    weights_shape = [kernel_h, kernel_w, 1, 1]
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')
    weights = variables.model_variable('weights',
                                       shape=weights_shape,
                                       dtype=dtype,
                                       initializer=weights_initializer,
                                       regularizer=weights_regularizer,
                                       collections=weights_collections,
                                       trainable=trainable)
    depthwise_weights = array_ops.tile(weights, [1, 1, num_filters_in, 1])
    outputs = nn.depthwise_conv2d(inputs, depthwise_weights,
                                  [1, stride_h, stride_w, 1], padding)
    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_filters_in,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections,
                                          trainable=trainable)
        outputs = nn.bias_add(outputs, biases)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


@add_arg_scope
def convolution2d_transpose(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    padding='SAME',
    data_format=DATA_FORMAT_NHWC,
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer,
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a convolution2d_transpose with an optional batch normalization layer.

  The function creates a variable called `weights`, representing the
  kernel, that is convolved with the input. If `batch_norm_params` is `None`, a
  second variable called 'biases' is added to the result of the operation.

  Args:
    inputs: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]` for `NHWC` data format or
      `[batch, in_channels, height, width]` for `NCHW` data format.
    num_outputs: integer, the number of output filters.
    kernel_size: a list of length 2 holding the [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: a list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: one of 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.

  Returns:
    a tensor representing the output of the operation.

  Raises:
    ValueError: if 'kernel_size' is not a list of length 2.
    ValueError: if `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: if `C` dimension of `inputs` is None.
  """
  with variable_scope.variable_scope(
      scope, 'Conv2d_transpose', [inputs], reuse=reuse) as sc:
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
      raise ValueError('data_format has to be either NCHW or NHWC.')
    dtype = inputs.dtype.base_dtype
    kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
    stride_h, stride_w = utils.two_element_tuple(stride)
    if data_format == DATA_FORMAT_NCHW:
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      h_axis, w_axis, c_axis = 1, 2, 3
    num_filters_in = inputs.get_shape()[c_axis].value
    if num_filters_in is None:
      raise ValueError('`C` dimension of `inputs` must be known but is None.')
    weights_shape = [kernel_h, kernel_w, num_outputs, num_filters_in]
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')
    weights = variables.model_variable(
        'weights',
        shape=weights_shape,
        dtype=dtype,
        initializer=weights_initializer,
        regularizer=weights_regularizer,
        trainable=trainable,
        collections=weights_collections)

    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    height, width = inputs_shape[h_axis], inputs_shape[w_axis]

    def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
      if isinstance(dim_size, ops.Tensor):
        dim_size = math_ops.mul(dim_size, stride_size)
      elif dim_size is not None:
        dim_size *= stride_size

      if padding == 'VALID' and dim_size is not None:
        dim_size += max(kernel_size - stride_size, 0)
      return dim_size

    # Infer the dynamic output shape:
    out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
    out_width = get_deconv_dim(width, stride_w, kernel_w, padding)

    if data_format == DATA_FORMAT_NHWC:
      output_shape = [batch_size, out_height, out_width, num_outputs]
      strides = [1, stride_h, stride_w, 1]
    else:
      output_shape = [batch_size, num_outputs, out_height, out_width]
      strides = [1, 1, stride_h, stride_w]


    output_shape = array_ops.pack(output_shape)
    outputs = nn.conv2d_transpose(inputs, weights, output_shape,
                                  strides,
                                  padding=padding,
                                  data_format=data_format)

    # Infer the static output shape:
    out_shape = inputs.get_shape().as_list()
    out_shape[c_axis] = num_outputs
    out_shape[h_axis] = get_deconv_dim(out_shape[h_axis], stride_h, kernel_h, padding)
    out_shape[w_axis] = get_deconv_dim(out_shape[w_axis], stride_w, kernel_w, padding)
    outputs.set_shape(out_shape)

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections)
        outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


@add_arg_scope
def dropout(inputs,
            keep_prob=0.5,
            noise_shape=None,
            is_training=True,
            outputs_collections=None,
            scope=None):
  """Returns a dropout op applied to the input.

  With probability `keep_prob`, outputs the input element scaled up by
  `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
  sum is unchanged.

  Args:
    inputs: the tensor to pass to the nn.dropout op.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    is_training: A bool `Tensor` indicating whether or not the model
      is in training mode. If so, dropout is applied and values scaled.
      Otherwise, inputs is returned.
    outputs_collections: collection to add the outputs.
    scope: Optional scope for name_scope.

  Returns:
    a tensor representing the output of the operation.
  """
  with ops.name_scope(scope, 'Dropout', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dropout_fn = lambda: nn.dropout(inputs, keep_prob, noise_shape)
    id_fn = lambda: array_ops.identity(inputs)
    outputs = utils.smart_cond(is_training, dropout_fn, id_fn)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


@add_arg_scope
def flatten(inputs,
            outputs_collections=None,
            scope=None):
  """Flattens the input while maintaining the batch_size.

    Assumes that the first dimension represents the batch.

  Args:
    inputs: a tensor of size [batch_size, ...].
    outputs_collections: collection to add the outputs.
    scope: Optional scope for name_scope.

  Returns:
    a flattened tensor with shape [batch_size, k].
  Raises:
    ValueError: if inputs.shape is wrong.
  """
  with ops.name_scope(scope, 'Flatten', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if (inputs_rank is None) or (inputs_rank < 2):
      raise ValueError('Inputs must have a least 2 dimensions.')
    dims = inputs_shape[1:]
    if not dims.is_fully_defined():
      raise ValueError('Inputs 2nd dimension must be defined.')
    k = dims.num_elements()
    outputs = array_ops.reshape(inputs, [-1, k])
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


def _sparse_inner_flatten(inputs, new_rank):
  """Helper function for `inner_flatten`."""
  outer_dimensions = inputs.shape[:new_rank - 1]
  inner_dimensions = inputs.shape[new_rank - 1:]
  new_shape = array_ops.concat(0, (outer_dimensions,
                                   [math_ops.reduce_prod(inner_dimensions)]))
  flattened = sparse_ops.sparse_reshape(inputs, new_shape)
  return flattened


def _dense_inner_flatten(inputs, new_rank):
  """Helper function for `inner_flatten`."""
  rank_assertion = check_ops.assert_rank_at_least(
      inputs, new_rank, message='inputs has rank less than new_rank')
  with ops.control_dependencies([rank_assertion]):
    outer_dimensions = array_ops.slice(
        array_ops.shape(inputs), [0], [new_rank - 1])
    new_shape = array_ops.concat(0, (outer_dimensions, [-1]))
    reshaped = array_ops.reshape(inputs, new_shape)

  # if `new_rank` is an integer, try to calculate new shape.
  if isinstance(new_rank, six.integer_types):
    static_shape = inputs.get_shape()
    if static_shape is not None and static_shape.dims is not None:
      static_shape = static_shape.as_list()
      static_outer_dims = static_shape[:new_rank - 1]
      static_inner_dims = static_shape[new_rank - 1:]
      flattened_dimension = 1
      for inner_dim in static_inner_dims:
        if inner_dim is None:
          flattened_dimension = None
          break
        flattened_dimension *= inner_dim
      reshaped.set_shape(static_outer_dims + [flattened_dimension])
  return reshaped


@add_arg_scope
def _inner_flatten(inputs, new_rank, output_collections=None, scope=None):
  """Flattens inner dimensions of `inputs`, returns a Tensor with `new_rank`.

  For example:
  '''
      x = tf.random_uniform(shape=[1, 2, 3, 4, 5, 6])
      y = _inner_flatten(x, 4)
      assert y.get_shape().as_list() == [1, 2, 3, (4 * 5 * 6)]
  '''
  This layer will fail at run time if `new_rank` is greater than the current
  rank of `inputs`.

  Args:
    inputs: a `Tensor` or `SparseTensor`.
    new_rank: the desired rank of the returned `Tensor` or `SparseTensor`.
    output_collections: collection to which the outputs will be added.
    scope: optional scope for `name_scope`.
  Returns:
    A `Tensor` or `SparseTensor` conataining the same values as `inputs`, but
    with innermost dimensions flattened to obtain rank `new_rank`.

  Raises:
    TypeError: `inputs` is not a `Tensor` or `SparseTensor`.
  """
  with ops.name_scope(scope, 'InnerFlatten', [inputs, new_rank]) as sc:
    if isinstance(inputs, sparse_tensor.SparseTensor):
      flattened = _sparse_inner_flatten(inputs, new_rank)
    else:
      inputs = ops.convert_to_tensor(inputs)
      flattened = _dense_inner_flatten(inputs, new_rank)
  return utils.collect_named_outputs(output_collections, sc, flattened)


@add_arg_scope
def fully_connected(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer,
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
  """Adds a fully connected layer.

  `fully_connected` creates a variable called `weights`, representing a fully
  connected weight matrix, which is multiplied by the `inputs` to produce a
  `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
  `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
  None and a `biases_initializer` is provided then a `biases` variable would be
  created and added the hidden units. Finally, if `activation_fn` is not `None`,
  it is applied to the hidden units as well.

  Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
  prior to the initial matrix multiply by `weights`.

  Args:
    inputs: A tensor of with at least rank 2 and value for the last dimension,
      i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
    num_outputs: Integer or long, the number of output units in the layer.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collections per variable.
    outputs_collections: collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_scope.

  Returns:
     the tensor variable representing the result of the series of operations.

  Raises:
    ValueError: if x has rank less than 2 or if its last dimension is not set.
  """
  if not (isinstance(num_outputs, six.integer_types)):
    raise ValueError('num_outputs should be int or long, got %s.', num_outputs)
  with variable_scope.variable_scope(scope, 'fully_connected', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    dtype = inputs.dtype.base_dtype
    inputs_shape = inputs.get_shape()
    num_input_units = utils.last_dimension(inputs_shape, min_rank=2)

    static_shape = inputs_shape.as_list()
    static_shape[-1] = num_outputs

    out_shape = array_ops.unpack(array_ops.shape(inputs))
    out_shape[-1] = num_outputs

    weights_shape = [num_input_units, num_outputs]
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')
    weights = variables.model_variable('weights',
                                       shape=weights_shape,
                                       dtype=dtype,
                                       initializer=weights_initializer,
                                       regularizer=weights_regularizer,
                                       collections=weights_collections,
                                       trainable=trainable)
    if len(static_shape) > 2:
      # Reshape inputs
      inputs = array_ops.reshape(inputs, [-1, num_input_units])
    outputs = standard_ops.matmul(inputs, weights)
    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections,
                                          trainable=trainable)
        outputs = nn.bias_add(outputs, biases)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    if len(static_shape) > 2:
      # Reshape back outputs
      outputs = array_ops.reshape(outputs, array_ops.pack(out_shape))
      outputs.set_shape(static_shape)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


@add_arg_scope
def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
  """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.

    "Layer Normalization"

    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  Can be used as a normalizer function for conv2d and fully_connected.

  Args:
    inputs: a tensor with 2 or more dimensions. The normalization
            occurs over all but the first dimension.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """
  with variable_scope.variable_scope(scope, 'LayerNorm', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = list(range(1, inputs_rank))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma = variables.model_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.ones_initializer(),
          collections=gamma_collections,
          trainable=trainable)
    # Calculate the moments on the last axis (layer activations).
    mean, variance = nn.moments(inputs, axis, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1E-12
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope,
                                       outputs)


@add_arg_scope
def max_pool2d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NHWC,
               outputs_collections=None,
               scope=None):
  """Adds a 2D Max Pooling op.

  It is assumed that the pooling is done per image but not in batch or channels.

  Args:
    inputs: A 4-D tensor of shape `[batch_size, height, width, channels]` if
      `data_format` is `NHWC`, and `[batch_size, channels, height, width]` if
      `data_format` is `NCHW`.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: A list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: The padding method, either 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.

  Returns:
    A `Tensor` representing the results of the pooling operation.

  Raises:
    ValueError: if `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If 'kernel_size' is not a 2-D list
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with ops.name_scope(scope, 'MaxPool2D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
    stride_h, stride_w = utils.two_element_tuple(stride)
    if data_format == DATA_FORMAT_NHWC:
      ksize = [1, kernel_h, kernel_w, 1]
      strides = [1, stride_h, stride_w, 1]
    else:
      ksize = [1, 1, kernel_h, kernel_w]
      strides = [1, 1, stride_h, stride_w]
    outputs = nn.max_pool(inputs,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          data_format=data_format)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


@add_arg_scope
def pool(inputs,
         kernel_size,
         pooling_type,
         padding='VALID',
         data_format=None,
         dilation_rate=1,
         stride=1,
         outputs_collections=None,
         scope=None):
  # pylint: disable=line-too-long
  """Adds a pooling op.


  Args:
    inputs: Tensor of rank N+2, of shape
      `[batch_size] + input_spatial_shape + [num_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, num_channels] + input_spatial_shape` if data_format starts
      with "NC".  Pooling happens over the spatial dimensions only.
    kernel_size: Sequence of N ints >= 1.  Can also be a single integer to
      specify the same value for all spatial dimensions.
    pooling_type: Specifies pooling operation, must be "AVG" or "MAX".
    padding: The padding algorithm, must be "SAME" or "VALID".
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".  For
      N=3, currently the only valid value is "NDHWC".
    dilation_rate: Optional.  Dilation rate.  Sequence of N ints >= 1.  Defaults
      to [1]*N.  Can also be a single integer to specify the same value for all
      spatial dimensions.  If any value of dilation_rate is > 1, then all values
      of stride must be 1.
    stride: Optional.  Sequence of N ints >= 1.  Defaults to [1]*N.  Can also be
      a single integer to specify the same value for all spatial dimensions.  If
      any value of stride is > 1, then all values of dilation_rate must be 1.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.

  Returns:
    A `Tensor` representing the results of the pooling operation.

  Raises:
    ValueError: if arguments are invalid.

  """
  # pylint: enable=line-too-long
  with ops.name_scope(scope, '%s_pool' %
                      (pooling_type.lower()), [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims
    if input_rank is None:
      raise ValueError('Rank of inputs must be known')
    if input_rank < 3:
      raise ValueError('Rank of inputs must be >= 3')
    num_spatial_dims = input_rank - 2
    output = nn.pool(
        input=inputs,
        window_shape=utils.n_positive_integers(num_spatial_dims, kernel_size),
        pooling_type=pooling_type,
        padding=padding,
        data_format=data_format,
        dilation_rate=utils.n_positive_integers(num_spatial_dims,
                                                dilation_rate),
        strides=utils.n_positive_integers(num_spatial_dims, stride),
        name=sc)
    return utils.collect_named_outputs(outputs_collections, sc, output)


@add_arg_scope
def one_hot_encoding(labels,
                     num_classes,
                     on_value=1.0,
                     off_value=0.0,
                     outputs_collections=None,
                     scope=None):
  """Transform numeric labels into onehot_labels using `tf.one_hot`.

  Args:
    labels: [batch_size] target labels.
    num_classes: total number of classes.
    on_value: A scalar defining the on-value.
    off_value: A scalar defining the off-value.
    outputs_collections: collection to add the outputs.
    scope: Optional scope for name_scope.

  Returns:
    one hot encoding of the labels.
  """
  with ops.name_scope(scope, 'OneHotEncoding', [labels, num_classes]) as sc:
    labels = ops.convert_to_tensor(labels)
    if labels.dtype == dtypes.int32:
      labels = standard_ops.to_int64(labels)
    outputs = standard_ops.one_hot(labels,
                                   num_classes,
                                   on_value=on_value,
                                   off_value=off_value)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


def _apply_activation(y, activation_fn, output_collections):
  if activation_fn is not None:
    y = activation_fn(y)
  ops.add_to_collections(list(output_collections or []) +
                         [ops.GraphKeys.ACTIVATIONS], y)
  return y


def repeat(inputs, repetitions, layer, *args, **kwargs):
  """Applies the same layer with the same arguments repeatedly.

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

  Args:
    inputs: A `Tensor` suitable for layer.
    repetitions: Int, number of repetitions.
    layer: A layer with arguments `(inputs, *args, **kwargs)`
    *args: Extra args for the layer.
    **kwargs: Extra kwargs for the layer.

  Returns:
    a tensor result of applying the layer, repetitions times.
  Raises:
    ValueError: if the op is unknown or wrong.
  """
  scope = kwargs.pop('scope', None)
  with variable_scope.variable_scope(scope, 'Repeat', [inputs]):
    inputs = ops.convert_to_tensor(inputs)
    if scope is None:
      if hasattr(layer, '__name__'):
        scope = layer.__name__
      elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
        scope = layer.func.__name__  # In case layer is a functools.partial.
      else:
        scope = 'repeat'
    outputs = inputs
    for i in range(repetitions):
      kwargs['scope'] = scope + '_' + str(i+1)
      outputs = layer(outputs, *args, **kwargs)
    return outputs


@add_arg_scope
def separable_convolution2d(
    inputs,
    num_outputs,
    kernel_size,
    depth_multiplier,
    stride=1,
    padding='SAME',
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer,
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a depth-separable 2D convolution with optional batch_norm layer.

  This op first performs a depthwise convolution that acts separately on
  channels, creating a variable called `depthwise_weights`. If `num_outputs`
  is not None, it adds a pointwise convolution that mixes channels, creating a
  variable called `pointwise_weights`. Then, if `batch_norm_params` is None,
  it adds bias to the result, creating a variable called 'biases', otherwise
  it adds a batch normalization layer. It finally applies an activation function
  to produce the end result.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_outputs: the number of pointwise convolution output filters. If is
      None, then we skip the pointwise convolution stage.
    kernel_size: a list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    depth_multiplier: the number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: a list of length 2: [stride_height, stride_width], specifying the
      depthwise convolution stride. Can be an int if both strides are the same.
    padding: one of 'VALID' or 'SAME'.
    activation_fn: activation function, set to None to skip it and maintain
      a linear activation.
    normalizer_fn: normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional list of collections for all the variables or
      a dictionay containing a different list of collection per variable.
    outputs_collections: collection to add the outputs.
    trainable: whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.

  Returns:
    A `Tensor` representing the output of the operation.
  """
  with variable_scope.variable_scope(
      scope, 'SeparableConv2d', [inputs], reuse=reuse) as sc:
    dtype = inputs.dtype.base_dtype
    kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
    stride_h, stride_w = utils.two_element_tuple(stride)
    num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    weights_collections = utils.get_variable_collections(
        variables_collections, 'weights')

    depthwise_shape = [kernel_h, kernel_w,
                       num_filters_in, depth_multiplier]
    depthwise_weights = variables.model_variable(
        'depthwise_weights',
        shape=depthwise_shape,
        dtype=dtype,
        initializer=weights_initializer,
        regularizer=weights_regularizer,
        trainable=trainable,
        collections=weights_collections)
    strides = [1, stride_h, stride_w, 1]
    if num_outputs is not None:
      # Full separable convolution: Depthwise followed by pointwise convolution.
      pointwise_shape = [1, 1, depth_multiplier * num_filters_in,
                         num_outputs]
      pointwise_weights = variables.model_variable(
          'pointwise_weights',
          shape=pointwise_shape,
          dtype=dtype,
          initializer=weights_initializer,
          regularizer=weights_regularizer,
          trainable=trainable,
          collections=weights_collections)
      outputs = nn.separable_conv2d(inputs,
                                    depthwise_weights,
                                    pointwise_weights,
                                    strides,
                                    padding)
    else:
      # Depthwise convolution only.
      outputs = nn.depthwise_conv2d(inputs, depthwise_weights, strides, padding)
      num_outputs = depth_multiplier * num_filters_in

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      if biases_initializer is not None:
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          collections=biases_collections)
        outputs = nn.bias_add(outputs, biases)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


@add_arg_scope
def softmax(logits, scope=None):
  """Performs softmax on Nth dimension of N-dimensional logit tensor.

  For two-dimensional logits this reduces to tf.nn.softmax. The N-th dimension
  needs to have a specified number of elements (number of classes).

  Args:
    logits: N-dimensional `Tensor` with logits, where N > 1.
    scope: Optional scope for variable_scope.

  Returns:
    a `Tensor` with same shape and type as logits.
  """
  # TODO(jrru): Add axis argument which defaults to last dimension.
  with variable_scope.variable_scope(scope, 'softmax', [logits]):
    num_logits = utils.last_dimension(logits.get_shape(), min_rank=2)
    logits_2d = array_ops.reshape(logits, [-1, num_logits])
    predictions = nn.softmax(logits_2d)
    predictions = array_ops.reshape(predictions, array_ops.shape(logits))
    predictions.set_shape(logits.get_shape())
    return predictions


def stack(inputs, layer, stack_args, **kwargs):
  """Builds a stack of layers by applying layer repeatedly using stack_args.

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

  Args:
    inputs: A `Tensor` suitable for layer.
    layer: A layer with arguments `(inputs, *args, **kwargs)`
    stack_args: A list/tuple of parameters for each call of layer.
    **kwargs: Extra kwargs for the layer.

  Returns:
    a `Tensor` result of applying the stacked layers.

  Raises:
    ValueError: if the op is unknown or wrong.
  """
  scope = kwargs.pop('scope', None)
  if not isinstance(stack_args, (list, tuple)):
    raise ValueError('stack_args need to be a list or tuple')
  with variable_scope.variable_scope(scope, 'Stack', [inputs]):
    inputs = ops.convert_to_tensor(inputs)
    if scope is None:
      if hasattr(layer, '__name__'):
        scope = layer.__name__
      elif hasattr(layer, 'func') and hasattr(layer.func, '__name__'):
        scope = layer.func.__name__  # In case layer is a functools.partial.
      else:
        scope = 'stack'
    outputs = inputs
    for i in range(len(stack_args)):
      kwargs['scope'] = scope + '_' + str(i+1)
      layer_args = stack_args[i]
      if not isinstance(layer_args, (list, tuple)):
        layer_args = [layer_args]
      outputs = layer(outputs, *layer_args, **kwargs)
    return outputs


@add_arg_scope
def unit_norm(inputs, dim, epsilon=1e-7, scope=None):
  """Normalizes the given input across the specified dimension to unit length.

  Note that the rank of `input` must be known.

  Args:
    inputs: A `Tensor` of arbitrary size.
    dim: The dimension along which the input is normalized.
    epsilon: A small value to add to the inputs to avoid dividing by zero.
    scope: Optional scope for variable_scope.

  Returns:
    The normalized `Tensor`.

  Raises:
    ValueError: If dim is smaller than the number of dimensions in 'inputs'.
  """
  with variable_scope.variable_scope(scope, 'UnitNorm', [inputs]):
    if not inputs.get_shape():
      raise ValueError('The input rank must be known.')
    input_rank = len(inputs.get_shape().as_list())
    if dim < 0 or dim >= input_rank:
      raise ValueError(
          'dim must be positive but smaller than the input rank.')

    lengths = math_ops.sqrt(epsilon + math_ops.reduce_sum(
        math_ops.square(inputs), dim, True))
    multiples = []
    if dim > 0:
      multiples.append(array_ops.ones([dim], dtypes.int32))
    multiples.append(array_ops.slice(array_ops.shape(inputs), [dim], [1]))
    if dim < (input_rank - 1):
      multiples.append(array_ops.ones([input_rank - 1 - dim], dtypes.int32))
    multiples = array_ops.concat(0, multiples)
    return math_ops.div(inputs, array_ops.tile(lengths, multiples))


def legacy_fully_connected(x,
                           num_output_units,
                           activation_fn=None,
                           weight_init=initializers.xavier_initializer(),
                           bias_init=init_ops.zeros_initializer,
                           name=None,
                           weight_collections=(ops.GraphKeys.WEIGHTS,),
                           bias_collections=(ops.GraphKeys.BIASES,),
                           output_collections=(ops.GraphKeys.ACTIVATIONS,),
                           trainable=True,
                           weight_regularizer=None,
                           bias_regularizer=None):
  # pylint: disable=anomalous-backslash-in-string
  r"""Adds the parameters for a fully connected layer and returns the output.

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
  initializers (`weight_init` and `bias_init`) and in which collections to place
  the created variables (`weight_collections` and `bias_collections`; note that
  the variables are always added to the `VARIABLES` collection). The output of
  the layer can be placed in custom collections using `output_collections`.
  The collections arguments default to `WEIGHTS`, `BIASES` and `ACTIVATIONS`,
  respectively.

  A per layer regularization can be specified by setting `weight_regularizer`
  and `bias_regularizer`, which are applied to the weights and biases
  respectively, and whose output is added to the `REGULARIZATION_LOSSES`
  collection.

  Args:
    x: The input `Tensor`.
    num_output_units: The size of the output.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    weight_init: An optional weight initialization, defaults to
      `xavier_initializer`.
    bias_init: An initializer for the bias, defaults to 0. Set to `None` in
      order to disable bias.
    name: The name for this operation is used to name operations and to find
      variables. If specified it must be unique for this scope, otherwise a
      unique name starting with "fully_connected" will be created.  See
      `tf.variable_scope` for details.
    weight_collections: List of graph collections to which weights are added.
    bias_collections: List of graph collections to which biases are added.
    output_collections: List of graph collections to which outputs are added.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    weight_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for weights.
    bias_regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`. Used for biases.

  Returns:
    The output of the fully connected layer.

  Raises:
    ValueError: if x has rank less than 2 or if its last dimension is not set.
  """
  with variable_scope.variable_scope(name, 'fully_connected', [x]):
    x = ops.convert_to_tensor(x)
    dims = x.get_shape().dims
    if dims is None:
      raise ValueError('dims of x must be known but is None')
    if len(dims) < 2:
      raise ValueError('rank of x must be at least 2 not: %d' % len(dims))
    num_input_units = dims[-1].value
    if num_input_units is None:
      raise ValueError('last dimension of x must be known but is None')
    dtype = x.dtype.base_dtype

    weight_collections = set(list(weight_collections or []) +
                             [ops.GraphKeys.GLOBAL_VARIABLES])
    w = variable_scope.get_variable('weights',
                                    shape=[num_input_units, num_output_units],
                                    dtype=dtype,
                                    initializer=weight_init,
                                    collections=weight_collections,
                                    regularizer=weight_regularizer,
                                    trainable=trainable)
    x_2_dim = x if len(dims) <= 2 else array_ops.reshape(x,
                                                         [-1, num_input_units])
    y = standard_ops.matmul(x_2_dim, w)

    if bias_init is not None:
      bias_collections = set(list(bias_collections or []) +
                             [ops.GraphKeys.GLOBAL_VARIABLES])
      b = variable_scope.get_variable('bias',
                                      shape=[num_output_units],
                                      dtype=dtype,
                                      initializer=bias_init,
                                      collections=bias_collections,
                                      regularizer=bias_regularizer,
                                      trainable=trainable)

      y = nn.bias_add(y, b)

    if len(dims) > 2:
      out_shape = array_ops.unpack(array_ops.shape(x))
      out_shape[-1] = num_output_units

      y = array_ops.reshape(y, array_ops.pack(out_shape))

      static_shape = x.get_shape().as_list()
      static_shape[-1] = num_output_units
      y.set_shape(static_shape)

    return _apply_activation(y, activation_fn, output_collections)


# TODO(eiderm): Verify and fix autocomplete in colab (also relu6).
# Simple aliases which remove the activation_fn parameter.
legacy_relu = functools.partial(legacy_fully_connected, activation_fn=nn.relu)
legacy_linear = functools.partial(legacy_fully_connected, activation_fn=None)
relu = functools.partial(fully_connected, activation_fn=nn.relu)
relu6 = functools.partial(fully_connected, activation_fn=nn.relu6)
linear = functools.partial(fully_connected, activation_fn=None)

# Simple alias.
conv2d = convolution2d
conv2d_transpose = convolution2d_transpose
conv2d_in_plane = convolution2d_in_plane
separable_conv2d = separable_convolution2d
