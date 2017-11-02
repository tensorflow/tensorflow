# -*- coding: utf-8 -*-
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
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages
from tensorflow.python.layers.maxout import maxout

# TODO(b/28426988): Replace legacy_* fns migrated from slim.
# TODO(b/28426988): Remove legacy_* when all uses have migrated to new API.
__all__ = ['avg_pool2d',
           'avg_pool3d',
           'batch_norm',
           'bias_add',
           'conv2d',
           'conv3d',
           'conv2d_in_plane',
           'conv2d_transpose',
           'conv3d_transpose',
           'convolution',
           'convolution2d',
           'convolution2d_in_plane',
           'convolution2d_transpose',
           'convolution3d',
           'convolution3d_transpose',
           'dropout',
           'elu',
           'flatten',
           'fully_connected',
           'GDN',
           'gdn',
           'layer_norm',
           'linear',
           'pool',
           'max_pool2d',
           'max_pool3d',
           'one_hot_encoding',
           'relu',
           'relu6',
           'repeat',
           'scale_gradient',
           'separable_conv2d',
           'separable_convolution2d',
           'softmax',
           'spatial_softmax',
           'stack',
           'unit_norm',
           'legacy_fully_connected',
           'legacy_linear',
           'legacy_relu',
           'maxout']

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


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
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with ops.name_scope(scope, 'AvgPool2D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = pooling_layers.AveragePooling2D(pool_size=kernel_size,
                                            strides=stride,
                                            padding=padding,
                                            data_format=df,
                                            _scope=sc)
    outputs = layer.apply(inputs)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


@add_arg_scope
def avg_pool3d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NDHWC,
               outputs_collections=None,
               scope=None):
  """Adds a 3D average pooling op.

  It is assumed that the pooling is done per image but not in batch or channels.

  Args:
    inputs: A 5-D tensor of shape `[batch_size, depth, height, width, channels]`
      if `data_format` is `NDHWC`, and `[batch_size, channels, depth, height,
      width]` if `data_format` is `NCDHW`.
    kernel_size: A list of length 3: [kernel_depth, kernel_height, kernel_width]
      of the pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: A list of length 3: [stride_depth, stride_height, stride_width].
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: The padding method, either 'VALID' or 'SAME'.
    data_format: A string. `NDHWC` (default) and `NCDHW` are supported.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.

  Returns:
    A `Tensor` representing the results of the pooling operation.

  Raises:
    ValueError: If `data_format` is neither `NDHWC` nor `NCDHW`.
  """
  if data_format not in (DATA_FORMAT_NCDHW, DATA_FORMAT_NDHWC):
    raise ValueError('data_format has to be either NCDHW or NDHWC.')
  with ops.name_scope(scope, 'AvgPool3D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = pooling_layers.AveragePooling3D(pool_size=kernel_size,
                                            strides=stride,
                                            padding=padding,
                                            data_format=df,
                                            _scope=sc)
    outputs = layer.apply(inputs)
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
    zero_debias_moving_mean=False,
    scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"

    Sergey Ioffe, Christian Szegedy

  Can be used as a normalizer function for conv2d and fully_connected.

  Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:

  ```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  ```

  One can set updates_collections=None to force the updates in place, but that
  can have a speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance.
    center: If True, add offset of `beta` to normalized tensor.  If False,
      `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If the rank of `inputs` is neither 2 or 4.
    ValueError: If rank or `C` dimension of `inputs` is undefined.
  """
  # TODO(reedwm): Add support for fp16 inputs.
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    original_shape = inputs.get_shape()
    original_inputs = inputs
    original_rank = original_shape.ndims
    if original_rank is None:
      raise ValueError('Inputs %s has undefined rank' % inputs.name)
    elif original_rank not in [2, 4]:
      raise ValueError('Inputs %s has unsupported rank.'
                       ' Expected 2 or 4 but got %d' % (
                           inputs.name, original_rank))
    if original_rank == 2:
      channels = inputs.get_shape()[-1].value
      if channels is None:
        raise ValueError('`C` dimension must be known but is None')
      new_shape = [-1, 1, 1, channels]
      if data_format == DATA_FORMAT_NCHW:
        new_shape = [-1, channels, 1, 1]
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
    if center:
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      beta = variables.model_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=beta_initializer,
          collections=beta_collections,
          trainable=trainable_beta)
    else:
      beta = array_ops.constant(0.0, shape=params_shape)

    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      gamma = variables.model_variable(
          'gamma',
          shape=params_shape,
          dtype=dtype,
          initializer=gamma_initializer,
          collections=gamma_collections,
          trainable=trainable)
    else:
      gamma = array_ops.constant(1.0, shape=params_shape)

    # Create moving_mean and moving_variance variables and add them to the
    # appropriate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables (this needs to be handled carefully, as it may break
    # the checkpoint backward compatibility).
    with variable_scope.variable_scope(
        variable_scope.get_variable_scope()) as local_scope:
      local_scope.set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
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
        no_updates = lambda: outputs
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay, zero_debias=False)
          with ops.control_dependencies(
              [update_moving_mean, update_moving_variance]):
            return array_ops.identity(outputs)
        outputs = utils.smart_cond(is_training, _force_updates, no_updates)
      else:
        moving_vars_fn = lambda: (moving_mean, moving_variance)
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
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
      outputs = array_ops.reshape(outputs, array_ops.shape(original_inputs))
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=None,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99,
               adjustment=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"

    Sergey Ioffe, Christian Szegedy

  Can be used as a normalizer function for conv2d and fully_connected.

  Note: when training, the moving_mean and moving_variance need to be updated.
  By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
  need to be added as a dependency to the `train_op`. For example:

  ```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)
  ```

  One can set updates_collections=None to force the updates in place, but that
  can have a speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
      Lower `decay` value (recommend trying `decay`=0.9) if model experiences
      reasonably good training performance but poor validation and/or test
      performance. Try zero_debias_moving_mean=True for improved stability.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    param_regularizers: Optional regularizer for beta and gamma.
    updates_collections: Collections to collect the update ops for computation.
      The updates_ops need to be executed with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    batch_weights: An optional tensor of shape `[batch_size]`,
      containing a frequency weight for each batch item. If present,
      then the batch normalization uses weighted mean and
      variance. (This can be used to correct for bias in training
      example selection.)
    fused: if `True`, use a faster, fused implementation if possible.
      If `None`, use the system recommended implementation.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
      pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
    scope: Optional scope for `variable_scope`.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_decay: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `decay` is still applied
      to get the means and variances for inference.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  if fused is None:
    fused = True

  # Only use _fused_batch_norm if all of the following three
  # conditions are true:
  # (1) fused is set True;
  # (2) it is possible to use (currently it doesn't support batch weights,
  #   renorm, and the case when rank is neither 2 nor 4);
  # (3) it is used with zero_debias_moving_mean, or an input shape of rank 2,
  #   or non-default updates_collections (not implemented in
  #   normalization_layers.BatchNormalization yet); otherwise use the fused
  #   implementation in normalization_layers.BatchNormalization.
  inputs = ops.convert_to_tensor(inputs)
  rank = inputs.get_shape().ndims
  possible_to_fuse = (batch_weights is None and
                      not renorm and
                      rank in [2, 4] and
                      adjustment is None)
  if fused and possible_to_fuse and (
      zero_debias_moving_mean or rank == 2 or
      updates_collections is not ops.GraphKeys.UPDATE_OPS):
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
        zero_debias_moving_mean=zero_debias_moving_mean,
        scope=scope)

  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      scope, 'BatchNorm', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    # Determine whether we can use the core layer class.
    if (batch_weights is None and
        updates_collections is ops.GraphKeys.UPDATE_OPS and
        not zero_debias_moving_mean):
      # Use the core layer class.
      axis = 1 if data_format == DATA_FORMAT_NCHW else -1
      if not param_initializers:
        param_initializers = {}
      beta_initializer = param_initializers.get('beta',
                                                init_ops.zeros_initializer())
      gamma_initializer = param_initializers.get('gamma',
                                                 init_ops.ones_initializer())
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
      moving_variance_initializer = param_initializers.get(
          'moving_variance', init_ops.ones_initializer())
      if not param_regularizers:
        param_regularizers = {}
      beta_regularizer = param_regularizers.get('beta')
      gamma_regularizer = param_regularizers.get('gamma')
      layer = normalization_layers.BatchNormalization(
          axis=axis,
          momentum=decay,
          epsilon=epsilon,
          center=center,
          scale=scale,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer,
          moving_mean_initializer=moving_mean_initializer,
          moving_variance_initializer=moving_variance_initializer,
          beta_regularizer=beta_regularizer,
          gamma_regularizer=gamma_regularizer,
          trainable=trainable,
          renorm=renorm,
          renorm_clipping=renorm_clipping,
          renorm_momentum=renorm_decay,
          adjustment=adjustment,
          name=sc.name,
          _scope=sc,
          _reuse=reuse,
          fused=fused)
      outputs = layer.apply(inputs, training=is_training)

      # Add variables to collections.
      _add_variable_to_collections(
          layer.moving_mean, variables_collections, 'moving_mean')
      _add_variable_to_collections(
          layer.moving_variance, variables_collections, 'moving_variance')
      if layer.beta is not None:
        _add_variable_to_collections(layer.beta, variables_collections, 'beta')
      if layer.gamma is not None:
        _add_variable_to_collections(
            layer.gamma, variables_collections, 'gamma')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

    # Not supported by layer class: batch_weights argument,
    # and custom updates_collections. In that case, use the legacy BN
    # implementation.
    # Custom updates collections are not supported because the update logic
    # is different in this case, in particular w.r.t. "forced updates" and
    # update op reuse.
    if renorm:
      raise ValueError('renorm is not supported with batch_weights, '
                       'updates_collections or zero_debias_moving_mean')
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
                                                init_ops.zeros_initializer())
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
    # appropriate collections. We disable variable partitioning while creating
    # them, because assign_moving_average is not yet supported for partitioned
    # variables (this needs to be handled carefully, as it may break
    # the checkpoint backward compatibility).
    with variable_scope.variable_scope(
        variable_scope.get_variable_scope()) as local_scope:
      local_scope.set_partitioner(None)
      moving_mean_collections = utils.get_variable_collections(
          variables_collections, 'moving_mean')
      moving_mean_initializer = param_initializers.get(
          'moving_mean', init_ops.zeros_initializer())
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

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
      # Calculate the moments based on the individual batch.
      if batch_weights is None:
        if data_format == DATA_FORMAT_NCHW:
          mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
          mean = array_ops.reshape(mean, [-1])
          variance = array_ops.reshape(variance, [-1])
        else:
          mean, variance = nn.moments(inputs, moments_axes)
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
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
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
              moving_mean, mean, decay, zero_debias=zero_debias_moving_mean)
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
      if beta is not None:
        beta = array_ops.reshape(beta, params_shape_broadcast)
      if gamma is not None:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Compute batch_normalization.
    outputs = nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                     epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def bias_add(inputs,
             activation_fn=None,
             initializer=init_ops.zeros_initializer(),
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
    inputs: A tensor of with at least rank 2 and value for the last dimension,
      e.g. `[batch_size, depth]`, `[None, None, None, depth]`.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    initializer: An initializer for the bias, defaults to 0.
    regularizer: A regularizer like the result of
      `l1_regularizer` or `l2_regularizer`.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    data_format: A string. 'NHWC' and 'NCHW' are supported.
    scope: Optional scope for variable_scope.

  Returns:
    A tensor representing the result of adding biases to the inputs.

  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If `data_format` is `NCHW` and rank of `inputs` is not 4.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or `C` dimension of `inputs` is undefined.
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
    axis = 1 if data_format == DATA_FORMAT_NCHW else -1
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
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


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
                biases_initializer=init_ops.zeros_initializer(),
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

  Performs atrous convolution with input stride/dilation rate equal to `rate`
  if a value > 1 for any dimension of `rate` is specified.  In this case
  `stride` values != 1 are not supported.

  Args:
    inputs: A Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    num_outputs: Integer, the number of output filters.
    kernel_size: A sequence of N positive integers specifying the spatial
      dimensions of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: A sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: One of `"VALID"` or `"SAME"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    rate: A sequence of N positive integers specifying the dilation rate to use
      for atrous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A tensor representing the output of the operation.

  Raises:
    ValueError: If `data_format` is invalid.
    ValueError: Both 'rate' and `stride` are not uniformly 1.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  layer_variable_getter = _build_variable_getter(
      {'bias': 'biases', 'kernel': 'weights'})

  with variable_scope.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims

    if input_rank == 3:
      layer_class = convolutional_layers.Convolution1D
    elif input_rank == 4:
      layer_class = convolutional_layers.Convolution2D
    elif input_rank == 5:
      layer_class = convolutional_layers.Convolution3D
    else:
      raise ValueError('Convolution not supported for input with rank',
                       input_rank)

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = layer_class(filters=num_outputs,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding=padding,
                        data_format=df,
                        dilation_rate=rate,
                        activation=None,
                        use_bias=not normalizer_fn and biases_initializer,
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer,
                        kernel_regularizer=weights_regularizer,
                        bias_regularizer=biases_regularizer,
                        activity_regularizer=None,
                        trainable=trainable,
                        name=sc.name,
                        dtype=inputs.dtype.base_dtype,
                        _scope=sc,
                        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.use_bias:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

convolution2d = convolution
convolution3d = convolution


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
    biases_initializer=init_ops.zeros_initializer(),
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
    inputs: A 4-D tensor with dimensions [batch_size, height, width, channels].
    kernel_size: A list of length 2 holding the [kernel_height, kernel_width] of
      of the pooling. Can be an int if both values are the same.
    stride: A list of length 2 `[stride_height, stride_width]`.
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: The padding type to use, either 'SAME' or 'VALID'.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
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
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


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
    biases_initializer=init_ops.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a convolution2d_transpose with an optional batch normalization layer.

  The function creates a variable called `weights`, representing the
  kernel, that is convolved with the input. If `normalizer_fn` is `None`, a
  second variable called 'biases' is added to the result of the operation.

  Args:
    inputs: A 4-D `Tensor` of type `float` and shape
      `[batch, height, width, in_channels]` for `NHWC` data format or
      `[batch, in_channels, height, width]` for `NCHW` data format.
    num_outputs: Integer, the number of output filters.
    kernel_size: A list of length 2 holding the [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    stride: A list of length 2: [stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: One of 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: Whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.

  Returns:
    A tensor representing the output of the operation.

  Raises:
    ValueError: If 'kernel_size' is not a list of length 2.
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If `C` dimension of `inputs` is None.
  """
  layer_variable_getter = _build_variable_getter(
      {'bias': 'biases', 'kernel': 'weights'})

  with variable_scope.variable_scope(
      scope, 'Conv2d_transpose', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
      raise ValueError('data_format has to be either NCHW or NHWC.')

    inputs = ops.convert_to_tensor(inputs)

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = convolutional_layers.Convolution2DTranspose(
        filters=num_outputs,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        data_format=df,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.bias is not None:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def convolution3d_transpose(
    inputs,
    num_outputs,
    kernel_size,
    stride=1,
    padding='SAME',
    data_format=DATA_FORMAT_NDHWC,
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a convolution3d_transpose with an optional batch normalization layer.

  The function creates a variable called `weights`, representing the
  kernel, that is convolved with the input. If `batch_norm_params` is `None`, a
  second variable called 'biases' is added to the result of the operation.
  Args:
    inputs: A 5-D `Tensor` of type `float` and shape
      `[batch, depth, height, width, in_channels]` for `NDHWC` data format or
      `[batch, in_channels, depth, height, width]` for `NCDHW` data format.
    num_outputs: Integer, the number of output filters.
    kernel_size: A list of length 3 holding the [kernel_depth, kernel_height,
      kernel_width] of the filters. Can be an int if both values are the same.
    stride: A list of length 3: [stride_depth, stride_height, stride_width].
      Can be an int if both strides are the same.  Note that presently
      both strides must have the same value.
    padding: One of 'VALID' or 'SAME'.
    data_format: A string. `NDHWC` (default) and `NCDHW` are supported.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: Whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.
  Returns:
    A tensor representing the output of the operation.
  Raises:
    ValueError: If 'kernel_size' is not a list of length 3.
    ValueError: If `data_format` is neither `NDHWC` nor `NCDHW`.
    ValueError: If `C` dimension of `inputs` is None.
  """
  layer_variable_getter = _build_variable_getter(
      {'bias': 'biases', 'kernel': 'weights'})

  with variable_scope.variable_scope(
      scope, 'Conv3d_transpose', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    if data_format not in (DATA_FORMAT_NCDHW, DATA_FORMAT_NDHWC):
      raise ValueError('data_format has to be either NCDHW or NDHWC.')

    inputs = ops.convert_to_tensor(inputs)

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = convolutional_layers.Convolution3DTranspose(
        filters=num_outputs,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        data_format=df,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.bias is not None:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


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
    inputs: The tensor to pass to the nn.dropout op.
    keep_prob: A scalar `Tensor` with the same type as x. The probability
      that each element is kept.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    is_training: A bool `Tensor` indicating whether or not the model
      is in training mode. If so, dropout is applied and values scaled.
      Otherwise, inputs is returned.
    outputs_collections: Collection to add the outputs.
    scope: Optional scope for name_scope.

  Returns:
    A tensor representing the output of the operation.
  """
  with variable_scope.variable_scope(
      scope, 'Dropout', [inputs], custom_getter=_model_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)
    layer = core_layers.Dropout(rate=1 - keep_prob,
                                noise_shape=noise_shape,
                                name=sc.name,
                                _scope=sc)
    outputs = layer.apply(inputs, training=is_training)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def flatten(inputs,
            outputs_collections=None,
            scope=None):
  """Flattens the input while maintaining the batch_size.

    Assumes that the first dimension represents the batch.

  Args:
    inputs: A tensor of size [batch_size, ...].
    outputs_collections: Collection to add the outputs.
    scope: Optional scope for name_scope.

  Returns:
    A flattened tensor with shape [batch_size, k].
  Raises:
    ValueError: If inputs rank is unknown or less than 2.
  """
  with ops.name_scope(scope, 'Flatten', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    outputs = core_layers.flatten(inputs)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


def _sparse_inner_flatten(inputs, new_rank):
  """Helper function for `inner_flatten`."""
  inputs_rank = inputs.dense_shape.get_shape().as_list()[0]
  if inputs_rank < new_rank:
    raise ValueError(
        'Inputs has rank less than new_rank. {} must have rank at least'
        ' {}. Received rank {}, shape {}'.format(inputs, new_rank, inputs_rank,
                                                 inputs.get_shape()))

  outer_dimensions = inputs.dense_shape[:new_rank - 1]
  inner_dimensions = inputs.dense_shape[new_rank - 1:]
  new_shape = array_ops.concat((outer_dimensions,
                                [math_ops.reduce_prod(inner_dimensions)]), 0)
  flattened = sparse_ops.sparse_reshape(inputs, new_shape)
  return flattened


def _dense_inner_flatten(inputs, new_rank):
  """Helper function for `inner_flatten`."""
  rank_assertion = check_ops.assert_rank_at_least(
      inputs, new_rank, message='inputs has rank less than new_rank')
  with ops.control_dependencies([rank_assertion]):
    outer_dimensions = array_ops.strided_slice(
        array_ops.shape(inputs), [0], [new_rank - 1])
    new_shape = array_ops.concat((outer_dimensions, [-1]), 0)
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
    inputs: A `Tensor` or `SparseTensor`.
    new_rank: The desired rank of the returned `Tensor` or `SparseTensor`.
    output_collections: Collection to which the outputs will be added.
    scope: Optional scope for `name_scope`.
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


def _model_variable_getter(getter, name, shape=None, dtype=None,
                           initializer=None, regularizer=None, trainable=True,
                           collections=None, caching_device=None,
                           partitioner=None, rename=None, use_resource=None,
                           **_):
  """Getter that uses model_variable for compatibility with core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return variables.model_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, collections=collections, trainable=trainable,
      caching_device=caching_device, partitioner=partitioner,
      custom_getter=getter, use_resource=use_resource)


def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""
  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)
  return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = utils.get_variable_collections(
      collections_set, collections_name) or []
  variables_list = [variable]
  if isinstance(variable, tf_variables.PartitionedVariable):
    variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in ops.get_collection(collection):
        ops.add_to_collection(collection, var)


@add_arg_scope
def fully_connected(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                    biases_initializer=init_ops.zeros_initializer(),
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
    inputs: A tensor of at least rank 2 and static value for the last dimension;
      i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
    num_outputs: Integer or long, the number of output units in the layer.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collections per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_scope.

  Returns:
     The tensor variable representing the result of the series of operations.

  Raises:
    ValueError: If x has rank less than 2 or if its last dimension is not set.
  """
  if not isinstance(num_outputs, six.integer_types):
    raise ValueError(
        'num_outputs should be int or long, got %s.' % (num_outputs,))

  layer_variable_getter = _build_variable_getter({'bias': 'biases',
                                                  'kernel': 'weights'})

  with variable_scope.variable_scope(
      scope, 'fully_connected', [inputs],
      reuse=reuse, custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)
    layer = core_layers.Dense(
        units=num_outputs,
        activation=None,
        use_bias=not normalizer_fn and biases_initializer,
        kernel_initializer=weights_initializer,
        bias_initializer=biases_initializer,
        kernel_regularizer=weights_regularizer,
        bias_regularizer=biases_regularizer,
        activity_regularizer=None,
        trainable=trainable,
        name=sc.name,
        dtype=inputs.dtype.base_dtype,
        _scope=sc,
        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.bias is not None:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    # Apply normalizer function / layer.
    if normalizer_fn is not None:
      if not normalizer_params:
        normalizer_params = {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


class GDN(base.Layer):
  """Generalized divisive normalization layer.

  Based on the papers:

    "Density Modeling of Images using a Generalized Normalization
    Transformation"

    Johannes Ballé, Valero Laparra, Eero P. Simoncelli

    https://arxiv.org/abs/1511.06281

    "End-to-end Optimized Image Compression"

    Johannes Ballé, Valero Laparra, Eero P. Simoncelli

    https://arxiv.org/abs/1611.01704

  Implements an activation function that is essentially a multivariate
  generalization of a particular sigmoid-type function:

  ```
  y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
  ```

  where `i` and `j` run over channels. This implementation never sums across
  spatial dimensions. It is similar to local response normalization, but much
  more flexible, as `beta` and `gamma` are trainable parameters.

  Arguments:
    inverse: If `False` (default), compute GDN response. If `True`, compute IGDN
      response (one step of fixed point iteration to invert GDN; the division
      is replaced by multiplication).
    beta_min: Lower bound for beta, to prevent numerical error from causing
      square root of zero or negative values.
    gamma_init: The gamma matrix will be initialized as the identity matrix
      multiplied with this value. If set to zero, the layer is effectively
      initialized to the identity operation, since beta is initialized as one.
      A good default setting is somewhere between 0 and 0.5.
    reparam_offset: Offset added to the reparameterization of beta and gamma.
      The reparameterization of beta and gamma as their square roots lets the
      training slow down when their values are close to zero, which is desirable
      as small values in the denominator can lead to a situation where gradient
      noise on beta/gamma leads to extreme amounts of noise in the GDN
      activations. However, without the offset, we would get zero gradients if
      any elements of beta or gamma were exactly zero, and thus the training
      could get stuck. To prevent this, we add this small constant. The default
      value was empirically determined as a good starting point. Making it
      bigger potentially leads to more gradient noise on the activations, making
      it too small may lead to numerical precision issues.
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True`, also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require `reuse=True` in such
      cases.

  Properties:
    inverse: Boolean, whether GDN is computed (`True`) or IGDN (`False`).
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    beta: The beta parameter as defined above (1D `Tensor`).
    gamma: The gamma parameter as defined above (2D `Tensor`).
  """

  def __init__(self,
               inverse=False,
               beta_min=1e-6,
               gamma_init=.1,
               reparam_offset=2 ** -18,
               data_format='channels_last',
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(GDN, self).__init__(trainable=trainable, name=name,
                              activity_regularizer=activity_regularizer,
                              **kwargs)
    self.inverse = inverse
    self._beta_min = beta_min
    self._gamma_init = gamma_init
    self._reparam_offset = reparam_offset
    self.data_format = data_format
    self._channel_axis()  # trigger ValueError early
    self.input_spec = base.InputSpec(min_ndim=3, max_ndim=5)

  def _channel_axis(self):
    try:
      return {'channels_first': 1, 'channels_last': -1}[self.data_format]
    except KeyError:
      raise ValueError('Unsupported `data_format` for GDN layer: {}.'.format(
          self.data_format))

  @staticmethod
  def _lower_bound(inputs, bound, name=None):
    """Same as tf.maximum, but with helpful gradient for inputs < bound.

    The gradient is overwritten so that it is passed through if the input is not
    hitting the bound. If it is, only gradients that push `inputs` higher than
    the bound are passed through. No gradients are passed through to the bound.

    Args:
      inputs: input tensor
      bound: lower bound for the input tensor
      name: name for this op

    Returns:
      tf.maximum(inputs, bound)
    """
    with ops.name_scope(name, 'GDNLowerBound', [inputs, bound]) as scope:
      inputs = ops.convert_to_tensor(inputs, name='inputs')
      bound = ops.convert_to_tensor(bound, name='bound')
      with ops.get_default_graph().gradient_override_map(
          {'Maximum': 'GDNLowerBound'}):
        return math_ops.maximum(inputs, bound, name=scope)

  @staticmethod
  def _lower_bound_grad(op, grad):
    """Gradient for `_lower_bound`.

    Args:
      op: the tensorflow op for which to calculate a gradient
      grad: gradient with respect to the output of the op

    Returns:
      gradients with respect to the inputs of the op
    """
    inputs = op.inputs[0]
    bound = op.inputs[1]
    pass_through_if = math_ops.logical_or(inputs >= bound, grad < 0)
    return [math_ops.cast(pass_through_if, grad.dtype) * grad, None]

  def build(self, input_shape):
    channel_axis = self._channel_axis()
    input_shape = tensor_shape.TensorShape(input_shape)
    num_channels = input_shape[channel_axis].value
    if num_channels is None:
      raise ValueError('The channel dimension of the inputs to `GDN` '
                       'must be defined.')
    self._input_rank = input_shape.ndims
    self.input_spec = base.InputSpec(ndim=input_shape.ndims,
                                     axes={channel_axis: num_channels})

    pedestal = array_ops.constant(self._reparam_offset ** 2, dtype=self.dtype)
    beta_bound = array_ops.constant(
        (self._beta_min + self._reparam_offset ** 2) ** .5, dtype=self.dtype)
    gamma_bound = array_ops.constant(self._reparam_offset, dtype=self.dtype)

    def beta_initializer(shape, dtype=None, partition_info=None):
      del partition_info  # unused
      return math_ops.sqrt(array_ops.ones(shape, dtype=dtype) + pedestal)

    def gamma_initializer(shape, dtype=None, partition_info=None):
      del partition_info  # unused
      assert len(shape) == 2
      assert shape[0] == shape[1]
      eye = linalg_ops.eye(shape[0], dtype=dtype)
      return math_ops.sqrt(self._gamma_init * eye + pedestal)

    beta = self.add_variable('reparam_beta',
                             shape=[num_channels],
                             initializer=beta_initializer,
                             dtype=self.dtype,
                             trainable=True)
    beta = self._lower_bound(beta, beta_bound)
    self.beta = math_ops.square(beta) - pedestal

    gamma = self.add_variable('reparam_gamma',
                              shape=[num_channels, num_channels],
                              initializer=gamma_initializer,
                              dtype=self.dtype,
                              trainable=True)
    gamma = self._lower_bound(gamma, gamma_bound)
    self.gamma = math_ops.square(gamma) - pedestal

    self.built = True

  def call(self, inputs):
    inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
    ndim = self._input_rank

    shape = self.gamma.get_shape().as_list()
    gamma = array_ops.reshape(self.gamma, (ndim - 2) * [1] + shape)

    # Compute normalization pool.
    if self.data_format == 'channels_first':
      norm_pool = nn.convolution(math_ops.square(inputs), gamma, 'VALID',
                                 data_format='NC' + 'DHW'[-(ndim - 2):])
      if ndim == 3:
        norm_pool = array_ops.expand_dims(norm_pool, 2)
        norm_pool = nn.bias_add(norm_pool, self.beta, data_format='NCHW')
        norm_pool = array_ops.squeeze(norm_pool, [2])
      elif ndim == 5:
        shape = array_ops.shape(norm_pool)
        norm_pool = array_ops.reshape(norm_pool, shape[:3] + [-1])
        norm_pool = nn.bias_add(norm_pool, self.beta, data_format='NCHW')
        norm_pool = array_ops.reshape(norm_pool, shape)
      else:  # ndim == 4
        norm_pool = nn.bias_add(norm_pool, self.beta, data_format='NCHW')
    else:  # channels_last
      norm_pool = nn.convolution(math_ops.square(inputs), gamma, 'VALID')
      norm_pool = nn.bias_add(norm_pool, self.beta, data_format='NHWC')
    norm_pool = math_ops.sqrt(norm_pool)

    if self.inverse:
      outputs = inputs * norm_pool
    else:
      outputs = inputs / norm_pool
    outputs.set_shape(inputs.get_shape())
    return outputs

  def _compute_output_shape(self, input_shape):
    channel_axis = self._channel_axis()
    input_shape = tensor_shape.TensorShape(input_shape)
    if not 3 <= input_shape.ndim <= 5:
      raise ValueError('`input_shape` must be of rank 3 to 5, inclusive.')
    if input_shape[channel_axis].value is None:
      raise ValueError(
          'The channel dimension of `input_shape` must be defined.')
    return input_shape


ops.RegisterGradient('GDNLowerBound')(GDN._lower_bound_grad)  # pylint:disable=protected-access


def gdn(inputs,
        inverse=False,
        beta_min=1e-6,
        gamma_init=.1,
        reparam_offset=2 ** -18,
        data_format='channels_last',
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None):
  """Functional interface for GDN layer.

  Based on the papers:

    "Density Modeling of Images using a Generalized Normalization
    Transformation"
    Johannes Ballé, Valero Laparra, Eero P. Simoncelli
    https://arxiv.org/abs/1511.06281

    "End-to-end Optimized Image Compression"
    Johannes Ballé, Valero Laparra, Eero P. Simoncelli
    https://arxiv.org/abs/1611.01704

  Implements an activation function that is essentially a multivariate
  generalization of a particular sigmoid-type function:

  ```
  y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
  ```

  where `i` and `j` run over channels. This implementation never sums across
  spatial dimensions. It is similar to local response normalization, but much
  more flexible, as `beta` and `gamma` are trainable parameters.

  Args:
    inputs: Tensor input.
    inverse: If `False` (default), compute GDN response. If `True`, compute IGDN
      response (one step of fixed point iteration to invert GDN; the division
      is replaced by multiplication).
    beta_min: Lower bound for beta, to prevent numerical error from causing
      square root of zero or negative values.
    gamma_init: The gamma matrix will be initialized as the identity matrix
      multiplied with this value. If set to zero, the layer is effectively
      initialized to the identity operation, since beta is initialized as one.
      A good default setting is somewhere between 0 and 0.5.
    reparam_offset: Offset added to the reparameterization of beta and gamma.
      The reparameterization of beta and gamma as their square roots lets the
      training slow down when their values are close to zero, which is desirable
      as small values in the denominator can lead to a situation where gradient
      noise on beta/gamma leads to extreme amounts of noise in the GDN
      activations. However, without the offset, we would get zero gradients if
      any elements of beta or gamma were exactly zero, and thus the training
      could get stuck. To prevent this, we add this small constant. The default
      value was empirically determined as a good starting point. Making it
      bigger potentially leads to more gradient noise on the activations, making
      it too small may lead to numerical precision issues.
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True`, also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require `reuse=True` in such
      cases.
    reuse: Boolean, whether to reuse the weights of a previous layer by the same
      name.

  Returns:
    Output tensor.
  """
  layer = GDN(inverse=inverse,
              beta_min=beta_min,
              gamma_init=gamma_init,
              reparam_offset=reparam_offset,
              data_format=data_format,
              activity_regularizer=activity_regularizer,
              trainable=trainable,
              name=name,
              dtype=inputs.dtype.base_dtype,
              _scope=name,
              _reuse=reuse)
  return layer.apply(inputs)


@add_arg_scope
def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               begin_norm_axis=1,
               begin_params_axis=-1,
               scope=None):
  """Adds a Layer Normalization layer.

  Based on the paper:

    "Layer Normalization"

    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    https://arxiv.org/abs/1607.06450.

  Can be used as a normalizer function for conv2d and fully_connected.

  Given a tensor `inputs` of rank `R`, moments are calculated and normalization
  is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
  if requested, is performed over axes `begin_params_axis .. R - 1`.

  By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
  meaning that normalization is performed over all but the first axis
  (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
  parameters are calculated for the rightmost axis (the `C` if `inputs` is
  `NHWC`).  Scaling and recentering is performed via broadcast of the
  `beta` and `gamma` parameters with the normalized tensor.

  The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
  and this part of the inputs' shape must be fully defined.

  Args:
    inputs: A tensor having rank `R`. The normalization is performed over
      axes `begin_norm_axis ... R - 1` and centering and scaling parameters
      are calculated over `begin_params_axis ... R - 1`.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    begin_norm_axis: The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation, having the same
    shape and dtype as `inputs`.

  Raises:
    ValueError: If the rank of `inputs` is not known at graph build time,
      or if `inputs.shape[begin_params_axis:]` is not fully defined at
      graph build time.
  """
  with variable_scope.variable_scope(scope, 'LayerNorm', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.shape
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
      raise ValueError(
          'begin_params_axis (%d) and begin_norm_axis (%d) '
          'must be < rank(inputs) (%d)'
          % (begin_params_axis, begin_norm_axis, inputs_rank))
    params_shape = inputs_shape[begin_params_axis:]
    if not params_shape.is_fully_defined():
      raise ValueError(
          'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' % (
              inputs.name, begin_params_axis, inputs_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable(
          'beta',
          shape=params_shape,
          dtype=dtype,
          initializer=init_ops.zeros_initializer(),
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
    norm_axes = list(range(begin_norm_axis, inputs_rank))
    mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1e-12
    outputs = nn.batch_normalization(
        inputs, mean, variance, offset=beta, scale=gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


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
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If 'kernel_size' is not a 2-D list
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  with ops.name_scope(scope, 'MaxPool2D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = pooling_layers.MaxPooling2D(pool_size=kernel_size,
                                        strides=stride,
                                        padding=padding,
                                        data_format=df,
                                        _scope=sc)
    outputs = layer.apply(inputs)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)


@add_arg_scope
def max_pool3d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               data_format=DATA_FORMAT_NDHWC,
               outputs_collections=None,
               scope=None):
  """Adds a 3D Max Pooling op.

  It is assumed that the pooling is done per image but not in batch or channels.

  Args:
    inputs: A 5-D tensor of shape `[batch_size, depth, height, width, channels]`
      if `data_format` is `NDHWC`, and `[batch_size, channels, depth, height,
      width]` if `data_format` is `NCDHW`.
    kernel_size: A list of length 3: [kernel_depth, kernel_height, kernel_width]
      of the pooling kernel over which the op is computed. Can be an int if both
      values are the same.
    stride: A list of length 3: [stride_depth, stride_height, stride_width].
      Can be an int if both strides are the same. Note that presently
      both strides must have the same value.
    padding: The padding method, either 'VALID' or 'SAME'.
    data_format: A string. `NDHWC` (default) and `NCDHW` are supported.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.

  Returns:
    A `Tensor` representing the results of the pooling operation.

  Raises:
    ValueError: If `data_format` is neither `NDHWC` nor `NCDHW`.
    ValueError: If 'kernel_size' is not a 3-D list
  """
  if data_format not in (DATA_FORMAT_NCDHW, DATA_FORMAT_NDHWC):
    raise ValueError('data_format has to be either NCDHW or NDHWC.')
  with ops.name_scope(scope, 'MaxPool3D', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = pooling_layers.MaxPooling3D(pool_size=kernel_size,
                                        strides=stride,
                                        padding=padding,
                                        data_format=df,
                                        _scope=sc)
    outputs = layer.apply(inputs)
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
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
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
    ValueError: If arguments are invalid.

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
    num_classes: Total number of classes.
    on_value: A scalar defining the on-value.
    off_value: A scalar defining the off-value.
    outputs_collections: Collection to add the outputs.
    scope: Optional scope for name_scope.

  Returns:
    One-hot encoding of the labels.
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
    A tensor result of applying the layer, repetitions times.
  Raises:
    ValueError: If the op is unknown or wrong.
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


def _scale_gradient_shape(op):
  """Shape helper function for scale_gradient function below."""
  return [op.inputs[0].shape]


def _scale_gradient_grad(op, grad):
  """Python gradient helper function for scale_gradient function below."""
  return [grad * op.inputs[1], None]


@function.Defun(python_grad_func=_scale_gradient_grad,
                shape_func=_scale_gradient_shape)
def scale_gradient(inputs, gradient_multiplier):
  """Identity operation, but with the gradient multiplied by a tensor.

  The TensorFlow gradient system will compute the gradient with respect to
  `inputs` as the product of the gradient with respect to the `output`
  multiplied by a specified `gradient_multiplier` tensor.  If
  `gradient_multiplier` is equal to 1, then this results in the true gradient.
  Otherwise, it results in a scaled gradient.

  This can be useful for adjusting the relative learning rate of different
  parameter tensors when performing gradient descent, and because this rescaling
  can be inserted at arbitrary locations within a graph, is often more
  convenient to apply than simply rescaling the final computed gradients.

  Args:
    inputs: Tensor to be output.
    gradient_multiplier: Tensor by which to multiply the gradient with respect
      to `output` to compute the gradient with respect to `inputs`.  Its shape
      must be broadcastable to the shape of `inputs`.

  Returns:
    output Tensor, equal to `inputs`.
  """
  # gradient_multiplier is implicitly saved by decorator, and only used for
  # gradient computation.
  del gradient_multiplier

  return inputs


@add_arg_scope
def separable_convolution2d(
    inputs,
    num_outputs,
    kernel_size,
    depth_multiplier,
    stride=1,
    padding='SAME',
    data_format=DATA_FORMAT_NHWC,
    rate=1,
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer(),
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
  variable called `pointwise_weights`. Then, if `normalizer_fn` is None,
  it adds bias to the result, creating a variable called 'biases', otherwise,
  the `normalizer_fn` is applied. It finally applies an activation function
  to produce the end result.

  Args:
    inputs: A tensor of size [batch_size, height, width, channels].
    num_outputs: The number of pointwise convolution output filters. If is
      None, then we skip the pointwise convolution stage.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      depthwise convolution stride. Can be an int if both strides are the same.
    padding: One of 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    rate: A list of length 2: [rate_height, rate_width], specifying the dilation
      rates for atrous convolution. Can be an int if both rates are the same.
      If any value is larger than one, then both stride values need to be one.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: Whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.

  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `data_format` is invalid.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  layer_variable_getter = _build_variable_getter(
      {'bias': 'biases',
       'depthwise_kernel': 'depthwise_weights',
       'pointwise_kernel': 'pointwise_weights'})

  with variable_scope.variable_scope(
      scope, 'SeparableConv2d', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    if num_outputs is not None:
      # Apply separable conv using the SeparableConvolution2D layer.
      layer = convolutional_layers.SeparableConvolution2D(
          filters=num_outputs,
          kernel_size=kernel_size,
          strides=stride,
          padding=padding,
          data_format=df,
          dilation_rate=utils.two_element_tuple(rate),
          activation=None,
          depth_multiplier=depth_multiplier,
          use_bias=not normalizer_fn and biases_initializer,
          depthwise_initializer=weights_initializer,
          pointwise_initializer=weights_initializer,
          bias_initializer=biases_initializer,
          depthwise_regularizer=weights_regularizer,
          pointwise_regularizer=weights_regularizer,
          bias_regularizer=biases_regularizer,
          activity_regularizer=None,
          trainable=trainable,
          name=sc.name,
          dtype=inputs.dtype.base_dtype,
          _scope=sc,
          _reuse=reuse)
      outputs = layer.apply(inputs)

      # Add variables to collections.
      _add_variable_to_collections(layer.depthwise_kernel,
                                   variables_collections, 'weights')
      _add_variable_to_collections(layer.pointwise_kernel,
                                   variables_collections, 'weights')
      if layer.bias is not None:
        _add_variable_to_collections(layer.bias,
                                     variables_collections, 'biases')

      if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      # Actually apply depthwise conv instead of separable conv.
      dtype = inputs.dtype.base_dtype
      kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
      stride_h, stride_w = utils.two_element_tuple(stride)
      num_filters_in = utils.channel_dimension(
          inputs.get_shape(), df, min_rank=4)
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

      outputs = nn.depthwise_conv2d(inputs, depthwise_weights, strides, padding,
                                    rate=utils.two_element_tuple(rate),
                                    data_format=data_format)
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
                                            trainable=trainable,
                                            collections=biases_collections)
          outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def softmax(logits, scope=None):
  """Performs softmax on Nth dimension of N-dimensional logit tensor.

  For two-dimensional logits this reduces to tf.nn.softmax. The N-th dimension
  needs to have a specified number of elements (number of classes).

  Args:
    logits: N-dimensional `Tensor` with logits, where N > 1.
    scope: Optional scope for variable_scope.

  Returns:
    A `Tensor` with same shape and type as logits.
  """
  # TODO(jrru): Add axis argument which defaults to last dimension.
  with variable_scope.variable_scope(scope, 'softmax', [logits]):
    num_logits = utils.last_dimension(logits.get_shape(), min_rank=2)
    logits_2d = array_ops.reshape(logits, [-1, num_logits])
    predictions = nn.softmax(logits_2d)
    predictions = array_ops.reshape(predictions, array_ops.shape(logits))
    if context.in_graph_mode():
      predictions.set_shape(logits.get_shape())
    return predictions


@add_arg_scope
def spatial_softmax(features,
                    temperature=None,
                    name=None,
                    variables_collections=None,
                    trainable=True,
                    data_format='NHWC'):
  """Computes the spatial softmax of a convolutional feature map.

  First computes the softmax over the spatial extent of each channel of a
  convolutional feature map. Then computes the expected 2D position of the
  points of maximal activation for each channel, resulting in a set of
  feature keypoints [x1, y1, ... xN, yN] for all N channels.

  Read more here:
  "Learning visual feature spaces for robotic manipulation with
  deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.

  Args:
    features: A `Tensor` of size [batch_size, W, H, num_channels]; the
      convolutional feature map.
    temperature: Softmax temperature (optional). If None, a learnable
      temperature is created.
    name: A name for this operation (optional).
    variables_collections: Collections for the temperature variable.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
  Returns:
    feature_keypoints: A `Tensor` with size [batch_size, num_channels * 2];
      the expected 2D locations of each channel's feature keypoint (normalized
      to the range (-1,1)). The inner dimension is arranged as
      [x1, y1, ... xN, yN].
  Raises:
    ValueError: If unexpected data_format specified.
    ValueError: If num_channels dimension is unspecified.
  """
  shape = array_ops.shape(features)
  static_shape = features.shape
  if data_format == DATA_FORMAT_NHWC:
    height, width, num_channels = shape[1], shape[2], static_shape[3]
  elif data_format == DATA_FORMAT_NCHW:
    num_channels, height, width = static_shape[1], shape[2], shape[3]
  else:
    raise ValueError('data_format has to be either NCHW or NHWC.')
  if num_channels.value is None:
    raise ValueError('The num_channels dimension of the inputs to '
                     '`spatial_softmax` should be defined. Found `None`.')

  with ops.name_scope(name, 'spatial_softmax', [features]) as name:
    # Create tensors for x and y coordinate values, scaled to range [-1, 1].
    pos_x, pos_y = array_ops.meshgrid(math_ops.lin_space(-1., 1., num=height),
                                      math_ops.lin_space(-1., 1., num=width),
                                      indexing='ij')
    pos_x = array_ops.reshape(pos_x, [height * width])
    pos_y = array_ops.reshape(pos_y, [height * width])
    if temperature is None:
      temperature_collections = utils.get_variable_collections(
          variables_collections, 'temperature')
      temperature = variables.model_variable(
          'temperature',
          shape=(),
          dtype=dtypes.float32,
          initializer=init_ops.ones_initializer(),
          collections=temperature_collections,
          trainable=trainable)
    if data_format == 'NCHW':
      features = array_ops.reshape(features, [-1, height * width])
    else:
      features = array_ops.reshape(
          array_ops.transpose(features, [0, 3, 1, 2]), [-1, height * width])

    softmax_attention = nn.softmax(features/temperature)
    expected_x = math_ops.reduce_sum(
        pos_x * softmax_attention, [1], keep_dims=True)
    expected_y = math_ops.reduce_sum(
        pos_y * softmax_attention, [1], keep_dims=True)
    expected_xy = array_ops.concat([expected_x, expected_y], 1)
    feature_keypoints = array_ops.reshape(
        expected_xy, [-1, num_channels.value * 2])
    feature_keypoints.set_shape([None, num_channels.value * 2])
    return feature_keypoints


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
    A `Tensor` result of applying the stacked layers.

  Raises:
    ValueError: If the op is unknown or wrong.
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
    multiples.append(
        array_ops.strided_slice(array_ops.shape(inputs), [dim], [dim + 1]))
    if dim < (input_rank - 1):
      multiples.append(array_ops.ones([input_rank - 1 - dim], dtypes.int32))
    multiples = array_ops.concat(multiples, 0)
    return math_ops.div(inputs, array_ops.tile(lengths, multiples))


def poincare_normalize(x, axis=1, epsilon=1e-5, name=None):
  """Project into the Poincare ball with norm <= 1.0 - epsilon.

  https://en.wikipedia.org/wiki/Poincare_ball_model

  Used in
  Poincare Embeddings for Learning Hierarchical Representations
  Maximilian Nickel, Douwe Kiela
  https://arxiv.org/pdf/1705.08039.pdf

  For a 1-D tensor with `axis = 0`, computes

                (x * (1 - epsilon)) / ||x||     if ||x|| > 1 - epsilon
      output =
                 x                              otherwise

  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `axis`.

  Args:
    x: A `Tensor`.
    axis: Axis along which to normalize.  A scalar or a vector of
      integers.
    epsilon: A small deviation from the edge of the unit sphere for numerical
      stability.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` with the same shape as `x`.
  """
  with ops.name_scope(name, 'poincare_normalize', [x]) as name:
    x = ops.convert_to_tensor(x, name='x')
    square_sum = math_ops.reduce_sum(math_ops.square(x), axis, keep_dims=True)
    x_inv_norm = math_ops.rsqrt(square_sum)
    x_inv_norm = math_ops.minimum((1. - epsilon) * x_inv_norm, 1.)
    return math_ops.multiply(x, x_inv_norm, name=name)


def legacy_fully_connected(x,
                           num_output_units,
                           activation_fn=None,
                           weight_init=initializers.xavier_initializer(),
                           bias_init=init_ops.zeros_initializer(),
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
    activation_fn: Activation function, default set to None to skip it and
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
    ValueError: If x has rank less than 2 or if its last dimension is not set.
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
      out_shape = array_ops.unstack(array_ops.shape(x))
      out_shape[-1] = num_output_units

      y = array_ops.reshape(y, array_ops.stack(out_shape))

      static_shape = x.get_shape().as_list()
      static_shape[-1] = num_output_units
      y.set_shape(static_shape)

    return _apply_activation(y, activation_fn, output_collections)


# TODO(eiderm): Verify and fix autocomplete in colab (also relu6).
# Simple aliases which remove the activation_fn parameter.
elu = functools.partial(fully_connected, activation_fn=nn.elu)
legacy_relu = functools.partial(legacy_fully_connected, activation_fn=nn.relu)
legacy_linear = functools.partial(legacy_fully_connected, activation_fn=None)
relu = functools.partial(fully_connected, activation_fn=nn.relu)
relu6 = functools.partial(fully_connected, activation_fn=nn.relu6)
linear = functools.partial(fully_connected, activation_fn=None)

# Simple alias.
conv2d = convolution2d
conv3d = convolution3d
conv2d_transpose = convolution2d_transpose
conv3d_transpose = convolution3d_transpose
conv2d_in_plane = convolution2d_in_plane
separable_conv2d = separable_convolution2d
