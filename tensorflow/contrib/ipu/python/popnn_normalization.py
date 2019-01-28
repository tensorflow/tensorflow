# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Contains the normalization layer classes and their functional aliases."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

# This implementation is based on:
# tensorflow/contrib/layers/python/layers/normalization.py
__all__ = [
    'group_norm',
]

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


@add_arg_scope
def group_norm(inputs,
               groups=2,
               channels_axis=-1,
               reduction_axes=(-3, -2),
               center=True,
               scale=True,
               epsilon=1e-6,
               param_initializers=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               training=True,
               trainable=True,
               scope=None):
  """Functional interface for the group normalization layer.

  Reference: https://arxiv.org/abs/1803.08494.

    "Group Normalization", Yuxin Wu, Kaiming He

  Args:
    inputs: A Tensor with at least 2 dimensions one which is channels. All
     shape dimensions must be fully defined.
    groups: Integer. Divide the channels into this number of groups over which
      normalization statistics are computed. This number must be commensurate
      with the number of channels in `inputs`.
    channels_axis: An integer. Specifies index of channels axis which will be
      broken into `groups`, each of which whose statistics will be computed
      across. Must be mutually exclusive with `reduction_axes`. Preferred usage
      is to specify negative integers to be agnostic as to whether a batch
      dimension is included.
    reduction_axes: Tuple of integers. Specifies dimensions over which
       statistics will be accumulated. Must be mutually exclusive with
       `channels_axis`. Statistics will not be accumulated across axes not
       specified in `reduction_axes` nor `channel_axis`. Preferred usage is to
       specify negative integers to be agnostic to whether a batch dimension is
       included.

      Some sample usage cases:
        NHWC format: channels_axis=-1, reduction_axes=[-3, -2]
        NCHW format: channels_axis=-3, reduction_axes=[-2, -1]

    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta and gamma.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    training: Whether this is operation is being used in a training network.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.


  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
    ValueError: If channels dimension is not 1 or 3.
    ValueError: If number of groups is not commensurate with number of channels.
    ValueError: If reduction_axes or channels_axis are out of bounds.
    ValueError: If reduction_axes are not mutually exclusive with channels_axis.
  """

  inputs = ops.convert_to_tensor(inputs)
  original_shape = inputs.shape

  if inputs.shape.ndims is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if channels_axis > (inputs.shape.ndims - 1):
    raise ValueError('Axis is out of bounds.')

  # Standardize the channels_axis to be positive and identify # of channels.
  if channels_axis < 0:
    channels_axis = inputs.shape.ndims + channels_axis
  channels = inputs.shape[channels_axis].value

  if channels_axis == 1:
    data_format = DATA_FORMAT_NCHW
  elif channels_axis == inputs.shape.ndims - 1:
    data_format = DATA_FORMAT_NHWC
  else:
    raise ValueError('Unsupported data format, group norm only supports NCHW'
                     '(channel axis 1) and NHWC (channel axis -1).')

  if channels is None:
    raise ValueError('Inputs %s has undefined channel dimension: %d.' % (
        inputs.name, channels_axis))

  # Standardize the reduction_axes to be positive.
  reduction_axes = list(reduction_axes)
  for i in range(len(reduction_axes)):
    if reduction_axes[i] < 0:
      reduction_axes[i] += inputs.shape.ndims

  for a in reduction_axes:
    if a > inputs.shape.ndims:
      raise ValueError('Axis is out of bounds.')
    if inputs.shape[a].value is None:
      raise ValueError('Inputs %s has undefined dimensions %d.' % (
          inputs.name, a))
    if channels_axis == a:
      raise ValueError('reduction_axis must be mutually exclusive '
                       'with channels_axis')
  if groups > channels:
    raise ValueError('Invalid groups %d for %d channels.' % (groups, channels))
  if channels % groups != 0:
    raise ValueError('%d channels is not commensurate with %d groups.' %
                     (channels, groups))

  with variable_scope.variable_scope(
      scope, 'GroupNorm', [inputs], reuse=reuse) as sc:
    # Note that the params_shape is the number of channels always.
    params_shape = [channels]

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    dtype = inputs.dtype.base_dtype
    if param_initializers is None:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(
          variables_collections, 'beta')
      beta_initializer = param_initializers.get(
          'beta', init_ops.zeros_initializer())
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    else:
      beta = array_ops.constant(0.0, dtype=dtype, shape=params_shape)

    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma_initializer = param_initializers.get(
          'gamma', init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
    else:
      gamma = array_ops.constant(1.0, dtype=dtype, shape=params_shape)


    if training:
      outputs, _, _, _ = gen_popnn_ops.popnn_group_norm_training(
                                                        inputs=inputs,
                                                        gamma=gamma,
                                                        beta=beta,
                                                        data_format=data_format,
                                                        epsilon=epsilon,
                                                        num_groups=groups)

    else:
      # Calculate the moments.
      mean, inv_std_dev = gen_popnn_ops.popnn_group_norm_statistics(
                                                        inputs=inputs,
                                                        data_format=data_format,
                                                        epsilon=epsilon,
                                                        num_groups=groups)

      outputs = gen_popnn_ops.popnn_group_norm_inference(inputs=inputs,
                                                         gamma=gamma,
                                                         beta=beta,
                                                         mean=mean,
                                                         inv_std_dev=inv_std_dev,
                                                         data_format=data_format,
                                                         epsilon=epsilon,
                                                         num_groups=groups)

    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
