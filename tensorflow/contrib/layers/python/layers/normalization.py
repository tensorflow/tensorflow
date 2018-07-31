# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers
from tensorflow.python.eager import context
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.layers import Wrapper

__all__ = [
    'group_norm',
    'instance_norm',
    'WeightNorm'
]

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


class WeightNorm(Wrapper):
  """ This wrapper reparameterizes a layer by decoupling the weight's
  magnitude and direction. This speeds up convergence by improving the
  conditioning of the optimization problem.

  Weight Normalization: A Simple Reparameterization to Accelerate
  Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
  Tim Salimans, Diederik P. Kingma (2016)

  WeightNorm wrapper works for keras and tf layers.

  ```python
    net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
           input_shape=(32, 32, 3), data_init=True)(x)
    net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                     data_init=True)
    net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                     data_init=True)(net)
    net = WeightNorm(tf.keras.layers.Dense(n_classes),
                     data_init=True)(net)
  ```

  Arguments:
    layer: a layer instance.
    data_init: If `True` use data dependent variable initialization

  Raises:
    ValueError: If not initialized with a `Layer` instance.
    ValueError: If `Layer` does not contain a `kernel` of weights
    NotImplementedError: If `data_init` is True and running graph execution
  """
  def __init__(self, layer, data_init=False, **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `WeightNorm` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer))

    if not context.executing_eagerly() and data_init:
      raise NotImplementedError(
          'Data dependent variable initialization is not available for '
          'graph execution')

    self.initialized = True
    if data_init:
      self.initialized = False

    super(WeightNorm, self).__init__(layer, **kwargs)
    self._track_checkpointable(layer, name='layer')

  def _compute_weights(self):
    """Generate weights by combining the direction of weight vector
     with it's norm """
    with variable_scope.variable_scope('compute_weights'):
      self.layer.kernel = nn_impl.l2_normalize(
          self.layer.v, axis=self.norm_axes) * self.layer.g

  def _init_norm(self, weights):
    """Set the norm of the weight vector"""
    from tensorflow.python.ops.linalg_ops import norm
    with variable_scope.variable_scope('init_norm'):
      flat = array_ops.reshape(weights, [-1, self.layer_depth])
      return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

  def _data_dep_init(self, inputs):
    """Data dependent initialization for eager execution"""
    from tensorflow.python.ops.nn import moments
    from tensorflow.python.ops.math_ops import sqrt

    with variable_scope.variable_scope('data_dep_init'):
      # Generate data dependent init values
      activation = self.layer.activation
      self.layer.activation = None
      x_init = self.layer.call(inputs)
      m_init, v_init = moments(x_init, self.norm_axes)
      scale_init = 1. / sqrt(v_init + 1e-10)

    # Assign data dependent init values
    self.layer.g = self.layer.g * scale_init
    self.layer.bias = (-m_init * scale_init)
    self.layer.activation = activation
    self.initialized = True

  def build(self, input_shape):
    """Build `Layer`"""
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    self.input_spec = InputSpec(shape=input_shape)

    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = False

      if not hasattr(self.layer, 'kernel'):
        raise ValueError(
            '`WeightNorm` must wrap a layer that'
            ' contains a `kernel` for weights'
        )

      # The kernel's filter or unit dimension is -1
      self.layer_depth = int(self.layer.kernel.shape[-1])
      self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

      self.layer.v = self.layer.kernel
      self.layer.g = self.layer.add_variable(
          name="g",
          shape=(self.layer_depth,),
          initializer=initializers.get('ones'),
          dtype=self.layer.kernel.dtype,
          trainable=True)

      with ops.control_dependencies([self.layer.g.assign(
          self._init_norm(self.layer.v))]):
        self._compute_weights()

      self.layer.built = True

    super(WeightNorm, self).build()
    self.built = True

  def call(self, inputs):
    """Call `Layer`"""
    if context.executing_eagerly():
      if not self.initialized:
        self._data_dep_init(inputs)
      self._compute_weights()  # Recompute weights for each forward pass

    output = self.layer.call(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tensor_shape.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())

@add_arg_scope
def instance_norm(inputs,
                  center=True,
                  scale=True,
                  epsilon=1e-6,
                  activation_fn=None,
                  param_initializers=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  data_format=DATA_FORMAT_NHWC,
                  scope=None):
  """Functional interface for the instance normalization layer.

  Reference: https://arxiv.org/abs/1607.08022.

    "Instance Normalization: The Missing Ingredient for Fast Stylization"
    Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
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
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  inputs = ops.convert_to_tensor(inputs)
  inputs_shape = inputs.shape
  inputs_rank = inputs.shape.ndims

  if inputs_rank is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  with variable_scope.variable_scope(
      scope, 'InstanceNorm', [inputs], reuse=reuse) as sc:
    if data_format == DATA_FORMAT_NCHW:
      reduction_axis = 1
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      reduction_axis = inputs_rank - 1
      params_shape_broadcast = None
    moments_axes = list(range(inputs_rank))
    del moments_axes[reduction_axis]
    del moments_axes[0]
    params_shape = inputs_shape[reduction_axis:reduction_axis + 1]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

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
      if params_shape_broadcast:
        beta = array_ops.reshape(beta, params_shape_broadcast)
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
      if params_shape_broadcast:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Calculate the moments (instance activations).
    mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)

    # Compute instance normalization.
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon, name='instancenorm')
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def group_norm(inputs,
               groups=32,
               channels_axis=-1,
               reduction_axes=(-3, -2),
               center=True,
               scale=True,
               epsilon=1e-6,
               activation_fn=None,
               param_initializers=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
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
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
    ValueError: If number of groups is not commensurate with number of channels.
    ValueError: If reduction_axes or channels_axis are out of bounds.
    ValueError: If reduction_axes are not mutually exclusive with channels_axis.
  """
  # TODO(shlens): Support partially defined shapes for the inputs.
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

  # Determine axes before channels. Some examples of common image formats:
  #  'NCHW': before = [N], after = [HW]
  #  'NHWC': before = [NHW], after = []
  axes_before_channels = inputs.shape.as_list()[:channels_axis]
  axes_after_channels = inputs.shape.as_list()[channels_axis+1:]

  # Manually broadcast the parameters to conform to the number of groups.
  params_shape_broadcast = ([1] * len(axes_before_channels) +
                            [groups, channels // groups] +
                            [1] * len(axes_after_channels))

  # Reshape the input by the group within the channel dimension.
  inputs_shape = (axes_before_channels + [groups, channels // groups] +
                  axes_after_channels)
  inputs = array_ops.reshape(inputs, inputs_shape)

  # Determine the dimensions across which moments are calculated.
  moments_axes = [channels_axis + 1]
  for a in reduction_axes:
    if a > channels_axis:
      moments_axes.append(a + 1)
    else:
      moments_axes.append(a)

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
      beta = array_ops.reshape(beta, params_shape_broadcast)

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
      gamma = array_ops.reshape(gamma, params_shape_broadcast)

    # Calculate the moments.
    mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)

    # Compute normalization.
    # TODO(shlens): Fix nn.batch_normalization to handle the 5-D Tensor
    # appropriately so that this operation may be faster.
    gain = math_ops.rsqrt(variance + epsilon)
    offset = -mean * gain
    if gamma is not None:
      gain *= gamma
      offset *= gamma
    if beta is not None:
      offset += beta
    outputs = inputs * gain + offset

    # Collapse the groups into the channel dimension.
    outputs = array_ops.reshape(outputs, original_shape)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
