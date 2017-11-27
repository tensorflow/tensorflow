# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Layers that act as activation functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import constraints
from tensorflow.python.keras._impl.keras import initializers
from tensorflow.python.keras._impl.keras import regularizers
from tensorflow.python.keras._impl.keras.engine import InputSpec
from tensorflow.python.keras._impl.keras.engine import Layer


class LeakyReLU(Layer):
  """Leaky version of a Rectified Linear Unit.

  It allows a small gradient when the unit is not active:
  `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as the input.

  Arguments:
      alpha: float >= 0. Negative slope coefficient.

  """

  def __init__(self, alpha=0.3, **kwargs):
    super(LeakyReLU, self).__init__(**kwargs)
    self.supports_masking = True
    self.alpha = K.cast_to_floatx(alpha)

  def call(self, inputs):
    return K.relu(inputs, alpha=self.alpha)

  def get_config(self):
    config = {'alpha': float(self.alpha)}
    base_config = super(LeakyReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class PReLU(Layer):
  """Parametric Rectified Linear Unit.

  It follows:
  `f(x) = alpha * x for x < 0`,
  `f(x) = x for x >= 0`,
  where `alpha` is a learned array with the same shape as x.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as the input.

  Arguments:
      alpha_initializer: initializer function for the weights.
      alpha_regularizer: regularizer for the weights.
      alpha_constraint: constraint for the weights.
      shared_axes: the axes along which to share learnable
          parameters for the activation function.
          For example, if the incoming feature maps
          are from a 2D convolution
          with output shape `(batch, height, width, channels)`,
          and you wish to share parameters across space
          so that each filter only has one set of parameters,
          set `shared_axes=[1, 2]`.

  """

  def __init__(self,
               alpha_initializer='zeros',
               alpha_regularizer=None,
               alpha_constraint=None,
               shared_axes=None,
               **kwargs):
    super(PReLU, self).__init__(**kwargs)
    self.supports_masking = True
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.alpha_regularizer = regularizers.get(alpha_regularizer)
    self.alpha_constraint = constraints.get(alpha_constraint)
    if shared_axes is None:
      self.shared_axes = None
    elif not isinstance(shared_axes, (list, tuple)):
      self.shared_axes = [shared_axes]
    else:
      self.shared_axes = list(shared_axes)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    param_shape = input_shape[1:]
    self.param_broadcast = [False] * len(param_shape)
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1
        self.param_broadcast[i - 1] = True
    self.alpha = self.add_weight(
        shape=param_shape,
        name='alpha',
        initializer=self.alpha_initializer,
        regularizer=self.alpha_regularizer,
        constraint=self.alpha_constraint)
    # Set input spec
    axes = {}
    if self.shared_axes:
      for i in range(1, len(input_shape)):
        if i not in self.shared_axes:
          axes[i] = input_shape[i]
    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
    self.built = True

  def call(self, inputs, mask=None):
    pos = K.relu(inputs)
    if K.backend() == 'theano':
      neg = (K.pattern_broadcast(self.alpha, self.param_broadcast) *
             (inputs - K.abs(inputs)) * 0.5)
    else:
      neg = -self.alpha * K.relu(-inputs)
    return pos + neg

  def get_config(self):
    config = {
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
        'alpha_constraint': constraints.serialize(self.alpha_constraint),
        'shared_axes': self.shared_axes
    }
    base_config = super(PReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ELU(Layer):
  """Exponential Linear Unit.

  It follows:
  `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
  `f(x) = x for x >= 0`.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as the input.

  Arguments:
      alpha: scale for the negative factor.

  """

  def __init__(self, alpha=1.0, **kwargs):
    super(ELU, self).__init__(**kwargs)
    self.supports_masking = True
    self.alpha = K.cast_to_floatx(alpha)

  def call(self, inputs):
    return K.elu(inputs, self.alpha)

  def get_config(self):
    config = {'alpha': float(self.alpha)}
    base_config = super(ELU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ThresholdedReLU(Layer):
  """Thresholded Rectified Linear Unit.

  It follows:
  `f(x) = x for x > theta`,
  `f(x) = 0 otherwise`.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as the input.

  Arguments:
      theta: float >= 0. Threshold location of activation.

  """

  def __init__(self, theta=1.0, **kwargs):
    super(ThresholdedReLU, self).__init__(**kwargs)
    self.supports_masking = True
    self.theta = K.cast_to_floatx(theta)

  def call(self, inputs, mask=None):
    return inputs * K.cast(inputs > self.theta, K.floatx())

  def get_config(self):
    config = {'theta': float(self.theta)}
    base_config = super(ThresholdedReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
