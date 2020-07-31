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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.LeakyReLU')
class LeakyReLU(Layer):
  """Leaky version of a Rectified Linear Unit.

  It allows a small gradient when the unit is not active:

  ```
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
  ```

  Usage:

  >>> layer = tf.keras.layers.LeakyReLU()
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-0.9, -0.3, 0.0, 2.0]
  >>> layer = tf.keras.layers.LeakyReLU(alpha=0.1)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-0.3, -0.1, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    alpha: Float >= 0. Negative slope coefficient. Default to 0.3.

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

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.PReLU')
class PReLU(Layer):
  """Parametric Rectified Linear Unit.

  It follows:

  ```
    f(x) = alpha * x for x < 0
    f(x) = x for x >= 0
  ```

  where `alpha` is a learned array with the same shape as x.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    alpha_initializer: Initializer function for the weights.
    alpha_regularizer: Regularizer for the weights.
    alpha_constraint: Constraint for the weights.
    shared_axes: The axes along which to share learnable
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

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    param_shape = list(input_shape[1:])
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1
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

  def call(self, inputs):
    pos = K.relu(inputs)
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

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.ELU')
class ELU(Layer):
  """Exponential Linear Unit.

  It follows:

  ```
    f(x) =  alpha * (exp(x) - 1.) for x < 0
    f(x) = x for x >= 0
  ```

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    alpha: Scale for the negative factor.
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

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.ThresholdedReLU')
class ThresholdedReLU(Layer):
  """Thresholded Rectified Linear Unit.

  It follows:

  ```
    f(x) = x for x > theta
    f(x) = 0 otherwise`
  ```

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    theta: Float >= 0. Threshold location of activation.
  """

  def __init__(self, theta=1.0, **kwargs):
    super(ThresholdedReLU, self).__init__(**kwargs)
    self.supports_masking = True
    self.theta = K.cast_to_floatx(theta)

  def call(self, inputs):
    theta = math_ops.cast(self.theta, inputs.dtype)
    return inputs * math_ops.cast(math_ops.greater(inputs, theta), inputs.dtype)

  def get_config(self):
    config = {'theta': float(self.theta)}
    base_config = super(ThresholdedReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.Softmax')
class Softmax(Layer):
  """Softmax activation function.

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    axis: Integer, axis along which the softmax normalization is applied.
  """

  def __init__(self, axis=-1, **kwargs):
    super(Softmax, self).__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis

  def call(self, inputs):
    return K.softmax(inputs, axis=self.axis)

  def get_config(self):
    config = {'axis': self.axis}
    base_config = super(Softmax, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape


@keras_export('keras.layers.ReLU')
class ReLU(Layer):
  """Rectified Linear Unit activation function.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:

  ```
    f(x) = max_value if x >= max_value
    f(x) = x if threshold <= x < max_value
    f(x) = negative_slope * (x - threshold) otherwise
  ```

  Usage:

  >>> layer = tf.keras.layers.ReLU()
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.ReLU(max_value=1.0)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 1.0]
  >>> layer = tf.keras.layers.ReLU(negative_slope=1.0)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [-3.0, -1.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.ReLU(threshold=1.5)
  >>> output = layer([-3.0, -1.0, 1.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as the input.

  Arguments:
    max_value: Float >= 0. Maximum activation value. Default to None, which
      means unlimited.
    negative_slope: Float >= 0. Negative slope coefficient. Default to 0.
    threshold: Float. Threshold value for thresholded activation. Default to 0.
  """

  def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
    super(ReLU, self).__init__(**kwargs)
    if max_value is not None and max_value < 0.:
      raise ValueError('max_value of Relu layer '
                       'cannot be negative value: ' + str(max_value))
    if negative_slope < 0.:
      raise ValueError('negative_slope of Relu layer '
                       'cannot be negative value: ' + str(negative_slope))
    if threshold is None:
      raise ValueError('threshold of Relu layer '
                       'cannot be None. Required a float')

    self.support_masking = True
    if max_value is not None:
      max_value = K.cast_to_floatx(max_value)
    self.max_value = max_value
    self.negative_slope = K.cast_to_floatx(negative_slope)
    self.threshold = K.cast_to_floatx(threshold)

  def call(self, inputs):
    # alpha is used for leaky relu slope in activations instead of
    # negative_slope.
    return K.relu(inputs,
                  alpha=self.negative_slope,
                  max_value=self.max_value,
                  threshold=self.threshold)

  def get_config(self):
    config = {
        'max_value': self.max_value,
        'negative_slope': self.negative_slope,
        'threshold': self.threshold
    }
    base_config = super(ReLU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape
