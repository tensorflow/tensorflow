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
"""Built-in activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export

# b/123041942
# In TF 2.x, if the `tf.nn.softmax` is used as an activation function in Keras
# layers, it gets serialized as 'softmax_v2' instead of 'softmax' as the
# internal method name is returned in serialization. This results in errors in
# model exporting and loading as Keras can't find any activation function with
# the name of `softmax_v2`.

# This dict maps the activation function name from its v2 version to its
# canonical name.
_TF_ACTIVATIONS_V2 = {
    'softmax_v2': 'softmax',
}


@keras_export('keras.activations.softmax')
def softmax(x, axis=-1):
  """Applies the softmax activation function.

  This activation, often used to generate outputs representing probabilities, first applies the standard exponential function to the input and then normalizes the result.

  ```python
  t = tf.constant([[-1.1, 1.1, .7])
  softmax(t).numpy() # ==> [[0.06220971, 0.5614435 , 0.37634683]]
  sum(sum(softmax(t))).numpy() # ==> 1.0
  ```

  Arguments:
      `x` : A `Tensor` or `Variable`.
      `axis`: An `integer` giving the axis along which the softmax normalization is to be applied.

  Returns:
      A `Tensor` representing the softmax transformation of the input tensor.

  Raises:
      ValueError: if the dimension of `x` is 1.
      ValueError: if an element of the input is not of a permissible `float`, `half`, or `double` type.
  """
  ndim = K.ndim(x)
  if ndim == 2:
    return nn.softmax(x)
  elif ndim > 2:
    e = math_ops.exp(x - math_ops.reduce_max(x, axis=axis, keepdims=True))
    s = math_ops.reduce_sum(e, axis=axis, keepdims=True)
    return e / s
  else:
    raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                     'Received input: %s' % (x,))


@keras_export('keras.activations.elu')
def elu(x, alpha=1.0):
  """Exponential linear unit.

  Arguments:
      x: Input tensor.
      alpha: A scalar, slope of negative section.

  Returns:
      The exponential linear activation: `x` if `x > 0` and
        `alpha * (exp(x)-1)` if `x < 0`.

  Reference:
      - [Fast and Accurate Deep Network Learning by Exponential
        Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
  """
  return K.elu(x, alpha)


@keras_export('keras.activations.selu')
def selu(x):
  """Scaled Exponential Linear Unit (SELU).

  SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
  are pre-defined constants. The values of `alpha` and `scale` are
  chosen so that the mean and variance of the inputs are preserved
  between two consecutive layers as long as the weights are initialized
  correctly (see `lecun_normal` initialization) and the number of inputs
  is "large enough" (see references for more information).

  Arguments:
      x: A tensor or variable to compute the activation function for.

  Returns:
      The scaled exponential unit activation: `scale * elu(x, alpha)`.

  # Note
      - To be used together with the initialization "lecun_normal".
      - To be used together with the dropout variant "AlphaDropout".

  References:
      - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * K.elu(x, alpha)


@keras_export('keras.activations.softplus')
def softplus(x):
  """Softplus activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The softplus activation: `log(exp(x) + 1)`.
  """
  return nn.softplus(x)


@keras_export('keras.activations.softsign')
def softsign(x):
  """Softsign activation function.

  Arguments:
      x: Input tensor.

  Returns:
      The softplus activation: `x / (abs(x) + 1)`.
  """
  return nn.softsign(x)


@keras_export('keras.activations.relu')
def relu(x, alpha=0., max_value=None, threshold=0.):
  """Applies the ReLU activation function.

  With default values, this returns the standard ReLU activation: the element-wise maximum of 0 and the input tensor. This activation is used in many graph architectures.

  Modifying default parameter values allows you to use thresholds other than zero, to set maximum values of the activation, and to use non-zero multiples of the input for values below the threshold.
  
  ```python
  t = tf.constant([.55, -.25, 0, -1.25, .75, 1.333])

  relu(t).eval() # ==> [0.55  0.    0.    0.    0.75  1.333]
  relu(t, alpha=.05).eval() # ==> [0.55   -0.0125  0.     -0.0625  0.75    1.333]
  relu(t, max_value=1.0).eval() # ==> [0.55 0.   0.   0.   0.75 1.]
  relu(t, threshold=-.5).eval() # ==> [ 0.55  -0.25   0.    -0.     0.75   1.333]
  ```

  Arguments:
    `x`: A `Tensor` or `Variable`.
    `alpha`: A `float` that governs the slope of the activation function for values lower than the threshold. By default, this is `0.`; small positive values correspond to 'leaky' ReLUs.
    `max_value`: A `float` or `None` giving the saturation threshold (the largest value the function can take). The default is `None`, indicating that there is no such threshold.
    `threshold`: A `float` giving the threshold value below which values will be damped or set to zero (`0.` by default).

  Returns:
    A `Tensor` representing the input tensor transformed by the ReLU activation function.

  Raises:
    ValueError: if an element of the input is not of a permissible `float` or `int` type.
  """
  return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)


@keras_export('keras.activations.tanh')
def tanh(x):
  return nn.tanh(x)


@keras_export('keras.activations.sigmoid')
def sigmoid(x):
  return nn.sigmoid(x)


@keras_export('keras.activations.exponential')
def exponential(x):
  return math_ops.exp(x)


@keras_export('keras.activations.hard_sigmoid')
def hard_sigmoid(x):
  """Hard sigmoid activation function.

  Faster to compute than sigmoid activation.

  Arguments:
      x: Input tensor.

  Returns:
      Hard sigmoid activation:
      - `0` if `x < -2.5`
      - `1` if `x > 2.5`
      - `0.2 * x + 0.5` if `-2.5 <= x <= 2.5`.
  """
  return K.hard_sigmoid(x)


@keras_export('keras.activations.linear')
def linear(x):
  return x


@keras_export('keras.activations.serialize')
def serialize(activation):
  if activation.__name__ in _TF_ACTIVATIONS_V2:
    return _TF_ACTIVATIONS_V2[activation.__name__]
  return activation.__name__


@keras_export('keras.activations.deserialize')
def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='activation function')


@keras_export('keras.activations.get')
def get(identifier):
  if identifier is None:
    return linear
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'activation function identifier:', identifier)
