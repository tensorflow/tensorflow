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
"""Built-in activation functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
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
  """The softmax activation function transforms the outputs so that all values are in

  range (0, 1) and sum to 1. It is often used as the activation for the last
  layer of a classification network because the result could be interpreted as
  a probability distribution. The softmax of x is calculated by
  exp(x)/tf.reduce_sum(exp(x)).

  Arguments:
      x : Input tensor.
      axis: Integer, axis along which the softmax normalization is applied.

  Returns:
      Tensor, output of softmax transformation (all values are non-negative
        and sum to 1).

  Raises:
      ValueError: In case `dim(x) == 1`.
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

  The Scaled Exponential Linear Unit (SELU) activation function is:
  `scale * x` if `x > 0` and `scale * alpha * (exp(x) - 1)` if `x < 0`
  where `alpha` and `scale` are pre-defined constants
  (`alpha = 1.67326324`
  and `scale = 1.05070098`).
  The SELU activation function multiplies  `scale` > 1 with the
  `[elu](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/activations/elu)`
  (Exponential Linear Unit (ELU)) to ensure a slope larger than one
  for positive net inputs.

  The values of `alpha` and `scale` are
  chosen so that the mean and variance of the inputs are preserved
  between two consecutive layers as long as the weights are initialized
  correctly (see [`lecun_normal` initialization]
  (https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_normal))
  and the number of inputs is "large enough"
  (see references for more information).

  ![](https://cdn-images-1.medium.com/max/1600/1*m0e8lZU_Zrkh4ESfQkY2Pw.png)
  (Courtesy: Blog on Towards DataScience at
  https://towardsdatascience.com/selu-make-fnns-great-again-snn-8d61526802a9)

  Example Usage:
  ```python3
  n_classes = 10 #10-class problem
  model = models.Sequential()
  model.add(Dense(64, kernel_initializer='lecun_normal', activation='selu',
  input_shape=(28, 28, 1))))
  model.add(Dense(32, kernel_initializer='lecun_normal', activation='selu'))
  model.add(Dense(16, kernel_initializer='lecun_normal', activation='selu'))
  model.add(Dense(n_classes, activation='softmax'))
  ```

  Arguments:
      x: A tensor or variable to compute the activation function for.

  Returns:
      The scaled exponential unit activation: `scale * elu(x, alpha)`.

  # Note
      - To be used together with the initialization "[lecun_normal]
      (https://www.tensorflow.org/api_docs/python/tf/keras/initializers/lecun_normal)".
      - To be used together with the dropout variant "[AlphaDropout]
      (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AlphaDropout)".

  References:
      [Self-Normalizing Neural Networks (Klambauer et al, 2017)]
      (https://arxiv.org/abs/1706.02515)
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
def relu(x, alpha=0., max_value=None, threshold=0):
  """Rectified Linear Unit.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:
  `f(x) = max_value` for `x >= max_value`,
  `f(x) = x` for `threshold <= x < max_value`,
  `f(x) = alpha * (x - threshold)` otherwise.

  Arguments:
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: float. Saturation threshold.
      threshold: float. Threshold value for thresholded activation.

  Returns:
      A tensor.
  """
  return K.relu(x, alpha=alpha, max_value=max_value, threshold=threshold)


@keras_export('keras.activations.tanh')
def tanh(x):
  """Hyperbolic tangent activation function.
  
  For example:
  ```python
  # Constant 1-D tensor populated with value list.
  a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32) #Input tensor
  b = tf.keras.activations.tanh(a) 
  # b = array([-0.9950547, -0.7615942, 0., 0.7615942, 0.9950547], dtype=float32)  #Output tensor
  ```
  
  Arguments:
      x: Input tensor.
  Returns:
      Tensor of same shape and dtype of input `x`, with tanh activation: `tanh(x) = sinh(x)/cosh(x) = ((exp(x) -
      exp(-x))/(exp(x) + exp(-x)))`.
  """
  return nn.tanh(x)


@keras_export('keras.activations.sigmoid')
def sigmoid(x):
  """Sigmoid activation function.
  Applies the sigmoid activation function. The sigmoid function is defined as
  1 divided by (1 + exp(-x)). It's curve is like an "S" and is like a smoothed
  version of the Heaviside (Unit Step Function) function. For small values
  (<-5) the sigmoid returns a value close to zero and for larger values (>5) the result of the function gets close to 1.
 For example:
  ```python
  # Constant 1-D tensor populated with value list.
  a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32) #Input tensor
  b = tf.keras.activations.sigmoid(a) 
  # b = array([0.04742587, 0.26894143, 0.5, 0.7310586 , 0.95257413], dtype=float32)  #Output tensor
  ```
  
  Arguments:
      x: Input tensor.
  Returns:
      Tensor with the sigmoid activation: `(1.0 / (1.0 + exp(-x)))`. Tensor will be of same shape and dtype of input `x`.
  """
  return nn.sigmoid(x)


@keras_export('keras.activations.exponential')
def exponential(x):
  """Exponential activation function.
 For example:
  ```python
  # Constant 1-D tensor populated with value list.
  a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32) #Input tensor
  b = tf.keras.activations.exponential(a) 
  # b = array([ 0.04978707, 0.36787945,  1., 2.7182817 , 20.085537], dtype=float32)  #Output tensor
  ```
  
  Arguments:
      x: Input tensor.
  Returns:
      Tensor with exponential activation: `exp(x)`. Tensor will be of same shape and dtype of input `x`.
  """
  return math_ops.exp(x)


@keras_export('keras.activations.hard_sigmoid')
def hard_sigmoid(x):
  """Hard sigmoid activation function.
  Faster to compute than sigmoid activation.
   
   For example:
  ```python
  # Constant 1-D tensor populated with value list.
  a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32) #Input tensor
  b = tf.keras.activations.sigmoid(a) 
  # b = <tf.Tensor: id=11, shape=(5,), dtype=float32, numpy=array([0. , 0.3, 0.5, 0.7, 1. ], dtype=float32)  #Output tensor
  ```
  Arguments:
      x: Input tensor.
  Returns: Tensor with hard sigmoid activation.
  """

  return K.hard_sigmoid(x)


@keras_export('keras.activations.linear')
def linear(x):
  """Linear activation function.
  For example:
  ```python
  # Constant 1-D tensor populated with value list.
  a = tf.constant([-3.0,-1.0, 0.0,1.0,3.0], dtype = tf.float32) #Input tensor
  b = tf.keras.activations.linear(a) 
  # b = <tf.Tensor: shape=(5,), dtype=float32, numpy=array([-3., -1.,  0.,  1.,  3.], dtype=float32)>  #Output tensor
  ```
  
  Arguments:
      x: Input tensor.
  Returns:
      The same output tensor as input tensor `x`.
  """
  return x



@keras_export('keras.activations.serialize')
def serialize(activation):
    """ Returns name attribute (`__name__`) of function.
  Arguments:
      x : Function
  Returns:
      String denoting the name attribute of the input function
      
  For example:
  ```python
  tf.keras.activations.serialize(tf.keras)  
  #Output:'tensorflow.python.keras.api._v2.keras'
  
  tf.keras.activations.serialize(tf.keras.activations.sigmoid) 
  #Output: 'sigmoid'
  
  tf.keras.activations.serialize('abcd')  
  #Output: ValueError: ('Cannot serialize', 'abcd')
  ```
  Raises:
      ValueError: The input function is not a valid one.
  """
  if (hasattr(activation, '__name__') and
      activation.__name__ in _TF_ACTIVATIONS_V2):
    return _TF_ACTIVATIONS_V2[activation.__name__]
  return serialize_keras_object(activation)

@keras_export('keras.activations.deserialize')
def deserialize(name, custom_objects=None):
    """ Returns activation function denoted by input string.
  Arguments:
      x : String
  Returns:
      Tensorlow Activation function denoted by input string.
      
  For example:
  ```python
  tf.keras.activations.deserialize('linear')  
  #Output: <function linear at 0x1239596a8>
  
  tf.keras.activations.deserialize('sigmoid') 
  #Output: <function sigmoid at 0x123959510>
  
  tf.keras.activations.deserialize('abcd')  
  #Output: ValueError: Unknown activation function:abcd
  ```
  Raises:
      ValueError: `Unknown activation function` if the input string does not denote any defined activation function.
  """
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='activation function')


@keras_export('keras.activations.get')
def get(identifier):
    """Returns function.
  Arguments:
      `x` : Function or string
  Returns:
      Activation function denoted by input:
      - `Linear activation function` if input is `None`.
      - Function corresponding to the input string or input function.
      
  For example:
  ```python
  tf.keras.activations.get('softmax') . 
  #Output: <function softmax at 0x1222a3d90>
  tf.keras.activations.get(tf.keras.activations.softmax)
  #Output: <function softmax at 0x1222a3d90>
  tf.keras.activations.get(None)
  #Output: <function linear at 0x1239596a8>
  
  tf.keras.activations.get(abs)
  #Output: <built-in function abs>
  
  tf.keras.activations.get('abcd')
  #Output: ValueError: Unknown activation function:abcd
  ```
 
  Raises:
      ValueError: Input is an unknown function or string, i.e., the input does not denote any defined function.
  """
  if identifier is None:
    return linear
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  elif isinstance(identifier, dict):
    return deserialize_keras_object(
        identifier, printable_module_name='activation')
  else:
    raise TypeError(
        'Could not interpret activation function identifier: {}'.format(
            repr(identifier)))
