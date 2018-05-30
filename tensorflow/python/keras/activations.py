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
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.activations.softmax')
def softmax(x, axis=-1):
  """Softmax activation function.

  Arguments:
      x : Tensor.
      axis: Integer, axis along which the softmax normalization is applied.

  Returns:
      Tensor, output of softmax transformation.

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
    raise ValueError('Cannot apply softmax to a tensor that is 1D')


@tf_export('keras.activations.elu')
def elu(x, alpha=1.0):
  return K.elu(x, alpha)


@tf_export('keras.activations.selu')
def selu(x):
  """Scaled Exponential Linear Unit. (Klambauer et al., 2017).

  Arguments:
      x: A tensor or variable to compute the activation function for.

  Returns:
      Tensor with the same shape and dtype as `x`.

  # Note
      - To be used together with the initialization "lecun_normal".
      - To be used together with the dropout variant "AlphaDropout".

  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * K.elu(x, alpha)


@tf_export('keras.activations.softplus')
def softplus(x):
  return nn.softplus(x)


@tf_export('keras.activations.softsign')
def softsign(x):
  return nn.softsign(x)


@tf_export('keras.activations.relu')
def relu(x, alpha=0., max_value=None):
  return K.relu(x, alpha=alpha, max_value=max_value)


@tf_export('keras.activations.tanh')
def tanh(x):
  return nn.tanh(x)


@tf_export('keras.activations.sigmoid')
def sigmoid(x):
  return nn.sigmoid(x)


@tf_export('keras.activations.hard_sigmoid')
def hard_sigmoid(x):
  return K.hard_sigmoid(x)


@tf_export('keras.activations.linear')
def linear(x):
  return x


@tf_export('keras.activations.serialize')
def serialize(activation):
  return activation.__name__


@tf_export('keras.activations.deserialize')
def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='activation function')


@tf_export('keras.activations.get')
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
