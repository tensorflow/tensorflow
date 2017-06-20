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
"""Keras built-in activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.platform import tf_logging as logging


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
    return K.softmax(x)
  elif ndim > 2:
    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    s = K.sum(e, axis=axis, keepdims=True)
    return e / s
  else:
    raise ValueError('Cannot apply softmax to a tensor that is 1D')


def elu(x, alpha=1.0):
  return K.elu(x, alpha)


def softplus(x):
  return K.softplus(x)


def softsign(x):
  return K.softsign(x)


def relu(x, alpha=0., max_value=None):
  return K.relu(x, alpha=alpha, max_value=max_value)


def tanh(x):
  return K.tanh(x)


def sigmoid(x):
  return K.sigmoid(x)


def hard_sigmoid(x):
  return K.hard_sigmoid(x)


def linear(x):
  return x


def serialize(activation):
  return activation.__name__


def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='activation function')


def get(identifier):
  if identifier is None:
    return linear
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    if isinstance(identifier, Layer):
      logging.warning(
          'Do not pass a layer instance (such as {identifier}) as the '
          'activation argument of another layer. Instead, advanced '
          'activation layers should be used just like any other '
          'layer in a model.'.format(identifier=identifier.__class__.__name__))
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'activation function identifier:', identifier)
