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
# pylint: disable=unused-import
"""Built-in loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras._impl.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.metrics.mean_squared_error',
           'keras.losses.mean_squared_error')
def mean_squared_error(y_true, y_pred):
  return K.mean(math_ops.square(y_pred - y_true), axis=-1)


@tf_export('keras.metrics.mean_absolute_error',
           'keras.losses.mean_absolute_error')
def mean_absolute_error(y_true, y_pred):
  return K.mean(math_ops.abs(y_pred - y_true), axis=-1)


@tf_export('keras.metrics.mean_absolute_percentage_error',
           'keras.losses.mean_absolute_percentage_error')
def mean_absolute_percentage_error(y_true, y_pred):
  diff = math_ops.abs(
      (y_true - y_pred) / K.clip(math_ops.abs(y_true), K.epsilon(), None))
  return 100. * K.mean(diff, axis=-1)


@tf_export('keras.metrics.mean_squared_logarithmic_error',
           'keras.losses.mean_squared_logarithmic_error')
def mean_squared_logarithmic_error(y_true, y_pred):
  first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
  second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
  return K.mean(math_ops.square(first_log - second_log), axis=-1)


@tf_export('keras.metrics.squared_hinge', 'keras.losses.squared_hinge')
def squared_hinge(y_true, y_pred):
  return K.mean(
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)


@tf_export('keras.metrics.hinge', 'keras.losses.hinge')
def hinge(y_true, y_pred):
  return K.mean(math_ops.maximum(1. - y_true * y_pred, 0.), axis=-1)


@tf_export('keras.losses.categorical_hinge')
def categorical_hinge(y_true, y_pred):
  pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
  neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
  return math_ops.maximum(0., neg - pos + 1.)


@tf_export('keras.losses.logcosh')
def logcosh(y_true, y_pred):
  """Logarithm of the hyperbolic cosine of the prediction error.

  `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
  to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
  like the mean squared error, but will not be so strongly affected by the
  occasional wildly incorrect prediction.

  Arguments:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.

  Returns:
      Tensor with one scalar loss entry per sample.
  """

  def _logcosh(x):
    return x + nn.softplus(-2. * x) - math_ops.log(2.)

  return K.mean(_logcosh(y_pred - y_true), axis=-1)


@tf_export('keras.metrics.categorical_crossentropy',
           'keras.losses.categorical_crossentropy')
def categorical_crossentropy(y_true, y_pred):
  return K.categorical_crossentropy(y_true, y_pred)


@tf_export('keras.metrics.sparse_categorical_crossentropy',
           'keras.losses.sparse_categorical_crossentropy')
def sparse_categorical_crossentropy(y_true, y_pred):
  return K.sparse_categorical_crossentropy(y_true, y_pred)


@tf_export('keras.metrics.binary_crossentropy',
           'keras.losses.binary_crossentropy')
def binary_crossentropy(y_true, y_pred):
  return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


@tf_export('keras.metrics.kullback_leibler_divergence',
           'keras.losses.kullback_leibler_divergence')
def kullback_leibler_divergence(y_true, y_pred):
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1)
  return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)


@tf_export('keras.metrics.poisson', 'keras.losses.poisson')
def poisson(y_true, y_pred):
  return K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)


@tf_export('keras.metrics.cosine_proximity', 'keras.losses.cosine_proximity')
def cosine_proximity(y_true, y_pred):
  y_true = nn.l2_normalize(y_true, axis=-1)
  y_pred = nn.l2_normalize(y_pred, axis=-1)
  return -math_ops.reduce_sum(y_true * y_pred, axis=-1)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


@tf_export('keras.losses.serialize')
def serialize(loss):
  return serialize_keras_object(loss)


@tf_export('keras.losses.deserialize')
def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='loss function')


@tf_export('keras.losses.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'loss function identifier:', identifier)
