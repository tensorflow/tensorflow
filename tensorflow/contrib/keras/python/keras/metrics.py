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
"""Built-in Keras metrics functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib.keras.python.keras import backend as K
# pylint: disable=unused-import
from tensorflow.contrib.keras.python.keras.losses import binary_crossentropy
from tensorflow.contrib.keras.python.keras.losses import categorical_crossentropy
from tensorflow.contrib.keras.python.keras.losses import cosine_proximity
from tensorflow.contrib.keras.python.keras.losses import hinge
from tensorflow.contrib.keras.python.keras.losses import kullback_leibler_divergence
from tensorflow.contrib.keras.python.keras.losses import logcosh
from tensorflow.contrib.keras.python.keras.losses import mean_absolute_error
from tensorflow.contrib.keras.python.keras.losses import mean_absolute_percentage_error
from tensorflow.contrib.keras.python.keras.losses import mean_squared_error
from tensorflow.contrib.keras.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.contrib.keras.python.keras.losses import poisson
from tensorflow.contrib.keras.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.contrib.keras.python.keras.losses import squared_hinge
# pylint: disable=unused-import
from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object


def binary_accuracy(y_true, y_pred):
  return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
  return K.cast(
      K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
  return K.cast(
      K.equal(
          K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1),
                                         K.floatx())), K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(K.in_top_k(y_pred,
                           K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def serialize(metric):
  return metric.__name__


def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='metric function')


def get(identifier):
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'metric function identifier:', identifier)
