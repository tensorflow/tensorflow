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
"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.losses import binary_crossentropy
from tensorflow.python.keras._impl.keras.losses import categorical_crossentropy
from tensorflow.python.keras._impl.keras.losses import cosine_proximity
from tensorflow.python.keras._impl.keras.losses import hinge
from tensorflow.python.keras._impl.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras._impl.keras.losses import logcosh
from tensorflow.python.keras._impl.keras.losses import mean_absolute_error
from tensorflow.python.keras._impl.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras._impl.keras.losses import mean_squared_error
from tensorflow.python.keras._impl.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras._impl.keras.losses import poisson
from tensorflow.python.keras._impl.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras._impl.keras.losses import squared_hinge
from tensorflow.python.keras._impl.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras._impl.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.metrics.binary_accuracy')
def binary_accuracy(y_true, y_pred):
  return K.mean(math_ops.equal(y_true, math_ops.round(y_pred)), axis=-1)


@tf_export('keras.metrics.categorical_accuracy')
def categorical_accuracy(y_true, y_pred):
  return math_ops.cast(
      math_ops.equal(
          math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
      K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
  return math_ops.cast(
      math_ops.equal(
          math_ops.reduce_max(y_true, axis=-1),
          math_ops.cast(math_ops.argmax(y_pred, axis=-1), K.floatx())),
      K.floatx())


@tf_export('keras.metrics.top_k_categorical_accuracy')
def top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(
      nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), axis=-1)


@tf_export('keras.metrics.sparse_top_k_categorical_accuracy')
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(
      nn.in_top_k(y_pred,
                  math_ops.cast(math_ops.reduce_max(y_true, axis=-1), 'int32'),
                  k),
      axis=-1)

# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


@tf_export('keras.metrics.serialize')
def serialize(metric):
  return serialize_keras_object(metric)


@tf_export('keras.metrics.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='metric function')


@tf_export('keras.metrics.get')
def get(identifier):
  if isinstance(identifier, dict):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif isinstance(identifier, six.string_types):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'metric function identifier: %s' % identifier)
