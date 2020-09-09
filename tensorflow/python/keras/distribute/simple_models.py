# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""A simple functional keras model with one layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.distribute import model_collection_base
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.module import module
from tensorflow.python.ops import variables

_BATCH_SIZE = 10


def _get_data_for_simple_models():
  x_train = constant_op.constant(np.random.rand(1000, 3), dtype=dtypes.float32)
  y_train = constant_op.constant(np.random.rand(1000, 5), dtype=dtypes.float32)
  x_predict = constant_op.constant(
      np.random.rand(1000, 3), dtype=dtypes.float32)

  return x_train, y_train, x_predict


class SimpleFunctionalModel(model_collection_base.ModelAndInput):
  """A simple functional model and its inputs."""

  def get_model(self, **kwargs):
    output_name = 'output_1'

    x = keras.layers.Input(shape=(3,), dtype=dtypes.float32)
    y = keras.layers.Dense(5, dtype=dtypes.float32, name=output_name)(x)

    model = keras.Model(inputs=x, outputs=y)
    optimizer = gradient_descent.SGD(learning_rate=0.001)
    model.compile(
        loss='mse',
        metrics=['mae'],
        optimizer=optimizer)

    return model

  def get_data(self):
    return _get_data_for_simple_models()

  def get_batch_size(self):
    return _BATCH_SIZE


class SimpleSequentialModel(model_collection_base.ModelAndInput):
  """A simple sequential model and its inputs."""

  def get_model(self, **kwargs):
    output_name = 'output_1'

    model = keras.Sequential()
    y = keras.layers.Dense(
        5, dtype=dtypes.float32, name=output_name, input_dim=3)
    model.add(y)
    optimizer = gradient_descent.SGD(learning_rate=0.001)
    model.compile(
        loss='mse',
        metrics=['mae'],
        optimizer=optimizer)

    return model

  def get_data(self):
    return _get_data_for_simple_models()

  def get_batch_size(self):
    return _BATCH_SIZE


class _SimpleModel(keras.Model):

  def __init__(self):
    super(_SimpleModel, self).__init__()
    self._dense_layer = keras.layers.Dense(5, dtype=dtypes.float32)

  def call(self, inputs):
    return self._dense_layer(inputs)


class SimpleSubclassModel(model_collection_base.ModelAndInput):
  """A simple subclass model and its data."""

  def get_model(self, **kwargs):
    model = _SimpleModel()
    optimizer = gradient_descent.SGD(learning_rate=0.001)
    model.compile(
        loss='mse',
        metrics=['mae'],
        cloning=False,
        optimizer=optimizer)

    return model

  def get_data(self):
    return _get_data_for_simple_models()

  def get_batch_size(self):
    return _BATCH_SIZE


class _SimpleModule(module.Module):

  def __init__(self):
    self.v = variables.Variable(3.0)

  @def_function.function
  def __call__(self, x):
    return self.v * x


class SimpleTFModuleModel(model_collection_base.ModelAndInput):
  """A simple model based on tf.Module and its data."""

  def get_model(self, **kwargs):
    model = _SimpleModule()
    return model

  def get_data(self):
    return _get_data_for_simple_models()

  def get_batch_size(self):
    return _BATCH_SIZE
