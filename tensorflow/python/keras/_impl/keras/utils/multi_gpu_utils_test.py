# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multi-gpu training utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test


class TestMultiGPUModel(test.TestCase):

  def multi_gpu_test_simple_model(self):
    gpus = 2
    num_samples = 1000
    input_dim = 10
    output_dim = 1
    hidden_dim = 10
    epochs = 2
    target_gpu_id = [0, 2, 4]

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(hidden_dim,
                                   input_shape=(input_dim,)))
      model.add(keras.layers.Dense(output_dim))

      x = np.random.random((num_samples, input_dim))
      y = np.random.random((num_samples, output_dim))

      parallel_model = keras.utils.multi_gpu_model(model, gpus=gpus)
      parallel_model.compile(loss='mse', optimizer='rmsprop')
      parallel_model.fit(x, y, epochs=epochs)

      parallel_model = keras.utils.multi_gpu_model(model, gpus=target_gpu_id)
      parallel_model.compile(loss='mse', optimizer='rmsprop')
      parallel_model.fit(x, y, epochs=epochs)

  def multi_gpu_test_multi_io_model(self):
    gpus = 2
    num_samples = 1000
    input_dim_a = 10
    input_dim_b = 5
    output_dim_a = 1
    output_dim_b = 2
    hidden_dim = 10
    epochs = 2
    target_gpu_id = [0, 2, 4]

    with self.test_session():
      input_a = keras.Input((input_dim_a,))
      input_b = keras.Input((input_dim_b,))
      a = keras.layers.Dense(hidden_dim)(input_a)
      b = keras.layers.Dense(hidden_dim)(input_b)
      c = keras.layers.concatenate([a, b])
      output_a = keras.layers.Dense(output_dim_a)(c)
      output_b = keras.layers.Dense(output_dim_b)(c)
      model = keras.models.Model([input_a, input_b], [output_a, output_b])

      a_x = np.random.random((num_samples, input_dim_a))
      b_x = np.random.random((num_samples, input_dim_b))
      a_y = np.random.random((num_samples, output_dim_a))
      b_y = np.random.random((num_samples, output_dim_b))

      parallel_model = keras.utils.multi_gpu_model(model, gpus=gpus)
      parallel_model.compile(loss='mse', optimizer='rmsprop')
      parallel_model.fit([a_x, b_x], [a_y, b_y], epochs=epochs)

      parallel_model = keras.utils.multi_gpu_model(model, gpus=target_gpu_id)
      parallel_model.compile(loss='mse', optimizer='rmsprop')
      parallel_model.fit([a_x, b_x], [a_y, b_y], epochs=epochs)

  def multi_gpu_test_invalid_devices(self):
    with self.test_session():
      input_shape = (1000, 10)
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(10,
                                   activation='relu',
                                   input_shape=input_shape[1:]))
      model.add(keras.layers.Dense(1, activation='sigmoid'))
      model.compile(loss='mse', optimizer='rmsprop')

      x = np.random.random(input_shape)
      y = np.random.random((input_shape[0], 1))
      with self.assertRaises(ValueError):
        parallel_model = keras.utils.multi_gpu_model(
            model, gpus=len(keras.backend._get_available_gpus()) + 1)
        parallel_model.fit(x, y, epochs=2)

      with self.assertRaises(ValueError):
        parallel_model = keras.utils.multi_gpu_model(
            model, gpus=[0, 2, 4, 6, 8])
        parallel_model.fit(x, y, epochs=2)

      with self.assertRaises(ValueError):
        parallel_model = keras.utils.multi_gpu_model(model, gpus=1)
        parallel_model.fit(x, y, epochs=2)

      with self.assertRaises(ValueError):
        parallel_model = keras.utils.multi_gpu_model(model, gpus=[0])
        parallel_model.fit(x, y, epochs=2)
