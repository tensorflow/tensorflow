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

from tensorflow.python import data
from tensorflow.python import keras
from tensorflow.python.platform import test


def check_if_compatible_devices(gpus=2):
  available_devices = [
      keras.utils.multi_gpu_utils._normalize_device_name(name)
      for name in keras.utils.multi_gpu_utils._get_available_devices()
  ]
  if '/gpu:%d' % (gpus - 1) not in available_devices:
    return False
  return True


class TestMultiGPUModel(test.TestCase):

  def test_multi_gpu_test_simple_model(self):
    gpus = 2
    num_samples = 1000
    input_dim = 10
    output_dim = 1
    hidden_dim = 10
    epochs = 2
    target_gpu_id = [0, 1]

    if not check_if_compatible_devices(gpus=gpus):
      return

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

  def test_multi_gpu_test_multi_io_model(self):
    gpus = 2
    num_samples = 1000
    input_dim_a = 10
    input_dim_b = 5
    output_dim_a = 1
    output_dim_b = 2
    hidden_dim = 10
    epochs = 2
    target_gpu_id = [0, 1]

    if not check_if_compatible_devices(gpus=gpus):
      return

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

  def test_multi_gpu_test_invalid_devices(self):
    if not check_if_compatible_devices(gpus=2):
      return

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

  def test_nested_model_with_tensor_input(self):
    gpus = 2
    input_dim = 10
    shape = (input_dim,)
    num_samples = 16
    num_classes = 10

    if not check_if_compatible_devices(gpus=gpus):
      return

    with self.test_session():
      input_shape = (num_samples,) + shape
      x_train = np.random.randint(0, 255, input_shape)
      y_train = np.random.randint(0, num_classes, (input_shape[0],))
      keras.backend.set_learning_phase(True)

      y_train = keras.utils.to_categorical(y_train, num_classes)

      x_train = x_train.astype('float32')
      y_train = y_train.astype('float32')

      dataset = data.Dataset.from_tensor_slices((x_train, y_train))
      dataset = dataset.repeat()
      dataset = dataset.batch(4)
      iterator = dataset.make_one_shot_iterator()

      inputs, targets = iterator.get_next()

      input_tensor = keras.layers.Input(tensor=inputs)

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(3,
                                   input_shape=(input_dim,)))
      model.add(keras.layers.Dense(num_classes))

      output = model(input_tensor)
      outer_model = keras.Model(input_tensor, output)
      parallel_model = keras.utils.multi_gpu_model(outer_model, gpus=gpus)

      parallel_model.compile(
          loss='categorical_crossentropy',
          optimizer=keras.optimizers.RMSprop(lr=0.0001, decay=1e-6),
          metrics=['accuracy'],
          target_tensors=[targets])
      parallel_model.fit(epochs=1, steps_per_epoch=3)


if __name__ == '__main__':
  test.main()
