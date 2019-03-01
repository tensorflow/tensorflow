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
"""Tests metrics correctness using Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_with_all_model_types(exclude_models=['sequential'])
@keras_parameterized.run_all_keras_modes
class TestMetricsCorrectnessMultiIO(keras_parameterized.TestCase):

  def _get_multi_io_model(self):
    inp_1 = layers.Input(shape=(1,), name='input_1')
    inp_2 = layers.Input(shape=(1,), name='input_2')
    x = layers.Dense(3, kernel_initializer='ones', trainable=False)
    out_1 = layers.Dense(
        1, kernel_initializer='ones', name='output_1', trainable=False)
    out_2 = layers.Dense(
        1, kernel_initializer='ones', name='output_2', trainable=False)

    branch_a = [inp_1, x, out_1]
    branch_b = [inp_2, x, out_2]
    model = testing_utils.get_multi_io_model(branch_a, branch_b)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[metrics.MeanSquaredError(name='mean_squared_error')],
        weighted_metrics=[
            metrics.MeanSquaredError(name='mean_squared_error_2')
        ],
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def _custom_generator(self):
    batch_size = 2
    num_samples = 4
    inputs = np.asarray([[1.], [2.], [3.], [4.]])
    targets = np.asarray([[2.], [4.], [6.], [8.]])
    w1 = np.asarray([2., 3., 4., 5.])
    w2 = np.asarray([3.5, 2.5, 1.5, 0.5])
    i = 0
    while True:
      batch_index = i * batch_size % num_samples
      i += 1
      start = batch_index
      end = start + batch_size
      x = [inputs[start:end], inputs[start:end]]
      y = [targets[start:end], targets[start:end]]
      w = [w1[start:end], w2[start:end]]
      yield x, y, w

  def setUp(self):
    super(TestMetricsCorrectnessMultiIO, self).setUp()
    self.x = np.asarray([[1.], [2.], [3.], [4.]])
    self.y = np.asarray([[2.], [4.], [6.], [8.]])
    self.weights_1 = np.asarray([2., 3., 4., 5.])
    self.weights_2 = np.asarray([3.5, 2.5, 1.5, 0.5])

    # y_true = [[2.], [4.], [6.], [8.]], y_pred = [[3.], [6.], [9.], [12.]]

    # Metric `output_1`, `output_2`:
    #   Total = ((3 - 2)^2 + (6 - 4)^2) + ((9 - 6)^2 + (12 - 8)^2) = 30,
    #   Count = 2 + 2
    #   Result = 7.5

    # Weighted metric `output_1`:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
    #         = 130
    #   Count = (2 + 3) + (4 + 5)
    #   Result = 9.2857141

    # Weighted metric `output_2`:
    #   Total = ((3 - 2)^2 * 3.5 + (6 - 4)^2 * 2.5) +
    #           ((9 - 6)^2 * 1.5 + (12 - 8)^2 * 0.5)
    #         = 35
    #   Count = (3.5 + 2.5) + (1.5 + 0.5)
    #   Result = 4.375

    # Loss `output_1`:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
    #         = 130
    #   Count = 2 + 2
    #   Result = 32.5

    # Loss `output_2`:
    #   Total = ((3 - 2)^2 * 3.5 + (6 - 4)^2 * 2.5) +
    #           ((9 - 6)^2 * 1.5 + (12 - 8)^2 * 0.5)
    #         = 35
    #   Count = 2 + 2
    #   Result = 8.75

    # Total loss = 32.5 + 8.75 = 41.25

    wmse = 'mean_squared_error_2'
    if not tf2.enabled():
      wmse = 'weighted_' + wmse
    self.expected_fit_result = {
        'output_1_mean_squared_error': [7.5, 7.5],
        'output_2_mean_squared_error': [7.5, 7.5],
        'output_1_' + wmse: [9.286, 9.286],
        'output_2_' + wmse: [4.375, 4.375],
        'loss': [41.25, 41.25],
        'output_1_loss': [32.5, 32.5],
        'output_2_loss': [8.75, 8.75],
    }

    # In the order: 'loss', 'output_1_loss', 'output_2_loss',
    # 'output_1_mean_squared_error', 'output_1_mean_squared_error_2',
    # 'output_2_mean_squared_error', 'output_2_mean_squared_error_2'
    self.expected_batch_result = [41.25, 32.5, 8.75, 7.5, 9.286, 7.5, 4.375]

  def test_fit(self):
    model = self._get_multi_io_model()
    history = model.fit([self.x, self.x], [self.y, self.y],
                        sample_weight={
                            'output_1': self.weights_1,
                            'output_2': self.weights_2,
                        },
                        batch_size=2,
                        epochs=2,
                        shuffle=False)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval(self):
    model = self._get_multi_io_model()
    eval_result = model.evaluate([self.x, self.x], [self.y, self.y],
                                 batch_size=2,
                                 sample_weight={
                                     'output_1': self.weights_1,
                                     'output_2': self.weights_2,
                                 })
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

    # Verify that metric value is same with arbitrary weights and batch size.
    x = np.random.random((50, 1))
    y = np.random.random((50, 1))
    w = np.random.random((50,))
    mse1 = model.evaluate([x, x], [y, y], sample_weight=[w, w], batch_size=5)[3]
    mse2 = model.evaluate([x, x], [y, y], sample_weight=[w, w],
                          batch_size=10)[3]
    self.assertAllClose(mse1, mse2, 1e-3)

  def test_train_on_batch(self):
    model = self._get_multi_io_model()
    result = model.train_on_batch([self.x, self.x], [self.y, self.y],
                                  sample_weight={
                                      'output_1': self.weights_1,
                                      'output_2': self.weights_2,
                                  })
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_test_on_batch(self):
    model = self._get_multi_io_model()
    result = model.test_on_batch([self.x, self.x], [self.y, self.y],
                                 sample_weight={
                                     'output_1': self.weights_1,
                                     'output_2': self.weights_2,
                                 })
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_fit_generator(self):
    model = self._get_multi_io_model()
    history = model.fit_generator(
        self._custom_generator(), steps_per_epoch=2, epochs=2)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval_generator(self):
    model = self._get_multi_io_model()
    eval_result = model.evaluate_generator(self._custom_generator(), steps=2)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class TestMetricsCorrectnessSingleIO(keras_parameterized.TestCase):

  def _get_model(self):
    x = layers.Dense(3, kernel_initializer='ones', trainable=False)
    out = layers.Dense(
        1, kernel_initializer='ones', name='output', trainable=False)
    model = testing_utils.get_model_from_layers([x, out], input_shape=(1,))
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[metrics.MeanSquaredError(name='mean_squared_error')],
        weighted_metrics=[
            metrics.MeanSquaredError(name='mean_squared_error_2')
        ],
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def _custom_generator(self):
    batch_size = 2
    num_samples = 4
    x = np.asarray([[1.], [2.], [3.], [4.]])
    y = np.asarray([[2.], [4.], [6.], [8.]])
    w = np.asarray([2., 3., 4., 5.])
    i = 0
    while True:
      batch_index = i * batch_size % num_samples
      i += 1
      start = batch_index
      end = start + batch_size
      yield x[start:end], y[start:end], w[start:end]

  def setUp(self):
    super(TestMetricsCorrectnessSingleIO, self).setUp()
    self.x = np.asarray([[1.], [2.], [3.], [4.]])
    self.y = np.asarray([[2.], [4.], [6.], [8.]])
    self.weights = np.asarray([2., 3., 4., 5.])

    # y_true = [[2.], [4.], [6.], [8.]], y_pred = [[3.], [6.], [9.], [12.]]

    # Metric:
    #   Total = ((3 - 2)^2 + (6 - 4)^2) + ((9 - 6)^2 + (12 - 8)^2) = 30,
    #   Count = 2 + 2
    #   Result = 7.5

    # Weighted metric:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
    #         = 130
    #   Count = (2 + 3) + (4 + 5)
    #   Result = 9.2857141

    # Total loss:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
    #         = 130,
    #   Count = 2 + 2
    #   Result = 32.5

    wmse = 'mean_squared_error_2'
    if not tf2.enabled():
      wmse = 'weighted_' + wmse
    self.expected_fit_result = {
        'mean_squared_error': [7.5, 7.5],
        wmse: [9.286, 9.286],
        'loss': [32.5, 32.5]
    }

    # In the order: 'loss', 'mean_squared_error', 'mean_squared_error_2'
    self.expected_batch_result = [32.5, 7.5, 9.286]

  def test_fit(self):
    model = self._get_model()
    history = model.fit(
        self.x,
        self.y,
        sample_weight=self.weights,
        batch_size=2,
        epochs=2,
        shuffle=False)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval(self):
    model = self._get_model()
    eval_result = model.evaluate(
        self.x, self.y, batch_size=2, sample_weight=self.weights)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

    # Verify that metric value is same with arbitrary weights and batch size.
    x = np.random.random((50, 1))
    y = np.random.random((50, 1))
    w = np.random.random((50,))
    mse1 = model.evaluate(x, y, sample_weight=w, batch_size=5)[1]
    mse2 = model.evaluate(x, y, sample_weight=w, batch_size=10)[1]
    self.assertAllClose(mse1, mse2, 1e-3)

  def test_train_on_batch(self):
    model = self._get_model()
    result = model.train_on_batch(self.x, self.y, sample_weight=self.weights)
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_test_on_batch(self):
    model = self._get_model()
    result = model.test_on_batch(self.x, self.y, sample_weight=self.weights)
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_fit_generator(self):
    model = self._get_model()
    history = model.fit_generator(
        self._custom_generator(), steps_per_epoch=2, epochs=2)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval_generator(self):
    model = self._get_model()
    eval_result = model.evaluate_generator(self._custom_generator(), steps=2)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)


if __name__ == '__main__':
  test.main()
