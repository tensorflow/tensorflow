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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops.losses import loss_reduction
from tensorflow.python.platform import test
from tensorflow.python.util import nest


def get_multi_io_model():
  inp_1 = layers.Input(shape=(1,), name='input_1')
  inp_2 = layers.Input(shape=(1,), name='input_2')
  x = layers.Dense(3, kernel_initializer='ones', trainable=False)
  out_1 = layers.Dense(
      1, kernel_initializer='ones', name='output_1', trainable=False)
  out_2 = layers.Dense(
      1, kernel_initializer='ones', name='output_2', trainable=False)

  branch_a = [inp_1, x, out_1]
  branch_b = [inp_2, x, out_2]
  return testing_utils.get_multi_io_model(branch_a, branch_b)


def custom_generator_multi_io(sample_weights=None):
  batch_size = 2
  num_samples = 4
  inputs = np.asarray([[1.], [2.], [3.], [4.]])
  targets_1 = np.asarray([[2.], [4.], [6.], [8.]])
  targets_2 = np.asarray([[1.], [2.], [3.], [4.]])
  i = 0
  while True:
    batch_index = i * batch_size % num_samples
    i += 1
    start = batch_index
    end = start + batch_size
    x = [inputs[start:end], inputs[start:end]]
    y = [targets_1[start:end], targets_2[start:end]]
    if sample_weights:
      sw = nest.map_structure(lambda w: w[start:end], sample_weights)
    else:
      sw = None
    yield x, y, sw


@keras_parameterized.run_with_all_model_types(exclude_models=['sequential'])
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class TestMetricsCorrectnessMultiIO(keras_parameterized.TestCase):

  def _get_compiled_multi_io_model(self):
    model = get_multi_io_model()
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=[metrics.MeanSquaredError(name='mean_squared_error')],
        weighted_metrics=[
            metrics.MeanSquaredError(name='mean_squared_error_2')
        ],
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def setUp(self):
    super(TestMetricsCorrectnessMultiIO, self).setUp()
    self.x = np.asarray([[1.], [2.], [3.], [4.]])
    self.y1 = np.asarray([[2.], [4.], [6.], [8.]])
    self.y2 = np.asarray([[1.], [2.], [3.], [4.]])
    self.sample_weight_1 = np.asarray([2., 3., 4., 5.])
    self.sample_weight_2 = np.asarray([3.5, 2.5, 1.5, 0.5])

    # y_true_1 = [[2.], [4.], [6.], [8.]], y_pred = [[3.], [6.], [9.], [12.]]
    # y_true_2 = [[1.], [2.], [3.], [4.]], y_pred = [[3.], [6.], [9.], [12.]]

    # Weighted metric `output_1`:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
    #         = 130
    #   Count = (2 + 3) + (4 + 5)
    #   Result = 9.2857141

    # Weighted metric `output_2`:
    #   Total = ((3 - 1)^2 * 3.5 + (6 - 2)^2 * 2.5) +
    #           ((9 - 3)^2 * 1.5 + (12 - 4)^2 * 0.5)
    #         = 140
    #   Count = (3.5 + 2.5) + (1.5 + 0.5)
    #   Result = 17.5

    # Loss `output_1` with weights:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
    #         = 130
    #   Count = 2 + 2
    #   Result = 32.5

    # Loss `output_1` without weights/Metric `output_1`:
    #   Total = ((3 - 2)^2 + (6 - 4)^2) + ((9 - 6)^2 + (12 - 8)^2) = 30
    #   Count = 2 + 2
    #   Result = 7.5

    # Loss `output_2` with weights:
    #   Total = ((3 - 1)^2 * 3.5 + (6 - 2)^2 * 2.5) +
    #           ((9 - 3)^2 * 1.5 + (12 - 4)^2 * 0.5)
    #         = 140
    #   Count = 2 + 2
    #   Result = 35

    # Loss `output_2` without weights/Metric `output_2`:
    #   Total = ((3 - 1)^2 + (6 - 2)^2) + ((9 - 3)^2 + (12 - 4)^2) = 120
    #   Count = 2 + 2
    #   Result = 30

    # Total loss with weights = 32.5 + 35 = 67.5
    # Total loss without weights = 7.5 + 30 = 37.5

    self.wmse = 'mean_squared_error_2'
    self.expected_fit_result_with_weights = {
        'output_1_mean_squared_error': [7.5, 7.5],
        'output_2_mean_squared_error': [30, 30],
        'output_1_' + self.wmse: [9.286, 9.286],
        'output_2_' + self.wmse: [17.5, 17.5],
        'loss': [67.5, 67.5],
        'output_1_loss': [32.5, 32.5],
        'output_2_loss': [35, 35],
    }

    self.expected_fit_result_with_weights_output_2 = {
        'output_1_mean_squared_error': [7.5, 7.5],
        'output_2_mean_squared_error': [30, 30],
        'output_1_' + self.wmse: [7.5, 7.5],
        'output_2_' + self.wmse: [17.5, 17.5],
        'loss': [42.5, 42.5],
        'output_1_loss': [7.5, 7.5],
        'output_2_loss': [35, 35],
    }

    self.expected_fit_result = {
        'output_1_mean_squared_error': [7.5, 7.5],
        'output_2_mean_squared_error': [30, 30],
        'output_1_' + self.wmse: [7.5, 7.5],
        'output_2_' + self.wmse: [30, 30],
        'loss': [37.5, 37.5],
        'output_1_loss': [7.5, 7.5],
        'output_2_loss': [30, 30],
    }

    # In the order: 'loss', 'output_1_loss', 'output_2_loss',
    # 'output_1_mean_squared_error', 'output_1_mean_squared_error_2',
    # 'output_2_mean_squared_error', 'output_2_mean_squared_error_2'
    self.expected_batch_result_with_weights = [
        67.5, 32.5, 35, 7.5, 9.286, 30, 17.5
    ]
    self.expected_batch_result_with_weights_output_2 = [
        42.5, 7.5, 35, 7.5, 7.5, 30, 17.5
    ]
    self.expected_batch_result = [37.5, 7.5, 30, 7.5, 7.5, 30, 30]

  def test_fit(self):
    model = self._get_compiled_multi_io_model()
    history = model.fit([self.x, self.x], [self.y1, self.y2],
                        batch_size=2,
                        epochs=2,
                        shuffle=False)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_fit_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    history = model.fit([self.x, self.x], [self.y1, self.y2],
                        sample_weight={
                            'output_1': self.sample_weight_1,
                            'output_2': self.sample_weight_2,
                        },
                        batch_size=2,
                        epochs=2,
                        shuffle=False)
    for key, value in self.expected_fit_result_with_weights.items():
      self.assertAllClose(history.history[key], value, 1e-3)

    # Set weights for one output (use batch size).
    history = model.fit([self.x, self.x], [self.y1, self.y2],
                        sample_weight={'output_2': self.sample_weight_2},
                        batch_size=2,
                        epochs=2,
                        shuffle=False)

    for key, value in self.expected_fit_result_with_weights_output_2.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval(self):
    model = self._get_compiled_multi_io_model()
    eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                 batch_size=2)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

  def test_eval_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                 batch_size=2,
                                 sample_weight={
                                     'output_1': self.sample_weight_1,
                                     'output_2': self.sample_weight_2,
                                 })
    self.assertAllClose(eval_result, self.expected_batch_result_with_weights,
                        1e-3)

    # Set weights for one output.
    model = self._get_compiled_multi_io_model()
    eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                 batch_size=2,
                                 sample_weight={
                                     'output_2': self.sample_weight_2,
                                 })
    self.assertAllClose(eval_result,
                        self.expected_batch_result_with_weights_output_2, 1e-3)

    # Verify that metric value is same with arbitrary weights and batch size.
    x = np.random.random((50, 1))
    y = np.random.random((50, 1))
    w = np.random.random((50,))
    mse1 = model.evaluate([x, x], [y, y], sample_weight=[w, w], batch_size=5)[3]
    mse2 = model.evaluate([x, x], [y, y], sample_weight=[w, w],
                          batch_size=10)[3]
    self.assertAllClose(mse1, mse2, 1e-3)

  def test_train_on_batch(self):
    model = self._get_compiled_multi_io_model()
    result = model.train_on_batch([self.x, self.x], [self.y1, self.y2])
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_train_on_batch_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                  sample_weight={
                                      'output_1': self.sample_weight_1,
                                      'output_2': self.sample_weight_2,
                                  })
    self.assertAllClose(result, self.expected_batch_result_with_weights, 1e-3)

    # Set weights for one output.
    result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                  sample_weight={
                                      'output_2': self.sample_weight_2,
                                  })
    self.assertAllClose(result,
                        self.expected_batch_result_with_weights_output_2, 1e-3)

  def test_test_on_batch(self):
    model = self._get_compiled_multi_io_model()
    result = model.test_on_batch([self.x, self.x], [self.y1, self.y2])
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_test_on_batch_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    result = model.test_on_batch([self.x, self.x], [self.y1, self.y2],
                                 sample_weight={
                                     'output_1': self.sample_weight_1,
                                     'output_2': self.sample_weight_2,
                                 })
    self.assertAllClose(result, self.expected_batch_result_with_weights, 1e-3)

    # Set weights for one output.
    result = model.test_on_batch([self.x, self.x], [self.y1, self.y2],
                                 sample_weight={
                                     'output_2': self.sample_weight_2,
                                 })
    self.assertAllClose(result,
                        self.expected_batch_result_with_weights_output_2, 1e-3)

  def test_fit_generator(self):
    model = self._get_compiled_multi_io_model()
    history = model.fit_generator(
        custom_generator_multi_io(), steps_per_epoch=2, epochs=2)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_fit_generator_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    history = model.fit_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps_per_epoch=2,
        epochs=2)
    for key, value in self.expected_fit_result_with_weights.items():
      self.assertAllClose(history.history[key], value, 1e-3)

    # Set weights for one output.
    history = model.fit_generator(
        custom_generator_multi_io(
            sample_weights={'output_2': self.sample_weight_2}),
        steps_per_epoch=2,
        epochs=2)
    for key, value in self.expected_fit_result_with_weights_output_2.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval_generator(self):
    model = self._get_compiled_multi_io_model()
    eval_result = model.evaluate_generator(custom_generator_multi_io(), steps=2)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

  def test_eval_generator_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    eval_result = model.evaluate_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps=2)
    self.assertAllClose(eval_result, self.expected_batch_result_with_weights,
                        1e-3)

    # Set weights for one output.
    eval_result = model.evaluate_generator(
        custom_generator_multi_io(
            sample_weights={'output_2': self.sample_weight_2}),
        steps=2)
    self.assertAllClose(eval_result,
                        self.expected_batch_result_with_weights_output_2, 1e-3)


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
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

  def _custom_generator(self, sample_weight=None):
    batch_size = 2
    num_samples = 4
    x = np.asarray([[1.], [2.], [3.], [4.]])
    y = np.asarray([[2.], [4.], [6.], [8.]])
    w = sample_weight
    i = 0

    while True:
      batch_index = i * batch_size % num_samples
      i += 1
      start = batch_index
      end = start + batch_size
      yield x[start:end], y[start:end], None if w is None else w[start:end]

  def setUp(self):
    super(TestMetricsCorrectnessSingleIO, self).setUp()
    self.x = np.asarray([[1.], [2.], [3.], [4.]])
    self.y = np.asarray([[2.], [4.], [6.], [8.]])
    self.sample_weight = np.asarray([2., 3., 4., 5.])
    self.class_weight = {i: 1 for i in range(10)}
    self.class_weight.update({2: 2, 4: 3, 6: 4, 8: 5})

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

    # Total loss with weights:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5)
    #         = 130,
    #   Count = 2 + 2
    #   Result = 32.5

    # Total loss without weights:
    #   Total = ((3 - 2)^2 + (6 - 4)^2) +
    #           ((9 - 6)^2 + (12 - 8)^2)
    #         = 30,
    #   Count = 2 + 2
    #   Result = 7.5

    wmse = 'mean_squared_error_2'

    self.expected_fit_result_with_weights = {
        'mean_squared_error': [7.5, 7.5],
        wmse: [9.286, 9.286],
        'loss': [32.5, 32.5]
    }

    self.expected_fit_result = {
        'mean_squared_error': [7.5, 7.5],
        wmse: [7.5, 7.5],
        'loss': [7.5, 7.5]
    }

    # In the order: 'loss', 'mean_squared_error', 'mean_squared_error_2'
    self.expected_batch_result_with_weights = [32.5, 7.5, 9.286]
    self.expected_batch_result = [7.5, 7.5, 7.5]

  def test_fit(self):
    model = self._get_model()

    history = model.fit(
        self.x,
        self.y,
        batch_size=2,
        epochs=2,
        shuffle=False)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_fit_with_sample_weight(self):
    model = self._get_model()
    history = model.fit(
        self.x,
        self.y,
        sample_weight=self.sample_weight,
        batch_size=2,
        epochs=2,
        shuffle=False)
    for key, value in self.expected_fit_result_with_weights.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_fit_with_class_weight(self):
    model = self._get_model()
    history = model.fit(
        self.x,
        self.y,
        class_weight=self.class_weight,
        batch_size=2,
        epochs=2,
        shuffle=False)
    for key, value in self.expected_fit_result_with_weights.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval(self):
    model = self._get_model()
    eval_result = model.evaluate(self.x, self.y, batch_size=2)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

  def test_eval_with_sample_weight(self):
    model = self._get_model()
    eval_result = model.evaluate(
        self.x, self.y, batch_size=2, sample_weight=self.sample_weight)
    self.assertAllClose(eval_result, self.expected_batch_result_with_weights,
                        1e-3)

    # Verify that metric value is same with arbitrary weights and batch size.
    x = np.random.random((50, 1))
    y = np.random.random((50, 1))
    w = np.random.random((50,))
    mse1 = model.evaluate(x, y, sample_weight=w, batch_size=5)[1]
    mse2 = model.evaluate(x, y, sample_weight=w, batch_size=10)[1]
    self.assertAllClose(mse1, mse2, 1e-3)

  def test_train_on_batch(self):
    model = self._get_model()
    result = model.train_on_batch(self.x, self.y)
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_train_on_batch_with_sample_weight(self):
    model = self._get_model()
    result = model.train_on_batch(
        self.x, self.y, sample_weight=self.sample_weight)
    self.assertAllClose(result, self.expected_batch_result_with_weights, 1e-3)

  def test_train_on_batch_with_class_weight(self):
    model = self._get_model()
    result = model.train_on_batch(
        self.x, self.y, class_weight=self.class_weight)
    self.assertAllClose(result, self.expected_batch_result_with_weights, 1e-3)

  def test_test_on_batch(self):
    model = self._get_model()
    result = model.test_on_batch(self.x, self.y)
    self.assertAllClose(result, self.expected_batch_result, 1e-3)

  def test_test_on_batch_with_sample_weight(self):
    model = self._get_model()
    result = model.test_on_batch(
        self.x, self.y, sample_weight=self.sample_weight)
    self.assertAllClose(result, self.expected_batch_result_with_weights, 1e-3)

  def test_fit_generator(self):
    model = self._get_model()
    history = model.fit_generator(
        self._custom_generator(), steps_per_epoch=2, epochs=2)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_fit_generator_with_sample_weight(self):
    model = self._get_model()
    history = model.fit_generator(
        self._custom_generator(sample_weight=self.sample_weight),
        steps_per_epoch=2,
        epochs=2)
    for key, value in self.expected_fit_result_with_weights.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_fit_generator_with_class_weight(self):
    model = self._get_model()
    history = model.fit_generator(
        self._custom_generator(),
        steps_per_epoch=2,
        epochs=2,
        class_weight=self.class_weight)
    for key, value in self.expected_fit_result_with_weights.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval_generator(self):
    model = self._get_model()
    eval_result = model.evaluate_generator(self._custom_generator(), steps=2)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

  def test_eval_generator_with_sample_weight(self):
    model = self._get_model()
    eval_result = model.evaluate_generator(
        self._custom_generator(sample_weight=self.sample_weight), steps=2)
    self.assertAllClose(eval_result, self.expected_batch_result_with_weights,
                        1e-3)


@keras_parameterized.run_with_all_model_types(exclude_models=['sequential'])
@keras_parameterized.run_all_keras_modes
@parameterized.parameters([
    loss_reduction.ReductionV2.SUM_OVER_BATCH_SIZE,
    loss_reduction.ReductionV2.AUTO,
    loss_reduction.ReductionV2.SUM
])
class TestOutputLossMetrics(keras_parameterized.TestCase):

  def _get_compiled_multi_io_model(self, loss):
    model = get_multi_io_model()
    model.compile(
        optimizer='rmsprop',
        loss=loss,
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def setUp(self):
    super(TestOutputLossMetrics, self).setUp()
    self.x = np.asarray([[1.], [2.], [3.], [4.]])
    self.y1 = np.asarray([[2.], [4.], [6.], [8.]])
    self.y2 = np.asarray([[1.], [2.], [3.], [4.]])
    self.sample_weight_1 = np.asarray([2., 3., 4., 5.])
    self.sample_weight_2 = np.asarray([3.5, 2.5, 1.5, 0.5])

    # y_true = [[2.], [4.], [6.], [8.]], y_pred = [[3.], [6.], [9.], [12.]]

    # Loss `output_1`:
    #   Per-sample weighted losses
    #   Batch 1 = [(3 - 2)^2 * 2, (6 - 4)^2 * 3)] = [2, 12]
    #   Batch 2 = [((9 - 6)^2 * 4, (12 - 8)^2 * 5)] = [36, 80]

    #   Result (reduction=SUM) = ((2 + 12) + (36 + 80))/2 = 65
    #   Result (reduction=SUM_OVER_BATCH_SIZE/AUTO/NONE) = 130 / 4 = 32.5

    # Loss `output_2`:
    #   Per-sample weighted losses
    #   Batch 1 = [(3 - 1)^2 * 3.5, (6 - 2)^2 * 2.5)] = [14, 40]
    #   Batch 2 = [(9 - 3)^2 * 1.5, (12 - 4)^2 * 0.5)] = [54, 32]

    #   Result (reduction=SUM) = ((14 + 40) + (54 + 32))/2 = 70
    #   Result (reduction=SUM_OVER_BATCH_SIZE/AUTO/NONE) = 140 / 4 = 35

    # When reduction is 'NONE' loss value that is passed to the optimizer will
    # be vector loss but what is reported is a scalar, which is an average of
    # all the values in all the batch vectors.

    # Total loss = Output_loss_1 + Output_loss_2

    sum_over_batch_size_fit_result = {
        'loss': [67.5, 67.5],
        'output_1_loss': [32.5, 32.5],
        'output_2_loss': [35, 35],
    }

    self.expected_fit_result = {
        loss_reduction.ReductionV2.NONE:
            sum_over_batch_size_fit_result,
        loss_reduction.ReductionV2.SUM: {
            'loss': [135, 135],
            'output_1_loss': [65, 65],
            'output_2_loss': [70, 70],
        },
        loss_reduction.ReductionV2.AUTO:
            sum_over_batch_size_fit_result,
        loss_reduction.ReductionV2.SUM_OVER_BATCH_SIZE:
            sum_over_batch_size_fit_result,
    }

    # In the order: 'loss', 'output_1_loss', 'output_2_loss',
    self.expected_batch_result = {
        loss_reduction.ReductionV2.NONE: [67.5, 32.5, 35],
        loss_reduction.ReductionV2.SUM: [135, 65, 70],
        loss_reduction.ReductionV2.AUTO: [67.5, 32.5, 35],
        loss_reduction.ReductionV2.SUM_OVER_BATCH_SIZE: [67.5, 32.5, 35],
    }

  def test_fit(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    history = model.fit([self.x, self.x], [self.y1, self.y2],
                        sample_weight={
                            'output_1': self.sample_weight_1,
                            'output_2': self.sample_weight_2,
                        },
                        batch_size=2,
                        epochs=2,
                        shuffle=False)
    for key, value in self.expected_fit_result[reduction].items():
      self.assertAllClose(history.history[key], value)

  def test_eval(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                 batch_size=2,
                                 sample_weight={
                                     'output_1': self.sample_weight_1,
                                     'output_2': self.sample_weight_2,
                                 })
    self.assertAllClose(eval_result, self.expected_batch_result[reduction])

  def test_train_on_batch(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                  sample_weight={
                                      'output_1': self.sample_weight_1,
                                      'output_2': self.sample_weight_2,
                                  })

    expected_values = self.expected_batch_result[reduction]
    if reduction == loss_reduction.ReductionV2.SUM:
      # We are taking all the data as one batch, so undo the averaging here.
      expected_values = [x * 2 for x in self.expected_batch_result[reduction]]
    self.assertAllClose(result, expected_values)

  def test_test_on_batch(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    result = model.test_on_batch([self.x, self.x], [self.y1, self.y2],
                                 sample_weight={
                                     'output_1': self.sample_weight_1,
                                     'output_2': self.sample_weight_2,
                                 })
    expected_values = self.expected_batch_result[reduction]
    if reduction == loss_reduction.ReductionV2.SUM:
      # We are taking all the data as one batch, so undo the averaging here.
      expected_values = [x * 2 for x in self.expected_batch_result[reduction]]
    self.assertAllClose(result, expected_values)

  def test_fit_generator(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    history = model.fit_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps_per_epoch=2,
        epochs=2)
    for key, value in self.expected_fit_result[reduction].items():
      self.assertAllClose(history.history[key], value)

  def test_eval_generator(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    eval_result = model.evaluate_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps=2)
    self.assertAllClose(eval_result, self.expected_batch_result[reduction])


if __name__ == '__main__':
  test.main()
