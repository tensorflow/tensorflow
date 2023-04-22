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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.utils import losses_utils
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
  num_samples = 5
  inputs = np.asarray([[1.], [2.], [3.], [4.], [5.]])
  targets_1 = np.asarray([[2.], [4.], [6.], [8.], [10.]])
  targets_2 = np.asarray([[1.], [2.], [3.], [4.], [5.]])
  start = 0
  while True:
    if start > num_samples:
      start = 0
    end = start + batch_size
    x = [inputs[start:end], inputs[start:end]]
    y = [targets_1[start:end], targets_2[start:end]]
    if sample_weights:
      sw = nest.map_structure(lambda w: w[start:end], sample_weights)
    else:
      sw = None
    start = end
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
    self.x = np.asarray([[1.], [2.], [3.], [4.], [5.]])
    self.y1 = np.asarray([[2.], [4.], [6.], [8.], [10.]])
    self.y2 = np.asarray([[1.], [2.], [3.], [4.], [5.]])
    self.sample_weight_1 = np.asarray([2., 3., 4., 5., 6.])
    self.sample_weight_2 = np.asarray([3.5, 2.5, 1.5, 0.5, 3.])

    # y_true_1 = [[2.], [4.], [6.], [8.], [10.]]
    # y_pred_1 = [[3.], [6.], [9.], [12.], [15.]]
    # y_true_2 = [[1.], [2.], [3.], [4.], [5.]]
    # y_pred_2 = [[3.], [6.], [9.], [12.], [15.]]

    # Weighted metric `output_1`:
    #   Total = ((3 - 2)^2 * 2 + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5) +
    #           ((15 - 10)^2 *  6)
    #         = 280
    #   Count = (2 + 3) + (4 + 5) + 6 = 20
    #   Result = 14

    # Weighted metric `output_2`:
    #   Total = ((3 - 1)^2 * 3.5 + (6 - 2)^2 * 2.5) +
    #           ((9 - 3)^2 * 1.5 + (12 - 4)^2 * 0.5) +
    #           (15 - 5)^2 * 3.0
    #         = 440
    #   Count = (3.5 + 2.5) + (1.5 + 0.5) + 3.0 = 11.0
    #   Result = 40

    # Loss `output_1` with weights:
    #   Total = ((3 - 2)^2 * 2  + (6 - 4)^2 * 3) +
    #           ((9 - 6)^2 * 4 + (12 - 8)^2 * 5) +
    #           ((15 - 10)^2 *  6)
    #         = 280
    #   Count = 2 + 2 + 1
    #   Result = 56

    # Loss `output_1` without weights/Metric `output_1`:
    #   Total = ((3 - 2)^2 + (6 - 4)^2) + ((9 - 6)^2 + (12 - 8)^2) + (15 - 10)^2
    #         = 55
    #   Count = 2 + 2 + 1
    #   Result = 11

    # Loss `output_2` with weights:
    #   Total = ((3 - 1)^2 * 3.5 + (6 - 2)^2 * 2.5) +
    #           ((9 - 3)^2 * 1.5 + (12 - 4)^2 * 0.5) +
    #           (15 - 5)^2 * 3.0
    #         = 440
    #   Count = 2 + 2 + 1
    #   Result = 88

    # Loss `output_2` without weights/Metric `output_2`:
    #   Total = ((3 - 1)^2 + (6 - 2)^2) + ((9 - 3)^2 + (12 - 4)^2) + (15 - 5)^2
    #         = 220
    #   Count = 2 + 2 + 1
    #   Result = 44

    # Total loss with weights = 56 + 88 = 144
    # Total loss without weights = 11 + 44 = 55

    self.wmse = 'mean_squared_error_2'
    self.expected_fit_result_with_weights = {
        'output_1_mean_squared_error': [11, 11],
        'output_2_mean_squared_error': [44, 44],
        'output_1_' + self.wmse: [14, 14],
        'output_2_' + self.wmse: [40, 40],
        'loss': [144, 144],
        'output_1_loss': [56, 56],
        'output_2_loss': [88, 88],
    }

    self.expected_fit_result_with_weights_output_2 = {
        'output_1_mean_squared_error': [11, 11],
        'output_2_mean_squared_error': [44, 44],
        'output_1_' + self.wmse: [11, 11],
        'output_2_' + self.wmse: [40, 40],
        'loss': [99, 99],
        'output_1_loss': [11, 11],
        'output_2_loss': [88, 88],
    }

    self.expected_fit_result = {
        'output_1_mean_squared_error': [11, 11],
        'output_2_mean_squared_error': [44, 44],
        'output_1_' + self.wmse: [11, 11],
        'output_2_' + self.wmse: [44, 44],
        'loss': [55, 55],
        'output_1_loss': [11, 11],
        'output_2_loss': [44, 44],
    }

    # In the order: 'loss', 'output_1_loss', 'output_2_loss',
    # 'output_1_mean_squared_error', 'output_1_mean_squared_error_2',
    # 'output_2_mean_squared_error', 'output_2_mean_squared_error_2'
    self.expected_batch_result_with_weights = [144, 56, 88, 11, 14, 44, 40]
    self.expected_batch_result_with_weights_output_2 = [
        99, 11, 88, 11, 11, 44, 40
    ]
    self.expected_batch_result = [55, 11, 44, 11, 11, 44, 44]

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
        custom_generator_multi_io(), steps_per_epoch=3, epochs=2)
    for key, value in self.expected_fit_result.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_fit_generator_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    history = model.fit_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps_per_epoch=3,
        epochs=2)
    for key, value in self.expected_fit_result_with_weights.items():
      self.assertAllClose(history.history[key], value, 1e-3)

    # Set weights for one output.
    history = model.fit_generator(
        custom_generator_multi_io(
            sample_weights={'output_2': self.sample_weight_2}),
        steps_per_epoch=3,
        epochs=2)
    for key, value in self.expected_fit_result_with_weights_output_2.items():
      self.assertAllClose(history.history[key], value, 1e-3)

  def test_eval_generator(self):
    model = self._get_compiled_multi_io_model()
    eval_result = model.evaluate_generator(custom_generator_multi_io(), steps=3)
    self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

  def test_eval_generator_with_sample_weight(self):
    model = self._get_compiled_multi_io_model()
    eval_result = model.evaluate_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps=3)
    self.assertAllClose(eval_result, self.expected_batch_result_with_weights,
                        1e-3)

    # Set weights for one output.
    eval_result = model.evaluate_generator(
        custom_generator_multi_io(
            sample_weights={'output_2': self.sample_weight_2}),
        steps=3)
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
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
@parameterized.parameters([
    losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
    losses_utils.ReductionV2.AUTO,
    losses_utils.ReductionV2.SUM
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
    self.x = np.asarray([[1.], [2.], [3.], [4.], [5.]])
    self.y1 = np.asarray([[2.], [4.], [6.], [8.], [10.]])
    self.y2 = np.asarray([[1.], [2.], [3.], [4.], [5.]])
    self.sample_weight_1 = np.asarray([2., 3., 4., 5., 6.])
    self.sample_weight_2 = np.asarray([3.5, 2.5, 1.5, 0.5, 3.])

    # y_true_1 = [[2.], [4.], [6.], [8.], [10.]]
    # y_pred_1 = [[3.], [6.], [9.], [12.], [15.]]
    # y_true_2 = [[1.], [2.], [3.], [4.], [5.]]
    # y_pred_2 = [[3.], [6.], [9.], [12.], [15.]]

    # Loss `output_1`:
    #   Per-sample weighted losses
    #   Batch 1 = [(3 - 2)^2 * 2, (6 - 4)^2 * 3)] = [2, 12]
    #   Batch 2 = [((9 - 6)^2 * 4, (12 - 8)^2 * 5)] = [36, 80]
    #   Batch 3 = [(15 - 10)^2 * 6] = [150]

    #   Result (reduction=SUM) = ((2 + 12)*2 + (36 + 80)*2 + 150) / 5 = 82
    #   Result (reduction=SUM_OVER_BATCH_SIZE/AUTO/NONE) = 280 / 5 = 56

    # Loss `output_2`:
    #   Per-sample weighted losses
    #   Batch 1 = [(3 - 1)^2 * 3.5, (6 - 2)^2 * 2.5)] = [14, 40]
    #   Batch 2 = [(9 - 3)^2 * 1.5, (12 - 4)^2 * 0.5)] = [54, 32]
    #   Batch 3 = [(15 - 5)^2 * 3] = [300]

    #   Result (reduction=SUM) = ((14 + 40)*2 + (54 + 32)*2 + 300) / 5 = 116
    #   Result (reduction=SUM_OVER_BATCH_SIZE/AUTO/NONE) = 440 / 5 = 88

    # When reduction is 'NONE' loss value that is passed to the optimizer will
    # be vector loss but what is reported is a scalar, which is an average of
    # all the values in all the batch vectors.

    # Total loss = Output_loss_1 + Output_loss_2

    sum_over_batch_size_fit_result = {
        'loss': [144, 144],
        'output_1_loss': [56, 56],
        'output_2_loss': [88, 88],
    }

    self.expected_fit_result = {
        losses_utils.ReductionV2.NONE:
            sum_over_batch_size_fit_result,
        losses_utils.ReductionV2.SUM: {
            'loss': [198, 198],
            'output_1_loss': [82, 82],
            'output_2_loss': [116, 116],
        },
        losses_utils.ReductionV2.AUTO:
            sum_over_batch_size_fit_result,
        losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
            sum_over_batch_size_fit_result,
    }

    # In the order: 'loss', 'output_1_loss', 'output_2_loss',
    self.expected_batch_result = {
        losses_utils.ReductionV2.NONE: [144, 56, 88],
        losses_utils.ReductionV2.SUM: [198, 82, 116],
        losses_utils.ReductionV2.AUTO: [144, 56, 88],
        losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE: [144, 56, 88],
    }

    # 2 + 12 + 36 + 80 + 150 = 280
    # 14 + 40 + 54 + 32 + 300 = 440
    self.expected_single_batch_result = [720, 280, 440]

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
    if reduction == losses_utils.ReductionV2.SUM:
      expected_values = self.expected_single_batch_result
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
    if reduction == losses_utils.ReductionV2.SUM:
      expected_values = self.expected_single_batch_result
    self.assertAllClose(result, expected_values)

  def test_fit_generator(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    history = model.fit_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps_per_epoch=3,
        epochs=2)
    for key, value in self.expected_fit_result[reduction].items():
      self.assertAllClose(history.history[key], value)

  def test_eval_generator(self, reduction):
    model = self._get_compiled_multi_io_model(
        loss=losses.MeanSquaredError(reduction=reduction))
    eval_result = model.evaluate_generator(
        custom_generator_multi_io(
            sample_weights=[self.sample_weight_1, self.sample_weight_2]),
        steps=3)
    self.assertAllClose(eval_result, self.expected_batch_result[reduction])


if __name__ == '__main__':
  test.main()
