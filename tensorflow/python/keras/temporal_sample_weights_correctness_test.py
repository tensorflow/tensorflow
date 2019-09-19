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
"""Tests temporal sample weights correctness using Keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class Bias(layers.Layer):
  """Layer that add a bias to its inputs."""

  def build(self, input_shape):
    self.bias = self.add_variable('bias', (1,), initializer='zeros')

  def call(self, inputs):
    return inputs + self.bias

  def compute_output_shape(self, input_shape):
    return input_shape


def get_multi_io_temporal_model():
  timesteps = 2
  inp_1 = layers.Input(shape=(1,), name='input_1')
  inp_2 = layers.Input(shape=(1,), name='input_2')
  x = layers.RepeatVector(timesteps)
  out_1 = layers.TimeDistributed(Bias(), name='output_1')
  out_2 = layers.TimeDistributed(Bias(), name='output_2')

  branch_a = [inp_1, x, out_1]
  branch_b = [inp_2, x, out_2]
  return testing_utils.get_multi_io_model(branch_a, branch_b)


def get_compiled_multi_io_model_temporal(sample_weight_mode):
  model = get_multi_io_temporal_model()
  model.compile(
      optimizer=optimizer_v2.gradient_descent.SGD(0.1),
      loss='mae',
      metrics=[metrics.MeanAbsoluteError(name='mae')],
      weighted_metrics=[metrics.MeanAbsoluteError(name='mae_2')],
      sample_weight_mode=sample_weight_mode,
      run_eagerly=testing_utils.should_run_eagerly(),
      experimental_run_tf_function=testing_utils.should_run_tf_function())
  return model


def run_with_different_sample_weight_mode_inputs(fn, partial_sw=True):
  """Executes the given function with different sample weight mode inputs.

  Args:
    fn: Training or eval function to execute.
    partial_sw: Boolean flag to indicate whether temporal sample weight mode
      should be set partially just for one output.
  """
  model = get_compiled_multi_io_model_temporal(sample_weight_mode='temporal')
  fn(model)

  model = get_compiled_multi_io_model_temporal(
      sample_weight_mode=['temporal', 'temporal'])
  fn(model)

  model = get_compiled_multi_io_model_temporal(sample_weight_mode={
      'output_1': 'temporal',
      'output_2': 'temporal'
  })
  fn(model)

  if partial_sw:
    model = get_compiled_multi_io_model_temporal(
        sample_weight_mode=[None, 'temporal'])
    fn(model)

    # TODO(b/129700800): Enable after bug is fixed.
    # model = get_compiled_multi_io_model_temporal(sample_weight_mode={
    #     'output_2': 'temporal'
    # })
    # fn(model)


@keras_parameterized.run_with_all_model_types(exclude_models=['sequential'])
@keras_parameterized.run_all_keras_modes
class TestMetricsCorrectnessMultiIOTemporal(keras_parameterized.TestCase):

  def custom_generator_multi_io_temporal(self, sample_weights=None):
    """Generator for getting data for temporal multi io model.

    Args:
      sample_weights: List of sample_weights.

    Yields:
      Tuple of inputs, label, sample weights data.
    """
    batch_size = 3
    num_samples = 3
    if sample_weights:
      assert len(sample_weights) == 2
      w1 = sample_weights[0]
      w2 = sample_weights[1]
    else:
      w1 = None
      w2 = None
    iteration = 0
    while True:
      batch_index = iteration * batch_size % num_samples
      iteration += 1
      start = batch_index
      end = start + batch_size
      x = [self.x[start:end], self.x[start:end]]
      y = [self.y1[start:end], self.y2[start:end]]
      if sample_weights:
        w = [
            None if w1 is None else w1[start:end],
            None if w2 is None else w2[start:end]
        ]
      else:
        w = None
      yield x, y, w

  def setUp(self):
    super(TestMetricsCorrectnessMultiIOTemporal, self).setUp()

    self.x = np.asarray([[0.], [1.], [2.]])
    self.y1 = np.asarray([[[.5], [1.]], [[2.], [2.5]], [[3.5], [2.5]]])
    self.y2 = np.asarray([[[.5], [1.5]], [[2.], [1.5]], [[3.5], [3.]]])

    if tf2.enabled():
      self.wmae = 'mae_2'
    else:
      self.wmae = 'weighted_mae_2'

    # Without weights:
    # Epoch 1 - bias = 0
    #   y_pred_1 = [[[0.], [0.]], [[1.], [1.]], [[2.], [2.]]]
    #   y_pred_2 = [[[0.], [0.]], [[1.], [1.]], [[2.], [2.]]]
    #   mae (y1 - y_pred_1) = [[[.5], [1.]], [[1.], [1.5]], [[1.5], [.5]]]
    #   mae                 = [[3/3, 3/3]] = [[1, 1]] = 2/2 = 1
    #   mae_2 (y2 - y_pred_2) = [[[.5], [1.5]], [[1.], [.5]], [[1.5], [1.]]]
    #   mae_2                 = [[3/3, 3/3]] = [[1, 1]] = 2/2 = 1

    # Epoch 2 - bias = 0.1 (2/2 * 0.1)
    #   y_pred_1 = [[[.1], [.1]], [[1.1], [1.1]], [[2.1], [2.1]]]
    #   y_pred_2 = [[[.1], [.1]], [[1.1], [1.1]], [[2.1], [2.1]]]
    #   mae (y1 - y_pred_1) = [[[.4], [.9]], [[.9], [1.4]], [[1.4], [.4]]]
    #   mae                 = [[2.7/3, 2.7/3]] = [[0.9, 0.9]] = 1.8/2 = 0.9
    #   mae_2 (y2 - y_pred_2) = [[[.4], [1.4]], [[.9], [.4]], [[1.4], [.9]]]
    #   mae_2                 = [[2.7/3, 2.7/3]] = [[0.9, 0.9]] = 1.8/2 = 0.9

    self.expected_fit_result = {
        'output_1_mae': [1, 0.9],
        'output_2_mae': [1, 0.9],
        'output_1_' + self.wmae: [1, 0.9],
        'output_2_' + self.wmae: [1, 0.9],
        'loss': [2., 1.8],
        'output_1_loss': [1, 0.9],
        'output_2_loss': [1, 0.9],
    }

    self.sample_weight_1 = np.asarray([[.5, 2.], [.5, 2.], [.5, 2.]])
    self.sample_weight_2 = np.asarray([[2., .5], [2., .5], [2., .5]])

    # With weights:
    # Epoch 1
    #   y_pred_1 = [[[0.], [0.]], [[1.], [1.]], [[2.], [2.]]]
    #   y_pred_2 = [[[0.], [0.]], [[1.], [1.]], [[2.], [2.]]]
    #   mae (y1 - y_pred_1) = [[[.5], [1.]], [[1.], [1.5]], [[1.5], [.5]]]
    #      with weights     = [[[.5 * .5], [1 * 2]],
    #                          [[1 * .5], [1.5 * 2]],
    #                          [[1.5 * .5], [.5 * 2]]]
    #   mae (w/o weights)   = [[3/3, 3/3]] = [[1, 1]] = 2/2 = 1
    #   mae (weighted mean) = [[1.5/1.5, 6/6]] = [[1, 1]] = 2/2 = 1
    #   mae (sum over bs)   = [[1.5/3, 6/3]] = [[.5, 2]] = 2.5/2 = 1.25

    #   mae_2 (y2 - y_pred_2) = [[[.5], [1.5]], [[1.], [.5]], [[1.5], [1.]]]
    #     with weights        = [[[.5 * 2], [1.5 * .5]],
    #                            [[1. * 2], [.5 * .5]],
    #                            [[1.5 * 2], [1. * .5]]]
    #   mae_2 (w/o weights)   = [[3/3, 3/3]] = [[1, 1]] = 2/2 = 1
    #   mae_2 (weighted mean) = [[6/6, 1.5/1.5]] = [[1, 1]] = 2/2 = 1
    #   mae_2 (sum over bs)   = [[6/3, 1.5/3]] = [[2, .5]] = 2.5/2 = 1.25

    # Epoch 2 - bias = 0.125 (2.5/2 * 0.1)
    #   y_pred_1 = [[[0.125], [0.125]], [[1.125], [1.125]], [[2.125], [2.125]]]
    #   y_pred_2 = [[[0.125], [0.125]], [[1.125], [1.125]], [[2.125], [2.125]]]

    #   mae (y1 - y_pred_1) = [[[.375], [.875]],
    #                          [[.875], [1.375]],
    #                          [[1.375], [.375]]]
    #     with weights      = [[[.375 * .5], [.875 * 2.]],
    #                          [[.875 * .5], [1.375 * 2.]],
    #                          [[1.375 * .5], [.375 * 2.]]]
    #   mae (w/o weights)   = [[2.625/3, 2.625/3]] = (.875+.875)/2 = .875
    #   mae (weighted mean) = [[1.3125/1.5,  5.25/6]] = (.875+.875)/2 = .875
    #   mae (sum over bs)   = [[1.3125/3,  5.25/3]] = (0.4375+1.75)/2 = 1.09375

    #   mae_2 (y2 - y_pred_2) = [[[.375], [1.375]],
    #                            [[.875], [.375]],
    #                            [[1.375], [.875]]]
    #     with weights        = [[[.375 * 2.], [1.375 * .5]],
    #                            [[.875 * 2.], [.375 * .5]],
    #                            [[1.375 * 2.], [.875 * .5]]]
    #   mae_2 (w/o weights)   = [[2.625/3, 2.625/3]] = (.875+.875)/2 = .875
    #   mae_2 (weighted mean) = [[5.25/6, 1.3125/1.5]] = (.875+.875)/2 = .875
    #   mae_2 (sum over bs)  = [[5.25/3, 1.3125/3]] = (1.75+0.4375)/2 = 1.09375

    self.expected_fit_result_with_weights = {
        'output_1_mae': [1, 0.875],
        'output_2_mae': [1, 0.875],
        'output_1_' + self.wmae: [1, 0.875],
        'output_2_' + self.wmae: [1, 0.875],
        'loss': [2.5, 2.1875],
        'output_1_loss': [1.25, 1.09375],
        'output_2_loss': [1.25, 1.09375],
    }

    self.expected_fit_result_with_weights_output_2 = {
        'output_1_mae': [1., 0.9],
        'output_2_mae': [1, 0.875],
        'output_1_' + self.wmae: [1., 0.9],
        'output_2_' + self.wmae: [1., 0.875],
        'loss': [2.25, 1.99375],
        'output_1_loss': [1., 0.9],
        'output_2_loss': [1.25, 1.09375],
    }

    # In the order: 'loss', 'output_1_loss', 'output_2_loss',
    # 'output_1_mae', 'output_1_mae_2',
    # 'output_2_mae', 'output_2_mae_2'
    self.expected_batch_result_with_weights = [
        2.1875, 1.09375, 1.09375, 0.875, 0.875, 0.875, 0.875
    ]
    self.expected_batch_result_with_weights_output_2 = [
        1.99375, 0.9, 1.09375, 0.9, 0.9, 0.875, 0.875
    ]
    self.expected_batch_result = [1.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

  def test_fit(self):

    def _train_and_assert(model):
      history = model.fit([self.x, self.x], [self.y1, self.y2],
                          batch_size=3,
                          epochs=2,
                          shuffle=False)
      for key, value in self.expected_fit_result.items():
        self.assertAllClose(history.history[key], value, 1e-3)

    run_with_different_sample_weight_mode_inputs(_train_and_assert)

  def test_fit_with_sample_weight(self):

    def _train_and_assert(model):
      history = model.fit([self.x, self.x], [self.y1, self.y2],
                          sample_weight={
                              'output_1': self.sample_weight_1,
                              'output_2': self.sample_weight_2,
                          },
                          batch_size=3,
                          epochs=2,
                          shuffle=False)
      for key, value in self.expected_fit_result_with_weights.items():
        self.assertAllClose(history.history[key], value, 1e-3)

    run_with_different_sample_weight_mode_inputs(
        _train_and_assert, partial_sw=False)

  def test_fit_with_partial_sample_weight(self):

    def _train_and_assert(model):
      history = model.fit([self.x, self.x], [self.y1, self.y2],
                          sample_weight={
                              'output_2': self.sample_weight_2,
                          },
                          batch_size=3,
                          epochs=2,
                          shuffle=False)
      for key, value in self.expected_fit_result_with_weights_output_2.items():
        self.assertAllClose(history.history[key], value, 1e-3)

    run_with_different_sample_weight_mode_inputs(_train_and_assert)

  def test_eval(self):

    def _eval_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2])
      eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                   batch_size=3)
      self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

    run_with_different_sample_weight_mode_inputs(_eval_and_assert)

  def test_eval_with_sample_weight(self):

    def _eval_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                           sample_weight={
                               'output_1': self.sample_weight_1,
                               'output_2': self.sample_weight_2,
                           })
      eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                   batch_size=3,
                                   sample_weight={
                                       'output_1': self.sample_weight_1,
                                       'output_2': self.sample_weight_2,
                                   })
      self.assertAllClose(eval_result, self.expected_batch_result_with_weights,
                          1e-3)

    run_with_different_sample_weight_mode_inputs(
        _eval_and_assert, partial_sw=False)

  def test_eval_with_partial_sample_weight(self):

    def _eval_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                           sample_weight={
                               'output_2': self.sample_weight_2,
                           })
      eval_result = model.evaluate([self.x, self.x], [self.y1, self.y2],
                                   batch_size=3,
                                   sample_weight={
                                       'output_2': self.sample_weight_2,
                                   })
      self.assertAllClose(eval_result,
                          self.expected_batch_result_with_weights_output_2,
                          1e-3)

    run_with_different_sample_weight_mode_inputs(_eval_and_assert)

  def test_train_on_batch(self):

    def _train_and_assert(model):
      for _ in range(2):
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2])
      self.assertAllClose(result, self.expected_batch_result, 1e-3)

    run_with_different_sample_weight_mode_inputs(_train_and_assert)

  def test_train_on_batch_with_sample_weight(self):

    def _train_and_assert(model):
      for _ in range(2):
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                      sample_weight={
                                          'output_1': self.sample_weight_1,
                                          'output_2': self.sample_weight_2,
                                      })
      self.assertAllClose(result, self.expected_batch_result_with_weights, 1e-3)

    run_with_different_sample_weight_mode_inputs(
        _train_and_assert, partial_sw=False)

  def test_train_on_batch_with_partial_sample_weight(self):

    def _train_and_assert(model):
      for _ in range(2):
        result = model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                                      sample_weight={
                                          'output_2': self.sample_weight_2,
                                      })
      self.assertAllClose(result,
                          self.expected_batch_result_with_weights_output_2,
                          1e-3)

    run_with_different_sample_weight_mode_inputs(_train_and_assert)

  def test_test_on_batch(self):

    def _test_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2])
      result = model.test_on_batch([self.x, self.x], [self.y1, self.y2])
      self.assertAllClose(result, self.expected_batch_result, 1e-3)

    run_with_different_sample_weight_mode_inputs(_test_and_assert)

  def test_test_on_batch_with_sample_weight(self):

    def _test_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                           sample_weight={
                               'output_1': self.sample_weight_1,
                               'output_2': self.sample_weight_2,
                           })
      result = model.test_on_batch([self.x, self.x], [self.y1, self.y2],
                                   sample_weight={
                                       'output_1': self.sample_weight_1,
                                       'output_2': self.sample_weight_2,
                                   })
      self.assertAllClose(result, self.expected_batch_result_with_weights, 1e-3)

    run_with_different_sample_weight_mode_inputs(
        _test_and_assert, partial_sw=False)

  def test_test_on_batch_with_partial_sample_weight(self):

    def _test_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                           sample_weight={
                               'output_2': self.sample_weight_2,
                           })
      result = model.test_on_batch([self.x, self.x], [self.y1, self.y2],
                                   sample_weight={
                                       'output_2': self.sample_weight_2,
                                   })
      self.assertAllClose(result,
                          self.expected_batch_result_with_weights_output_2,
                          1e-3)

    run_with_different_sample_weight_mode_inputs(_test_and_assert)

  def test_fit_generator(self):

    def _train_and_assert(model):
      history = model.fit_generator(
          self.custom_generator_multi_io_temporal(),
          steps_per_epoch=1,
          epochs=2)
      for key, value in self.expected_fit_result.items():
        self.assertAllClose(history.history[key], value, 1e-3)

    run_with_different_sample_weight_mode_inputs(_train_and_assert)

  def test_fit_generator_with_sample_weight(self):

    def _train_and_assert(model):
      history = model.fit_generator(
          self.custom_generator_multi_io_temporal(
              sample_weights=[self.sample_weight_1, self.sample_weight_2]),
          steps_per_epoch=1,
          epochs=2)
      for key, value in self.expected_fit_result_with_weights.items():
        self.assertAllClose(history.history[key], value, 1e-3)

    run_with_different_sample_weight_mode_inputs(
        _train_and_assert, partial_sw=False)

  def test_fit_generator_with_partial_sample_weight(self):

    def _train_and_assert(model):
      history = model.fit_generator(
          self.custom_generator_multi_io_temporal(
              sample_weights=[None, self.sample_weight_2]),
          steps_per_epoch=1,
          epochs=2)
      for key, value in self.expected_fit_result_with_weights_output_2.items():
        self.assertAllClose(history.history[key], value, 1e-3)

    run_with_different_sample_weight_mode_inputs(_train_and_assert)

  def test_eval_generator(self):

    def _test_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2])
      eval_result = model.evaluate_generator(
          self.custom_generator_multi_io_temporal(), steps=1)
      self.assertAllClose(eval_result, self.expected_batch_result, 1e-3)

    run_with_different_sample_weight_mode_inputs(_test_and_assert)

  def test_eval_generator_with_sample_weight(self):

    def _test_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                           sample_weight={
                               'output_1': self.sample_weight_1,
                               'output_2': self.sample_weight_2,
                           })
      eval_result = model.evaluate_generator(
          self.custom_generator_multi_io_temporal(
              sample_weights=[self.sample_weight_1, self.sample_weight_2]),
          steps=2)
      self.assertAllClose(eval_result, self.expected_batch_result_with_weights,
                          1e-3)

    run_with_different_sample_weight_mode_inputs(
        _test_and_assert, partial_sw=False)

  def test_eval_generator_with_partial_sample_weight(self):

    def _test_and_assert(model):
      model.train_on_batch([self.x, self.x], [self.y1, self.y2],
                           sample_weight={
                               'output_2': self.sample_weight_2,
                           })
      eval_result = model.evaluate_generator(
          self.custom_generator_multi_io_temporal(
              sample_weights=[None, self.sample_weight_2]),
          steps=2)
      self.assertAllClose(eval_result,
                          self.expected_batch_result_with_weights_output_2,
                          1e-3)

    run_with_different_sample_weight_mode_inputs(_test_and_assert)

  def test_error_on_fit_with_class_weight(self):

    def _train_and_assert(model):
      with self.assertRaisesRegex(
          ValueError,
          r'`class_weight` not supported for 3\+ dimensional targets.'):
        model.fit([self.x, self.x], [self.y1, self.y2],
                  class_weight={'output_1': {
                      .5: .5,
                      2.: .5,
                      3.5: .5
                  }},
                  batch_size=3,
                  epochs=2,
                  shuffle=False)

    run_with_different_sample_weight_mode_inputs(_train_and_assert)


if __name__ == '__main__':
  test.main()
