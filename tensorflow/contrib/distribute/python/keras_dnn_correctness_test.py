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
"""Correctness tests for tf.keras DNN model using DistributionStrategy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import keras_correctness_test_base
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import test
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.training import gradient_descent


def all_strategy_combinations_with_eager_and_graph_modes():
  return combinations.combine(distribution=keras_correctness_test_base.
                              all_strategies,
                              mode=['graph', 'eager'])


def all_strategy_combinations_with_graph_mode():
  return combinations.combine(distribution=keras_correctness_test_base.
                              all_strategies, mode=['graph'])


class TestDistributionStrategyDnnCorrectness(
    keras_correctness_test_base.TestDistributionStrategyCorrectnessBase):

  def get_model(self, initial_weights=None, distribution=None):
    with keras_correctness_test_base.MaybeDistributionScope(distribution):
      # We add few non-linear layers to make it non-trivial.
      model = keras.Sequential()
      model.add(keras.layers.Dense(10, activation='relu', input_shape=(1,)))
      model.add(keras.layers.Dense(10, activation='relu'))
      model.add(keras.layers.Dense(10, activation='relu'))
      model.add(keras.layers.Dense(1))

      if initial_weights:
        model.set_weights(initial_weights)

      model.compile(
          loss=keras.losses.mean_squared_error,
          optimizer=gradient_descent_keras.SGD(0.5),
          metrics=['mse'])
      return model

  def get_data(self):
    # TODO(xiejw): Change this back to 10000, once we support final partial
    # batch.
    num_samples = 9984
    x_train = np.random.rand(num_samples, 1)
    y_train = 3 * x_train
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_predict = [[1.], [2.], [3.], [4.]]
    return x_train, y_train, x_predict

  @combinations.generate(keras_correctness_test_base.
                         all_strategy_and_input_config_combinations())
  def test_dnn_correctness(self, distribution, use_numpy, use_validation_data):
    self.run_correctness_test(distribution, use_numpy, use_validation_data)

  @combinations.generate(all_strategy_combinations_with_graph_mode())
  def test_dnn_with_dynamic_learning_rate(self, distribution):
    self.run_dynamic_lr_test(distribution)


class TestDistributionStrategyDnnMetricCorrectness(
    keras_correctness_test_base.TestDistributionStrategyCorrectnessBase):

  def get_model(self, distribution=None):
    with distribution.scope():
      model = keras.Sequential()
      model.add(keras.layers.Dense(1,
                                   input_shape=(1,),
                                   kernel_initializer='ones'))
      model.compile(
          loss=keras.losses.mean_squared_error,
          optimizer=gradient_descent.GradientDescentOptimizer(0.5),
          metrics=[keras.metrics.BinaryAccuracy()])
    return model

  def run_metric_correctness_test(self, distribution):
    with self.cached_session():
      self.set_up_test_config()
      self.skip_unsupported_test_configuration(distribution)

      x_train, y_train, _ = self.get_data()
      model = self.get_model(distribution=distribution)

      batch_size = 64
      batch_size = (keras_correctness_test_base.
                    get_batch_size(batch_size, distribution))
      train_dataset = dataset_ops.Dataset.from_tensor_slices((x_train, y_train))
      train_dataset = (keras_correctness_test_base.
                       batch_wrapper(train_dataset, batch_size, distribution))

      history = model.fit(x=train_dataset, epochs=2, steps_per_epoch=10)
      self.assertEqual(history.history['binary_accuracy'], [1.0, 1.0])

  @combinations.generate(all_strategy_combinations_with_eager_and_graph_modes())
  def test_simple_dnn_metric_correctness(self, distribution):
    self.run_metric_correctness_test(distribution)


class TestDistributionStrategyDnnMetricEvalCorrectness(
    keras_correctness_test_base.TestDistributionStrategyCorrectnessBase):

  def get_model(self, distribution=None):
    with distribution.scope():
      model = keras.Sequential()
      model.add(
          keras.layers.Dense(
              3, activation='relu', input_dim=4, kernel_initializer='ones'))
      model.add(
          keras.layers.Dense(
              1, activation='sigmoid', kernel_initializer='ones'))
      model.compile(
          loss='mae',
          metrics=['accuracy', keras.metrics.BinaryAccuracy()],
          optimizer=gradient_descent.GradientDescentOptimizer(0.001))
    return model

  def run_eval_metrics_correctness_test(self, distribution):
    with self.cached_session():
      self.set_up_test_config()
      self.skip_unsupported_test_configuration(distribution)

      model = self.get_model(distribution=distribution)

      # verify correctness of stateful and stateless metrics.
      x = np.ones((100, 4)).astype('float32')
      y = np.ones((100, 1)).astype('float32')
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).repeat()
      dataset = (keras_correctness_test_base.
                 batch_wrapper(dataset, 4, distribution))
      outs = model.evaluate(dataset, steps=10)
      self.assertEqual(outs[1], 1.)
      self.assertEqual(outs[2], 1.)

      y = np.zeros((100, 1)).astype('float32')
      dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).repeat()
      dataset = (keras_correctness_test_base.
                 batch_wrapper(dataset, 4, distribution))
      outs = model.evaluate(dataset, steps=10)
      self.assertEqual(outs[1], 0.)
      self.assertEqual(outs[2], 0.)

  @combinations.generate(all_strategy_combinations_with_eager_and_graph_modes())
  def test_identity_model_metric_eval_correctness(self, distribution):
    self.run_eval_metrics_correctness_test(distribution)


if __name__ == '__main__':
  test.main()
