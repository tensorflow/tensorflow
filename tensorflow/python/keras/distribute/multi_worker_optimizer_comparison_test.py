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
"""Tests Keras multi worker callbacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading

from absl.testing import parameterized

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras.optimizer_v2 import rmsprop
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent as gradient_descent_v1
from tensorflow.python.training import rmsprop as rmsprop_v1


class KerasMultiWorkerOptimizerTest(test_base.IndependentWorkerTestBase,
                                    parameterized.TestCase):

  def run_optimizer_comparison_with_simple_bias_model(
      self, strategy_cls, optimizer_class_1, optimizer_class_2):

    def get_input_datasets():
      # Simple training input.
      train_input = [[1]] * 16
      train_label = [[0]] * 16
      ds = dataset_ops.Dataset.from_tensor_slices((train_input, train_label))
      # TODO(rchao): Investigate to figure out the reason for having 8 workers
      # instead of 2 as expected.
      return ds.batch(8, drop_remainder=True)

    def get_simple_bias_model():

      class Bias(base_layer.Layer):

        def build(self, input_shape):
          self.bias = self.add_variable('bias', (1,), initializer='zeros')

        def call(self, inputs):
          return inputs + self.bias

      model = sequential.Sequential()
      model.add(Bias(input_shape=(1,)))

      return model

    self._lock = threading.Lock()
    cluster_spec = test_base.create_cluster_spec(num_workers=2, test_obj=self)
    self._barrier = dc._Barrier(2)

    def _independent_worker_fn(*args, **kwargs):  # pylint: disable=unused-argument
      """Simulates an Independent Worker inside a thread."""
      # TODO(rchao): Refactor to abstract the common boilerplate out.
      with test.mock.patch.object(dc, '_run_std_server',
                                  self._make_mock_run_std_server()):

        model = get_simple_bias_model()

        initial_weights = model.get_weights()

        def _get_model_results(optimizer, initial_weights):

          # Clear Keras session to reset device assignment
          keras.backend._SESSION.session = None
          strategy = strategy_cls()

          with strategy.scope():
            train_ds = get_input_datasets()
            model = get_simple_bias_model()
            model.set_weights(initial_weights)
            model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])

          return {
              'trained_loss_and_accuracy':
                  model.fit(x=train_ds, epochs=20).history,
              'trained_weights':
                  model.get_weights(),
          }

        results1 = _get_model_results(optimizer_class_1(0.01), initial_weights)
        results2 = _get_model_results(optimizer_class_2(0.01), initial_weights)

        for key in results1:
          self.assertAllClose(
              results1[key],
              results2[key],
              atol=1e-5,
              rtol=1e-5,
              msg='Fail to assert {}'.format(key))

    threads = self.run_multiple_tasks_in_threads(_independent_worker_fn,
                                                 cluster_spec)

    threads_to_join = []
    strategy = strategy_cls()
    if strategy.extended.experimental_between_graph:
      for ts in threads.values():
        threads_to_join.extend(ts)
    else:
      threads_to_join = [threads['worker'][0]]
    self.join_independent_workers(threads_to_join)

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[collective_strategy.CollectiveAllReduceStrategy],
          required_gpus=[0, 1]))
  def test_sgd_optimizer_v1_v2_comparison(self, strategy_cls):
    self.run_optimizer_comparison_with_simple_bias_model(
        strategy_cls, gradient_descent.SGD,
        gradient_descent_v1.GradientDescentOptimizer)

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          strategy_cls=[collective_strategy.CollectiveAllReduceStrategy],
          required_gpus=[0, 1]))
  def test_rmsprop_optimizer_v1_v2_comparison(self, strategy_cls):
    self.skipTest('There is an issue in collective ops (b/127700538) that '
                  'prevent us from running this test with rmsprop optimizers.')
    self.run_optimizer_comparison_with_simple_bias_model(
        strategy_cls, rmsprop.RMSprop, rmsprop_v1.RMSPropOptimizer)


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
