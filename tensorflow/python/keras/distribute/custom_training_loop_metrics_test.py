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
"""Tests for custom training loops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.framework import test_util
from tensorflow.python.keras import metrics
from tensorflow.python.keras.distribute import strategy_combinations
from tensorflow.python.platform import test


class KerasMetricsTest(test.TestCase, parameterized.TestCase):

  @ds_combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies +
          strategy_combinations.multiworker_strategies,
          mode=["eager"]
      ))
  def test_multiple_keras_metrics_experimental_run(self, distribution):
    with distribution.scope():
      loss_metric = metrics.Mean("loss", dtype=np.float32)
      loss_metric_2 = metrics.Mean("loss_2", dtype=np.float32)

    @def_function.function
    def train_step():
      def step_fn():
        loss = constant_op.constant(5.0, dtype=np.float32)
        loss_metric.update_state(loss)
        loss_metric_2.update_state(loss)

      distribution.run(step_fn)

    train_step()
    self.assertEqual(loss_metric.result().numpy(),
                     loss_metric_2.result().numpy())
    self.assertEqual(loss_metric.result().numpy(), 5.0)

  @ds_combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies+
          strategy_combinations.multiworker_strategies,
          mode=["eager"]
      ))
  def test_update_keras_metric_declared_in_strategy_scope(self, distribution):
    with distribution.scope():
      metric = metrics.Mean("test_metric", dtype=np.float32)

    dataset = dataset_ops.Dataset.range(10).batch(2)
    dataset = distribution.experimental_distribute_dataset(dataset)

    @def_function.function
    def step_fn(i):
      metric.update_state(i)

    for i in dataset:
      distribution.run(step_fn, args=(i,))

    # This should be the mean of integers 0-9 which has a sum of 45 and a count
    # of 10 resulting in mean of 4.5.
    self.assertEqual(metric.result().numpy(), 4.5)

  @ds_combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def test_update_keras_metric_outside_strategy_scope_cross_replica(
      self, distribution):
    metric = metrics.Mean("test_metric", dtype=np.float32)

    with distribution.scope():
      for i in range(10):
        metric.update_state(i)

    # This should be the mean of integers 0-9 which has a sum of 45 and a count
    # of 10 resulting in mean of 4.5.
    self.assertEqual(metric.result().numpy(), 4.5)

  @ds_combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies, mode=["eager"]))
  @test_util.disable_mlir_bridge("TODO(b/168036682): Support dynamic padder")
  def test_update_keras_metrics_dynamic_shape(self, distribution):
    with distribution.scope():
      metric = metrics.Mean("test_metric", dtype=np.float32)

    dataset = dataset_ops.Dataset.range(10).batch(2, drop_remainder=False)

    @def_function.function
    def train_fn(dataset):
      weights = constant_op.constant([0.1, 0.1])

      def step_fn(i):
        metric.update_state(i, weights)

      for i in dataset:
        distribution.run(step_fn, args=(i,))

    train_fn(dataset)

    # This should be the mean of integers 0-9 which has a sum of 45 and a count
    # of 10 resulting in mean of 4.5.
    self.assertEqual(metric.result().numpy(), 4.5)


if __name__ == "__main__":
  multi_process_runner.test_main()
