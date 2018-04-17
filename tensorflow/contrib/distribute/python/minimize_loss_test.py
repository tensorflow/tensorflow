# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for running legacy optimizer code with DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python.single_loss_example import batchnorm_example
from tensorflow.contrib.distribute.python.single_loss_example import minimize_loss_example
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.losses import losses_impl


class MinimizeLossStepTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          combinations.distributions_and_v1_optimizers(),
          combinations.combine(mode=["graph"], use_callable_loss=[True, False])
          + combinations.combine(mode=["eager"], use_callable_loss=[True]),
          combinations.combine(is_tpu=[False])) +
      combinations.combine(
          distribution=[combinations.tpu_strategy],
          optimizer_fn=[combinations.adam_optimizer_v1_fn],
          mode=["graph"],
          use_callable_loss=[False],
          is_tpu=[True]))
  def testTrainNetwork(self, distribution, optimizer_fn, use_callable_loss,
                       is_tpu):
    with distribution.scope():
      model_fn, dataset, layer = minimize_loss_example(
          optimizer_fn,
          use_bias=True,
          use_callable_loss=use_callable_loss)

      # TODO(isaprykin):  Eliminate `is_tpu`. Probably add a
      # `DistributionStrategy.create_monitor` so that each DistributionStrategy
      # could influence its training loop. That method would return an instance
      # of Monitor.  TPUMonitor would execute tpu.initialize_system() and
      # tpu.shutdown_system().
      if is_tpu:
        dataset = dataset.batch(2)

      iterator = distribution.distribute_dataset(dataset)

      def run_step():
        # TODO(isaprykin): Make iterator get_next() return a list of sub-
        # batches for each iteration. Pass iterator.get_next() and not iterator
        # to call_for_each_tower.
        return distribution.group(
            distribution.call_for_each_tower(
                model_fn,
                iterator.get_next() if not is_tpu else iterator,
                run_concurrently=layer.built))

      if not context.executing_eagerly():
        with self.test_session() as sess:
          if is_tpu:
            sess.run(tpu.initialize_system())
          run_step = sess.make_callable(run_step())
        self.evaluate(variables_lib.global_variables_initializer())

      weights, biases = [], []
      for _ in range(10):
        run_step()

        weights.append(self.evaluate(distribution.fetch(layer.kernel)))
        biases.append(self.evaluate(distribution.fetch(layer.bias)))

      if is_tpu:
        with self.test_session() as sess:
          sess.run(tpu.shutdown_system())

      error = abs(numpy.add(numpy.squeeze(weights), numpy.squeeze(biases)) - 1)
      is_not_increasing = all(y <= x for x, y in zip(error, error[1:]))
      self.assertTrue(is_not_increasing)

  @combinations.generate(
      combinations.times(
          combinations.distributions_and_v1_optimizers() +
          combinations.distributions_and_v2_optimizers(),
          combinations.combine(mode=["graph", "eager"])))
  def testOptimizerInsideModelFn(self, distribution, optimizer_fn):
    created_variables = []
    trainable_variables = []

    def appending_creator(next_creator, *args, **kwargs):
      v = next_creator(*args, **kwargs)
      created_variables.append(v.name)
      if "trainable" in kwargs and kwargs["trainable"]:
        trainable_variables.append(v.name)
      return v

    # Creator scope needs to be set before it's used inside
    # `distribution.scope`.
    with variable_scope.variable_creator_scope(
        appending_creator), distribution.scope():
      model_fn, dataset, layer = minimize_loss_example(
          optimizer_fn,
          use_bias=True,
          use_callable_loss=True,
          create_optimizer_inside_model_fn=True)

      iterator = distribution.distribute_dataset(dataset)

      def run_step():
        return distribution.group(
            distribution.call_for_each_tower(
                model_fn, iterator.get_next(), run_concurrently=layer.built))

      if not context.executing_eagerly():
        with self.test_session() as sess:
          run_step = sess.make_callable(run_step())
        self.evaluate(variables_lib.global_variables_initializer())

      run_step()

      def get_expected_variables(optimizer_fn, num_parameter_devices):
        variables_map = {
            "GradientDescent": ["dense/kernel", "dense/bias"],
            "Adam": [
                "dense/kernel", "dense/bias", "beta1_power", "beta2_power",
                "dense/kernel/Adam", "dense/kernel/Adam_1", "dense/bias/Adam",
                "dense/bias/Adam_1"
            ]
        }
        variables = variables_map[optimizer_fn().get_name()]
        variables.extend([
            v + "/replica_{}".format(replica)
            for v in variables
            for replica in range(1, num_parameter_devices)
        ])
        return set([v + ":0" for v in variables])

      self.assertEqual(
          get_expected_variables(optimizer_fn,
                                 len(distribution.parameter_devices)),
          set(created_variables))

  @combinations.generate(
      combinations.times(combinations.distributions_and_v1_optimizers(),
                         combinations.combine(
                             mode=["graph", "eager"],
                             momentum=[0.8, 0.9, 0.99],
                             renorm=[False, True])))
  def testTrainNetworkWithBatchNorm(self, distribution, optimizer_fn, momentum,
                                    renorm):
    """Verifies that moving mean updates are reduced across towers."""
    with distribution.scope():
      num_towers = len(distribution.worker_devices)
      model_fn, dataset, batchnorm = batchnorm_example(
          optimizer_fn,
          batch_per_epoch=num_towers,
          momentum=momentum,
          renorm=renorm)

      # Disable prefetching since that makes the specific input on each device
      # to be non deterministic, and this test relies on specific input being
      # on each device.
      if isinstance(distribution, mirrored_strategy.MirroredStrategy):
        distribution._prefetch_on_device = False
      iterator = distribution.distribute_dataset(dataset)

      def run_step():
        return control_flow_ops.group(
            distribution.unwrap(
                distribution.call_for_each_tower(
                    model_fn,
                    iterator.get_next(),
                    run_concurrently=batchnorm.built)) +
            ops.get_collection(ops.GraphKeys.UPDATE_OPS))

      if not context.executing_eagerly():
        with self.test_session() as sess:
          run_step = sess.make_callable(run_step())
        self.evaluate(variables_lib.global_variables_initializer())

      expected_moving_means = [0.] * 8

      def averaged_batch_mean(i):
        # Each batch has shape [16, 8] where the ith element in jth list is
        # (8 * j + i + tower_id * 100). So the batch mean in each tower is
        # (60 + i + tower_id * 100). So here comes its batch mean over all
        # towers:
        return 60. + i + (num_towers - 1.) / 2. * 100.

      for _ in range(10):
        run_step()
        moving_means = self.evaluate(distribution.fetch(batchnorm.moving_mean))

        # We make sure that the moving_mean is updated as if the sample mean is
        # calculated over all towers.
        for i, expected_moving_mean in enumerate(expected_moving_means):
          expected_moving_means[i] -= ((
              expected_moving_mean - averaged_batch_mean(i)) * (1.0 - momentum))
          self.assertNear(expected_moving_means[i], moving_means[i], 0.0001)

  @combinations.generate(
      combinations.times(
          combinations.combine(
              distribution=[combinations.one_device_strategy,
                            combinations.mirrored_strategy_with_gpu_and_cpu,
                            combinations.mirrored_strategy_with_two_gpus],
              optimizer_fn=[combinations.gradient_descent_optimizer_v1_fn,
                            combinations.gradient_descent_optimizer_v2_fn],
              loss_reduction=[losses_impl.Reduction.SUM,
                              losses_impl.Reduction.MEAN,
                              losses_impl.Reduction.SUM_OVER_BATCH_SIZE,
                              losses_impl.Reduction.SUM_OVER_NONZERO_WEIGHTS]),
          combinations.combine(mode=["graph"], use_callable_loss=[True, False])
          + combinations.combine(mode=["eager"], use_callable_loss=[True])))
  def testMeanVsSum(self, distribution, optimizer_fn, loss_reduction,
                    use_callable_loss):
    with distribution.scope():
      all_vars = []

      def model_fn(x, y):

        def loss_fn():
          # Use fixed initialization to make the steps deterministic.
          w = variable_scope.get_variable("w", initializer=[[2.]])
          all_vars.append(w)
          predict = math_ops.matmul(x, w)
          return losses_impl.mean_squared_error(
              y, predict, reduction=loss_reduction)

        optimizer = optimizer_fn()  # GradientDescent with 0.2 learning rate

        if use_callable_loss:
          return optimizer.minimize(loss_fn)
        else:
          return optimizer.minimize(loss_fn())

      features = dataset_ops.Dataset.from_tensors([[2.], [7.]])
      labels = dataset_ops.Dataset.from_tensors([[6.], [21.]])
      dataset = dataset_ops.Dataset.zip((features, labels)).repeat()
      iterator = distribution.distribute_dataset(dataset)

      def run_step():
        return distribution.group(
            distribution.call_for_each_tower(
                model_fn, *iterator.get_next(), run_concurrently=False))

      if not context.executing_eagerly():
        with self.test_session() as sess:
          run_step = sess.make_callable(run_step())
        self.evaluate(variables_lib.global_variables_initializer())

      run_step()

      self.assertEqual(distribution.num_towers, len(all_vars))
      v = all_vars[0]
      self.assertTrue(all([v is vi for vi in all_vars[1:]]))
      weight = numpy.squeeze(self.evaluate(distribution.fetch(v)))
      # Our model is:
      #   predict = x * w
      #   loss = (predict - y)^2
      #   dloss/dpredict = 2*(predict - y)
      #   dloss/dw = 2 * x^T @ (predict - y)
      # For our batch size of 2, assuming sum loss reduction:
      #   x = [2, 7]
      #   y = [6, 21]
      #   w_initial = 2
      #   predict = [4, 14]
      #   predict - y = [-2, -7]
      #   dloss/dw = 2 <[2, 7], [-2, -7]> = - 2(4 + 49) = -106
      # So unreplicated the update to w with lr=0.2 is -0.2 * -106 = 21.2
      # with sum loss reduction, or 10.6 with mean.
      if loss_reduction == losses_impl.Reduction.SUM:
        # Note that the "distribution.num_towers" factor will go away once
        # we split the input across towers, instead of pulling a complete
        # batch of input per tower.
        self.assertNear(weight, 2 + 21.2 * distribution.num_towers, 0.0001)
      else:
        # One of the mean loss reductions.
        self.assertNear(weight, 2 + 10.6, 0.0001)


if __name__ == "__main__":
  test.main()
