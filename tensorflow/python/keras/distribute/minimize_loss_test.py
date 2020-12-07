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

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras.distribute import optimizer_combinations
from tensorflow.python.keras.distribute.test_example import batchnorm_example
from tensorflow.python.keras.distribute.test_example import minimize_loss_example
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.platform import test


VAR_MAP_V1 = {
    "GradientDescent": ("dense/kernel", "dense/bias"),
    "Adagrad": ("dense/kernel/Adagrad", "dense/kernel", "dense/bias/Adagrad",
                "dense/bias"),
    "Ftrl": ("dense/kernel/Ftrl", "dense/kernel", "dense/bias/Ftrl",
             "dense/bias", "dense/kernel/Ftrl_1", "dense/bias/Ftrl_1"),
    "RMSProp": ("dense/kernel", "dense/bias/RMSProp", "dense/bias/RMSProp_1",
                "dense/bias", "dense/kernel/RMSProp_1", "dense/kernel/RMSProp")
}

VAR_MAP_V2 = {
    "SGD": ("dense/bias", "SGD/learning_rate", "SGD/decay", "SGD/iter",
            "dense/kernel", "SGD/momentum"),
    "Adagrad":
        ("Adagrad/iter", "dense/bias", "dense/kernel", "Adagrad/learning_rate",
         "Adagrad/decay", "Adagrad/dense/kernel/accumulator",
         "Adagrad/dense/bias/accumulator")
}


class MinimizeLossStepTest(test.TestCase, parameterized.TestCase):

  def _get_iterator(self, strategy, input_fn):
    iterator = strategy.make_input_fn_iterator(lambda _: input_fn())
    self.evaluate(iterator.initializer)
    return iterator

  @ds_combinations.generate(
      combinations.times(
          optimizer_combinations.distributions_and_v1_optimizers(),
          combinations.combine(mode=["graph"], use_callable_loss=[True, False])
          + combinations.combine(mode=["eager"], use_callable_loss=[True])) +
      combinations.times(
          optimizer_combinations.distributions_and_v2_optimizers(),
          combinations.combine(
              mode=["graph", "eager"], use_callable_loss=[True])) +
      combinations.combine(
          distribution=[strategy_combinations.tpu_strategy],
          optimizer_fn=optimizer_combinations.optimizers_v2,
          mode=["graph"],
          use_callable_loss=[True]) + combinations.combine(
              distribution=[strategy_combinations.tpu_strategy],
              optimizer_fn=optimizer_combinations.optimizers_v1,
              mode=["graph"],
              use_callable_loss=[True, False]))
  def testTrainNetwork(self, distribution, optimizer_fn, use_callable_loss):
    with distribution.scope():
      optimizer = optimizer_fn()
      model_fn, dataset_fn, layer = minimize_loss_example(
          optimizer, use_bias=True, use_callable_loss=use_callable_loss)

      def step_fn(ctx, inputs):
        del ctx  # Unused
        return distribution.group(
            distribution.extended.call_for_each_replica(
                model_fn, args=(inputs,)))

      iterator = self._get_iterator(distribution, dataset_fn)

      def run_step():
        return distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=2).run_op

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          run_step = sess.make_callable(run_step())
      self.evaluate(variables_lib.global_variables_initializer())

      weights, biases = [], []
      for _ in range(5):
        run_step()
        weights.append(self.evaluate(layer.kernel))
        biases.append(self.evaluate(layer.bias))

      error = abs(numpy.add(numpy.squeeze(weights), numpy.squeeze(biases)) - 1)
      is_not_increasing = all(y <= x for x, y in zip(error, error[1:]))
      self.assertTrue(is_not_increasing)

  @ds_combinations.generate(
      combinations.times(
          optimizer_combinations.distributions_and_v1_optimizers(),
          combinations.combine(mode=["graph"], use_callable_loss=[True, False])
          + combinations.combine(mode=["eager"], use_callable_loss=[True])) +
      combinations.times(
          optimizer_combinations.distributions_and_v2_optimizers(),
          combinations.combine(
              mode=["graph", "eager"], use_callable_loss=[True])))
  def testTrainNetworkByCallForEachReplica(self, distribution, optimizer_fn,
                                           use_callable_loss):
    with distribution.scope():
      optimizer = optimizer_fn()
      model_fn, dataset_fn, layer = minimize_loss_example(
          optimizer, use_bias=True, use_callable_loss=use_callable_loss)

      iterator = self._get_iterator(distribution, dataset_fn)

      def run_step():
        return distribution.group(
            distribution.extended.call_for_each_replica(
                model_fn, args=(iterator.get_next(),)))

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          run_step = sess.make_callable(run_step())
        self.evaluate(variables_lib.global_variables_initializer())

      weights, biases = [], []
      for _ in range(10):
        run_step()

        weights.append(self.evaluate(layer.kernel))
        biases.append(self.evaluate(layer.bias))

      error = abs(numpy.add(numpy.squeeze(weights), numpy.squeeze(biases)) - 1)
      is_not_increasing = all(y <= x for x, y in zip(error, error[1:]))
      self.assertTrue(is_not_increasing)

  @ds_combinations.generate(
      combinations.times(
          optimizer_combinations.distributions_and_v1_and_v2_optimizers(),
          combinations.combine(mode=["graph", "eager"])) + combinations.combine(
              distribution=[strategy_combinations.tpu_strategy],
              optimizer_fn=optimizer_combinations.optimizers_v1_and_v2,
              mode=["graph"]))
  def testOptimizerInsideModelFn(self, distribution, optimizer_fn):
    if (not context.executing_eagerly() and
        control_flow_v2_toggles.control_flow_v2_enabled()):
      self.skipTest("b/138751864")
    created_variables = []
    trainable_variables = []

    def appending_creator(next_creator, **kwargs):
      v = next_creator(**kwargs)
      created_variables.append(v.name)
      if "trainable" in kwargs and kwargs["trainable"]:
        trainable_variables.append(v.name)
      return v

    # Creator scope needs to be set before it's used inside
    # `distribution.scope`.
    with variable_scope.variable_creator_scope(
        appending_creator), distribution.scope():
      optimizer = optimizer_fn()
      model_fn, dataset_fn, _ = minimize_loss_example(
          optimizer, use_bias=True, use_callable_loss=True)

      def step_fn(ctx, inputs):
        del ctx  # Unused
        return distribution.group(
            distribution.extended.call_for_each_replica(
                model_fn, args=(inputs,)))

      iterator = self._get_iterator(distribution, dataset_fn)

      def run_step():
        return distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=1).run_op

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          run_step = sess.make_callable(run_step())
      self.evaluate(variables_lib.global_variables_initializer())
      run_step()

      def get_expected_variables(num_parameter_devices):
        name = optimizer._name

        if isinstance(optimizer, optimizer_v2.OptimizerV2):
          variables = VAR_MAP_V2[name]
        else:
          variables = VAR_MAP_V1[name]

        extended_variables = [
            v + "/replica_{}".format(replica)
            for v in variables
            for replica in range(1, num_parameter_devices)
        ]
        variables = list(variables) + extended_variables
        return set(v + ":0" for v in variables)

      self.assertEqual(
          get_expected_variables(len(distribution.extended.parameter_devices)),
          set(created_variables))

  @ds_combinations.generate(
      combinations.times(
          combinations.combine(momentum=[0.8, 0.9, 0.99], renorm=[False, True]),
          combinations.times(
              optimizer_combinations.distributions_and_v1_and_v2_optimizers(),
              combinations.combine(
                  mode=["graph", "eager"],
                  # TODO(isaprykin):  Allow False here.  Currently subsequent
                  # replicas will re-execute UPDATE_OPS of previous replicas.
                  update_ops_in_cross_replica_mode=[True])) +
          combinations.combine(
              distribution=[strategy_combinations.tpu_strategy],
              optimizer_fn=optimizer_combinations.optimizers_v1_and_v2,
              mode=["graph"],
              update_ops_in_cross_replica_mode=[False])))
  def testTrainNetworkWithBatchNorm(self, distribution, optimizer_fn, momentum,
                                    renorm, update_ops_in_cross_replica_mode):
    """Verifies that moving mean updates are reduced across replicas."""
    with distribution.scope():
      num_replicas = distribution.num_replicas_in_sync
      model_fn, dataset_fn, batchnorm = batchnorm_example(
          optimizer_fn,
          batch_per_epoch=num_replicas,
          momentum=momentum,
          renorm=renorm,
          update_ops_in_replica_mode=not update_ops_in_cross_replica_mode)

      def step_fn(ctx, inputs):
        del ctx  # Unused
        fetches = distribution.experimental_local_results(
            distribution.extended.call_for_each_replica(
                model_fn, args=(inputs,)))
        if update_ops_in_cross_replica_mode:
          fetches += tuple(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
        return control_flow_ops.group(fetches)

      iterator = self._get_iterator(distribution, dataset_fn)

      def run_step():
        return distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=1).run_op

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          run_step = sess.make_callable(run_step())
      self.evaluate(variables_lib.global_variables_initializer())

      expected_moving_means = [0.] * 8

      def averaged_batch_mean(i):
        # Each batch has shape [16, 8] where the ith element in jth list is
        # (8 * j + i + replica_id * 100). So the batch mean in each replica is
        # (60 + i + replica_id * 100). So here comes its batch mean over all
        # replicas:
        return 60. + i + (num_replicas - 1.) / 2. * 100.

      for _ in range(10):
        run_step()
        moving_means = self.evaluate(batchnorm.moving_mean)

        # We make sure that the moving_mean is updated as if the sample mean is
        # calculated over all replicas.
        for i, expected_moving_mean in enumerate(expected_moving_means):
          expected_moving_means[i] -= ((
              expected_moving_mean - averaged_batch_mean(i)) * (1.0 - momentum))
          self.assertNear(expected_moving_means[i], moving_means[i], 0.0001)

  @ds_combinations.generate(
      combinations.times(
          combinations.combine(loss_reduction=[
              losses_impl.Reduction.SUM, losses_impl.Reduction.MEAN,
              losses_impl.Reduction.SUM_OVER_BATCH_SIZE,
              losses_impl.Reduction.SUM_OVER_NONZERO_WEIGHTS
          ]),
          combinations.times(
              combinations.combine(distribution=[
                  strategy_combinations.one_device_strategy,
                  strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
                  strategy_combinations.mirrored_strategy_with_two_gpus
              ]),
              combinations.times(
                  combinations.combine(optimizer_fn=optimizer_combinations
                                       .gradient_descent_optimizer_v1_fn),
                  combinations.combine(
                      mode=["graph"], use_callable_loss=[True, False]) +
                  combinations.combine(
                      mode=["eager"], use_callable_loss=[True])) +
              combinations.times(
                  combinations.combine(optimizer_fn=optimizer_combinations
                                       .gradient_descent_optimizer_keras_v2_fn),
                  combinations.combine(
                      mode=["graph", "eager"], use_callable_loss=[True]))) +
          combinations.combine(
              distribution=[strategy_combinations.tpu_strategy],
              optimizer_fn=optimizer_combinations
              .gradient_descent_optimizer_v1_fn,
              mode=["graph"],
              use_callable_loss=[True, False]) + combinations.combine(
                  distribution=[strategy_combinations.tpu_strategy],
                  optimizer_fn=optimizer_combinations
                  .gradient_descent_optimizer_keras_v2_fn,
                  mode=["graph"],
                  use_callable_loss=[True])))
  def testMeanVsSum(self, distribution, optimizer_fn, loss_reduction,
                    use_callable_loss):
    with distribution.scope():
      all_vars = []

      def model_fn(inputs):
        x, y = inputs
        w = variable_scope.get_variable("w", initializer=[[2.]])
        all_vars.append(w)

        def loss_fn():
          # Use fixed initialization to make the steps deterministic.
          predict = math_ops.matmul(x, w)
          loss = losses_impl.mean_squared_error(
              y, predict, reduction=loss_reduction)
          if loss_reduction == losses_impl.Reduction.SUM:
            return loss
          return loss / distribution.num_replicas_in_sync

        optimizer = optimizer_fn()  # GradientDescent with 0.2 learning rate

        if isinstance(optimizer, optimizer_v2.OptimizerV2):
          return optimizer.minimize(loss_fn, [w])
        else:
          if use_callable_loss:
            return optimizer.minimize(loss_fn)
          else:
            return optimizer.minimize(loss_fn())

      def dataset_fn():
        features = dataset_ops.Dataset.from_tensors([[2.], [7.]])
        labels = dataset_ops.Dataset.from_tensors([[6.], [21.]])
        return dataset_ops.Dataset.zip((features, labels)).repeat()

      def step_fn(ctx, inputs):
        del ctx  # Unused
        return distribution.group(
            distribution.extended.call_for_each_replica(
                model_fn, args=(inputs,)))

      iterator = self._get_iterator(distribution, dataset_fn)

      def run_step():
        return distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=1).run_op

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          run_step = sess.make_callable(run_step())
      self.evaluate(variables_lib.global_variables_initializer())

      run_step()

      v = all_vars[0]
      self.assertTrue(all(v is vi for vi in all_vars[1:]))
      weight = numpy.squeeze(self.evaluate(v))
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
      # So unreplicated the update to w with lr=0.001 is -0.2 * -106 = 0.106
      # with sum loss reduction, or 0.053 with mean.
      if loss_reduction == losses_impl.Reduction.SUM:
        # Note that the "distribution.num_replicas_in_sync" factor will go away
        # once we split the input across replicas, instead of pulling a complete
        # batch of input per replica.
        self.assertNear(weight, 2 + 0.106 * distribution.num_replicas_in_sync,
                        0.0001)
      else:
        # One of the mean loss reductions.
        self.assertNear(weight, 2 + 0.053, 0.0001)

  @ds_combinations.generate(
      combinations.times(
          optimizer_combinations.distributions_and_v1_and_v2_optimizers(),
          combinations.combine(mode=["graph", "eager"]),
          combinations.combine(is_tpu=[False])) + combinations.combine(
              distribution=[strategy_combinations.tpu_strategy],
              optimizer_fn=optimizer_combinations.optimizers_v1_and_v2,
              mode=["graph"],
              is_tpu=[True]))
  def testRunStepsWithOutputContext(self, distribution, optimizer_fn, is_tpu):
    with distribution.scope():
      def dataset_fn():
        dataset = dataset_ops.Dataset.from_tensors([[1.]]).repeat()
        # TODO(priyag): batch with drop_remainder=True causes shapes to be
        # fully defined for TPU. Remove this when XLA supports dynamic shapes.
        return dataset.batch(batch_size=1, drop_remainder=True)

      optimizer = optimizer_fn()
      layer = core.Dense(1, use_bias=True)

      key1 = "foo"
      value1 = "bar"

      def model_fn(output_context, x):
        """A very simple model written by the user."""
        def loss_fn():
          y = array_ops.reshape(layer(x), []) - constant_op.constant(1.)
          return y * y

        if isinstance(optimizer, optimizer_v2.OptimizerV2):
          train_op = optimizer.minimize(
              loss_fn, lambda: layer.trainable_variables)
        else:
          train_op = optimizer.minimize(loss_fn)
        loss = loss_fn()
        output_context.set_last_step_output(
            name="replica_loss_reduced",
            output=loss,
            reduce_op=reduce_util.ReduceOp.MEAN)
        output_context.set_non_tensor_output(key1, value1)
        return (train_op, loss)

      def step_fn(output_context, inputs):
        (train_op, loss) = distribution.extended.call_for_each_replica(
            model_fn, args=(output_context, inputs))
        output_context.set_last_step_output(
            name="cross_replica_loss_reduced",
            output=loss,
            reduce_op=reduce_util.ReduceOp.MEAN)
        output_context.set_last_step_output(
            name="cross_replica_loss_not_reduced",
            output=loss)
        return distribution.group(train_op)

      iterator = self._get_iterator(distribution, dataset_fn)

      def run_step():
        initial_loss = lambda: constant_op.constant(1e7)
        # Initial values corresponding to reduced losses are just single
        # tensors. But for non reduced losses, we need to have initial
        # values that are of the same structure as non reduced losses. In
        # MirroredStrategy, this will be a list of losses, in TPUStrategy
        # it will be single tensor. Using `call_for_each_replica` followed
        # by `experimental_local_results` gives us the desired initial
        # value structure.
        not_reduced = distribution.experimental_local_results(
            distribution.extended.call_for_each_replica(initial_loss))
        initial_loop_values = {
            "replica_loss_reduced": initial_loss(),
            "cross_replica_loss_reduced": initial_loss(),
            "cross_replica_loss_not_reduced": not_reduced,
        }
        ctx = distribution.extended.experimental_run_steps_on_iterator(
            step_fn, iterator, iterations=2,
            initial_loop_values=initial_loop_values)

        self.assertEqual({key1: (value1,)}, ctx.non_tensor_outputs)
        self._verify_loss_output(
            initial_loss(),
            loss_output=ctx.last_step_outputs["replica_loss_reduced"],
            reduced=True, distribution=distribution)
        self._verify_loss_output(
            initial_loss(),
            loss_output=ctx.last_step_outputs["cross_replica_loss_reduced"],
            reduced=True, distribution=distribution)
        self._verify_loss_output(
            initial_loss(),
            loss_output=ctx.last_step_outputs["cross_replica_loss_not_reduced"],
            reduced=False, distribution=distribution)
        return (ctx.run_op, ctx.last_step_outputs["replica_loss_reduced"])

      if not context.executing_eagerly():
        with self.cached_session() as sess:
          run_step = sess.make_callable(run_step())
      self.evaluate(variables_lib.global_variables_initializer())

      weights, biases, losses = [], [], []
      for _ in range(5):
        _, loss = run_step()
        losses.append(loss)
        weights.append(self.evaluate(layer.kernel))
        biases.append(self.evaluate(layer.bias))

      loss_is_not_increasing = all(y <= x for x, y in zip(losses, losses[1:]))
      self.assertTrue(loss_is_not_increasing)

      error = abs(
          numpy.add(numpy.squeeze(weights), numpy.squeeze(biases)) - 1)
      error_is_not_increasing = all(y <= x for x, y in zip(error, error[1:]))
      self.assertTrue(error_is_not_increasing)

  def _verify_loss_output(self, initial_loss, loss_output, reduced,
                          distribution):
    if not reduced:
      self.assertLen(distribution.experimental_local_results(loss_output),
                     distribution.num_replicas_in_sync)
      loss_tensor = distribution.reduce(reduce_util.ReduceOp.MEAN, loss_output,
                                        axis=None)
    else:
      unwrapped_output = distribution.experimental_local_results(loss_output)
      self.assertLen(unwrapped_output, 1)
      loss_tensor = unwrapped_output[0]
    self.assertEqual(initial_loss.dtype, loss_tensor.dtype)
    self.assertEqual(initial_loss.shape, loss_tensor.shape)

  @ds_combinations.generate(
      optimizer_combinations.distributions_and_v2_optimizers())
  def test_empty_var_list(self, distribution, optimizer_fn):
    opt = optimizer_fn()
    with distribution.scope():

      def run_fn():
        opt.minimize(lambda: constant_op.constant(1.), [])
        opt.apply_gradients([])

      distribution.run(run_fn)


if __name__ == "__main__":
  test.main()
