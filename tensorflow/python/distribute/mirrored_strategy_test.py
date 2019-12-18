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
"""Tests for MirroredStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import values
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import training as keras_training
from tensorflow.python.keras.layers import core as keras_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import server_lib


GPU_TEST = "test_gpu" in sys.argv[0]


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.mirrored_strategy_with_two_gpus,
        ],
        mode=["graph", "eager"]))
class MirroredTwoDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase,
    parameterized.TestCase):

  def testMinimizeLoss(self, distribution):
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution)

  def testReplicaId(self, distribution):
    self._test_replica_id(distribution)

  def testNumReplicasInSync(self, distribution):
    self.assertEqual(2, distribution.num_replicas_in_sync)

  def testCallAndMergeExceptions(self, distribution):
    self._test_call_and_merge_exceptions(distribution)

  def testRunRegroupError(self, distribution):
    def run_fn():
      replica_id = int(self.evaluate(_replica_id()))
      # Generates a list with different lengths on different devices.
      # Will fail in _regroup() (if more than one device).
      return list(range(replica_id))

    with distribution.scope(), self.assertRaises(AssertionError):
      distribution.extended.call_for_each_replica(run_fn)

  def testReduceToCpu(self, distribution):
    with distribution.scope():
      result = distribution.extended.call_for_each_replica(_replica_id)
      reduced = distribution.reduce(reduce_util.ReduceOp.SUM, result, axis=None)
      expected = sum(range(distribution.num_replicas_in_sync))
      self.assertEqual(expected, self.evaluate(reduced))

  def reduce_axis_helper(self, distribution, replica_squared_fn):
    with distribution.scope():
      num_replicas = distribution.num_replicas_in_sync
      result = distribution.extended.call_for_each_replica(replica_squared_fn)
      # sum
      reduced = distribution.reduce(reduce_util.ReduceOp.SUM, result, axis=0)
      expected = sum(x * (x + 1) for x in range(num_replicas))
      self.assertNear(expected, self.evaluate(reduced), 0.00001)

      # mean
      reduced = distribution.reduce(reduce_util.ReduceOp.MEAN, result, axis=0)
      expected /= sum(x + 1 for x in range(num_replicas))
      self.assertNear(expected, self.evaluate(reduced), 0.00001)

  def testReduceAxisToCpu(self, distribution):
    for dtype in (dtypes.float32, dtypes.int32):
      def replica_squared_fn(dtype=dtype):
        # Lists with different lengths on different replicas.
        replica_id = _replica_id_as_int()
        return math_ops.cast([replica_id] * (replica_id + 1), dtype)

      self.reduce_axis_helper(distribution, replica_squared_fn)

  def set_v2_tensorshape(self, v2):
    if v2:
      tensor_shape.enable_v2_tensorshape()
    else:
      tensor_shape.disable_v2_tensorshape()

  def testReduceAxisToCpuUnknownShape(self, distribution):
    original_v2 = tensor_shape._TENSORSHAPE_V2_OVERRIDE  # pylint: disable=protected-access
    try:
      for v2 in (False, True):
        self.set_v2_tensorshape(v2)
        for dtype in (dtypes.float32, dtypes.int32):
          for shape in ((None,), None):  # Test both unknown size and rank.
            def replica_squared_fn(dtype=dtype, shape=shape):
              # Lists with different lengths on different replicas.
              replica_id = _replica_id_as_int()
              tensor = math_ops.cast([replica_id] * (replica_id + 1), dtype)
              # Erase shape information
              return array_ops.placeholder_with_default(tensor, shape=shape)

            self.reduce_axis_helper(distribution, replica_squared_fn)
    finally:
      self.set_v2_tensorshape(original_v2)

  def testReplicateDataset(self, distribution):
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i+1] for i in range(0, 10, 2)]
    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=2,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterable(distribution, input_fn, expected_values)

  def testMakeInputFnIteratorWithDataset(self, distribution):
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=2,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    iterator = distribution.make_input_fn_iterator(input_fn)
    self._test_input_fn_iterator(iterator, distribution.extended.worker_devices,
                                 expected_values)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    def fn():
      dataset = dataset_ops.Dataset.range(2).interleave(
          (lambda _: dataset_ops.Dataset.range(10)), cycle_length=2)
      it = dataset_ops.make_one_shot_iterator(dataset)
      return it.get_next
    expected_values = [[i, i] for i in range(0, 10)]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=2,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    iterator = distribution.make_input_fn_iterator(input_fn)
    self._test_input_fn_iterator(iterator, distribution.extended.worker_devices,
                                 expected_values, test_reinitialize=False,
                                 ignore_order=True)

  def testNumpyDataset(self, distribution):
    self._test_numpy_dataset(distribution)

  def testGlobalStepUpdate(self, distribution):
    self._test_global_step_update(distribution)

  def testRun(self, distribution):
    self._test_run(distribution)

  def testAllReduceSum(self, distribution):
    self._test_all_reduce_sum(distribution)

  def testAllReduceSumGradients(self, distribution):
    self._test_all_reduce_sum_gradients(distribution)

  def testAllReduceSumGradientTape(self, distribution):
    self._test_all_reduce_sum_gradient_tape(distribution)

  def testAllReduceMean(self, distribution):
    self._test_all_reduce_mean(distribution)

  def testAllReduceMeanGradients(self, distribution):
    self._test_all_reduce_mean_gradients(distribution)

  def testAllReduceMeanGradientTape(self, distribution):
    self._test_all_reduce_mean_gradient_tape(distribution)

  def testSummaryForReplicaZeroOnly(self, distribution):
    self._test_summary_for_replica_zero_only(distribution)

  def testTrainableVariables(self, distribution):
    self._test_trainable_variable(distribution)


def one_device_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_one_cpu,
          strategy_combinations.mirrored_strategy_with_one_gpu,
      ],
      mode=["graph", "eager"])


@combinations.generate(one_device_combinations())
class MirroredOneDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    strategy_test_lib.OneDeviceDistributionTestBase,
    parameterized.TestCase):

  def testMinimizeLoss(self, distribution):
    if context.executing_eagerly():
      self._test_minimize_loss_eager(distribution)
    else:
      self._test_minimize_loss_graph(distribution)

  def testReplicaId(self, distribution):
    self._test_replica_id(distribution)

  def testCallAndMergeExceptions(self, distribution):
    self._test_call_and_merge_exceptions(distribution)

  def testRun(self, distribution):
    self._test_run(distribution)

  def testAllReduceSum(self, distribution):
    self._test_all_reduce_sum(distribution)

  def testAllReduceSumGradients(self, distribution):
    self._test_all_reduce_sum_gradients(distribution)

  def testAllReduceSumGradientTape(self, distribution):
    self._test_all_reduce_sum_gradient_tape(distribution)

  def testAllReduceMean(self, distribution):
    self._test_all_reduce_mean(distribution)

  def testAllReduceMeanGradients(self, distribution):
    self._test_all_reduce_mean_gradients(distribution)

  def testAllReduceMeanGradientTape(self, distribution):
    self._test_all_reduce_mean_gradient_tape(distribution)


class MirroredStrategyVariableCreatorStackTest(
    test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          ],
          mode=["graph"]))
  def testCreatorStacksAreThreadLocal(self, distribution):
    def model_fn():
      replica_id_str = str(self.evaluate(_replica_id()))

      def thread_creator_fn(next_creator, *args, **kwargs):
        return next_creator(*args, **kwargs) + ":thread_" + replica_id_str

      with variable_scope.variable_creator_scope(thread_creator_fn):
        # Create a variable in this scope.
        v = variable_scope.variable(1.0)

        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    def main_thread_creator(next_creator, *args, **kwargs):
      # We are not using the underlying next_creator for test purposes.
      del next_creator, args, kwargs
      return "main_thread"

    with context.graph_mode(), \
        distribution.scope(), \
        variable_scope.variable_creator_scope(main_thread_creator):
      result = distribution.extended.call_for_each_replica(model_fn)
      result = distribution.experimental_local_results(result)
      expected = ("main_thread:thread_0", "main_thread:thread_1")
      self.assertEqual(expected, result)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class MirroredStrategyCallForEachReplicaTest(test.TestCase):

  def testExecutingEagerlyOutsideFunction(self, distribution):
    """Verify we preserve the value of executing_eagerly_outside_functions()."""
    def model_fn():
      return ops.executing_eagerly_outside_functions()

    originally = ops.executing_eagerly_outside_functions()
    with distribution.scope():
      in_scope = ops.executing_eagerly_outside_functions()
      in_model_fn = distribution.extended.call_for_each_replica(model_fn)
      unwrapped = distribution.experimental_local_results(in_model_fn)
      self.assertEqual(in_scope, unwrapped[0])
      self.assertEqual(in_scope, originally)

    # Verify this all again, but this time in a FuncGraph.
    with func_graph.FuncGraph("fg").as_default(), distribution.scope():
      in_scope = ops.executing_eagerly_outside_functions()
      in_model_fn = distribution.extended.call_for_each_replica(model_fn)
      unwrapped = distribution.experimental_local_results(in_model_fn)
      self.assertEqual(in_scope, unwrapped[0])
      self.assertEqual(in_scope, originally)

  def testFunctionInCallForEachReplica(self, distribution):
    traces = []
    @def_function.function
    def model_fn():
      traces.append(1)
      return ds_context.get_replica_context().replica_id_in_sync_group

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertEqual((0, 1), self.evaluate(result.values))
      self.assertLen(traces, distribution.num_replicas_in_sync)

  def testFunctionInCallForEachReplicaInsideAnotherFunction(self, distribution):
    traces = []
    @def_function.function
    def model_fn():
      traces.append(1)
      return ds_context.get_replica_context().replica_id_in_sync_group

    @def_function.function
    def step():
      return distribution.extended.call_for_each_replica(model_fn)

    with distribution.scope():
      result = step()
      self.assertEqual((0, 1), self.evaluate(result.values))
      self.assertLen(traces, distribution.num_replicas_in_sync)

  def testNestedFunctionInCallForEachReplicaWithMergeCall(self, distribution):
    def merge_fn(_):
      pass

    @def_function.function
    def model_fn():
      def body_fn(i):
        ds_context.get_replica_context().merge_call(merge_fn)
        return i + 1
      return control_flow_ops.while_loop_v2(lambda i: i < 2, body_fn, [0])

    with distribution.scope():
      with self.assertRaisesRegexp(
          RuntimeError, "`merge_call` called while defining a new graph."):
        distribution.extended.call_for_each_replica(model_fn)

  def testFunctionInCallForEachReplicaWithMergeCall(self, distribution):
    def merge_fn(_):
      pass

    @def_function.function
    def model_fn():
      ds_context.get_replica_context().merge_call(merge_fn)
      return 0.

    with distribution.scope():
      self.assertEqual(
          self.evaluate(distribution.extended.call_for_each_replica(model_fn)),
          0.)

  def testFunctionInCallForEachReplicaCached(self, distribution):
    traces = []

    @def_function.function
    def model_fn():
      traces.append(None)

    self.assertEmpty(traces)

    for i in range(10):
      distribution.extended.call_for_each_replica(model_fn)

      if i == 0:
        num_devices = len(traces)
        self.assertGreater(num_devices, 0)
      else:
        # model_fn should not have been re-evaluated so the length should remain
        # the same.
        self.assertLen(traces, num_devices)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph"]))
class MirroredStrategyNameScopeTest(test.TestCase):
  # NOTE(priyag): Names and name scopes are ignored in eager, hence we are not
  # testing this in eager mode.

  def testNameScope(self, distribution):
    def model_fn():
      with ops.name_scope("foo"):
        a = constant_op.constant(1.0, name="a")
        ds_context.get_replica_context().merge_call(lambda _: _)
        b = constant_op.constant(1.0, name="b")
      return a, b

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        result = distribution.extended.call_for_each_replica(model_fn)
        self.assertEqual(2, len(result))
        for v, name in zip(result, ["a", "b"]):
          self.assertIsInstance(v, values.DistributedValues)
          v0, v1 = distribution.experimental_local_results(v)
          self.assertEqual("main/foo/" + name + ":0", v0.name)
          self.assertEqual("main/replica_1/foo/" + name + ":0", v1.name)

  def testWithDefaultName(self, distribution):
    def model_fn():
      with ops.name_scope(None, "foo"):
        a = constant_op.constant(1.0, name="a")
        ds_context.get_replica_context().merge_call(lambda _: _)
        b = constant_op.constant(2.0, name="b")
      return a, b

    with context.graph_mode(), distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertEqual(2, len(result))
      for v, name in zip(result, ["a", "b"]):
        self.assertIsInstance(v, values.DistributedValues)
        v0, v1 = distribution.experimental_local_results(v)
        self.assertEqual("foo/" + name + ":0", v0.name)
        self.assertEqual("replica_1/foo/" + name + ":0", v1.name)

  # variable_scope.variable() respects name scopes when creating
  # variables. On the other hand variable_scope.get_variable() ignores name
  # scopes but respects variable scope when creating variables. We test both
  # methods of creating variables to make sure that we have the same
  # variable names in both cases.
  def testNameScopeWithVariable(self, distribution):
    def in_cross_replica(_):
      c = variable_scope.variable(1.0, name="c")
      return c

    def model_fn():
      b = variable_scope.variable(1.0, name="b")
      with ops.name_scope("foo"):
        c = ds_context.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        a = variable_scope.variable(1.0, name="a")
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.experimental_local_results(a)
      b0, b1 = distribution.experimental_local_results(result_b)
      c0, c1 = distribution.experimental_local_results(result_c)
      self.assertEqual("main/a:0", a0.name)
      self.assertEqual("main/a/replica_1:0", a1.name)
      self.assertEqual("main/b:0", b0.name)
      self.assertEqual("main/b/replica_1:0", b1.name)
      self.assertEqual("main/foo/c:0", c0.name)
      self.assertEqual("main/foo/c/replica_1:0", c1.name)

  def testNameScopeWithGetVariable(self, distribution):
    def in_cross_replica(_):
      c = variable_scope.get_variable("c", [1])
      return c

    def model_fn():
      b = variable_scope.get_variable("b", [1])
      with ops.name_scope("foo"):
        c = ds_context.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with ops.name_scope("main"):
        a = variable_scope.get_variable("a", [1])
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.experimental_local_results(a)
      b0, b1 = distribution.experimental_local_results(result_b)
      c0, c1 = distribution.experimental_local_results(result_c)
      self.assertEqual("a:0", a0.name)
      self.assertEqual("a/replica_1:0", a1.name)
      self.assertEqual("b:0", b0.name)
      self.assertEqual("b/replica_1:0", b1.name)
      self.assertEqual("c:0", c0.name)
      self.assertEqual("c/replica_1:0", c1.name)

  def testVariableScopeWithGetVariable(self, distribution):

    def in_cross_replica(_):
      c = variable_scope.get_variable("c", [1])
      return c

    def model_fn():
      b = variable_scope.get_variable("b", [1])
      with variable_scope.variable_scope("foo"):
        c = ds_context.get_replica_context().merge_call(in_cross_replica)
      return b, c

    with context.graph_mode(), distribution.scope():
      with variable_scope.variable_scope("main"):
        a = variable_scope.get_variable("a", [1])
        result = distribution.extended.call_for_each_replica(model_fn)
      result_b = result[0]
      result_c = result[1]
      self.assertIsInstance(result_b, values.DistributedValues)
      self.assertIsInstance(result_c, values.DistributedValues)
      a0, a1 = distribution.experimental_local_results(a)
      b0, b1 = distribution.experimental_local_results(result_b)
      c0, c1 = distribution.experimental_local_results(result_c)
      self.assertEqual("main/a:0", a0.name)
      self.assertEqual("main/a/replica_1:0", a1.name)
      self.assertEqual("main/b:0", b0.name)
      self.assertEqual("main/b/replica_1:0", b1.name)
      self.assertEqual("main/foo/c:0", c0.name)
      self.assertEqual("main/foo/c/replica_1:0", c1.name)


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored3Devices",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(
                    ["/device:GPU:0", "/device:GPU:1", "/device:CPU:0"]),
                required_gpus=2)
        ],
        mode=["graph", "eager"]))
class MirroredThreeDeviceDistributionTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):

  def testThreeDevices(self, distribution):
    def model_fn():
      v = variable_scope.variable(1.0, name="foo")
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(result, values.MirroredVariable)
      self.assertEqual("foo:0", result.name)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class MirroredVariableUpdateTest(test.TestCase):
  # The following tests check assign, assign_add and assign_sub on Mirrored
  # variables in replica and cross replica context.

  def testAssignMirroredVarReplicaContextWithoutAggregationType(self,
                                                                distribution):
    # Test that we always have an aggregation type set on the mirrored variable
    # if we assign to it in replica mode.
    def var_fn():
      v = variable_scope.variable(1.0, name="foo")
      return v

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        return mirrored_var.assign(5.0)

      with self.assertRaisesRegexp(
          ValueError, "You must specify an aggregation method to update a "
                      "MirroredVariable in Replica Context. You can do so by"):
        self.evaluate(distribution.experimental_local_results(
            distribution.extended.call_for_each_replica(model_fn)))

  def testAssignMirroredVarReplicaContextWithSum(self, distribution):
    # Test that we don't reduce a non-per-replica value with the "sum"
    # aggregation type.
    def var_fn():
      v = variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.SUM)
      return v

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())

      def model_fn():
        return mirrored_var.assign(5.0)

      with self.assertRaisesRegexp(
          ValueError, "A non-DistributedValues value 5.0 cannot be reduced "
          "with the given reduce op ReduceOp.SUM."):
        self.evaluate(distribution.experimental_local_results(
            distribution.extended.call_for_each_replica(model_fn)))

  def testAssignMirroredVarCrossDeviceContext(self, distribution):
    def var_fn():
      return variable_scope.variable(1.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign(6.0))
      self.assertEqual(6.0, mirrored_var_result)

  def testAssignMirroredVarReplicaContext(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(
            ds_context.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign(value)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(0.5, self.evaluate(mirrored_var))

  def testAssignMirroredVarReplicaContextWithSingleValue(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        return mirrored_var.assign(5.0)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(5.0, self.evaluate(mirrored_var))

  def testAssignAddMirroredVarCrossDeviceContext(self, distribution):
    def var_fn():
      return variable_scope.variable(1.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      # read_value == True
      mirrored_var_result = self.evaluate(
          mirrored_var.assign_add(6.0, read_value=True))
      self.assertEqual(7.0, mirrored_var_result)
      self.assertEqual(7.0, self.evaluate(mirrored_var.values[0]))
      self.assertEqual(7.0, self.evaluate(mirrored_var.values[1]))
      self.assertEqual(
          distribution.extended.worker_devices[0], mirrored_var.devices[0])
      self.assertEqual(
          distribution.extended.worker_devices[1], mirrored_var.devices[1])

      # read_value == False
      self.evaluate(mirrored_var.assign_add(2.0, read_value=False))
      self.assertEqual(9.0, self.evaluate(mirrored_var.values[0]))
      self.assertEqual(9.0, self.evaluate(mirrored_var.values[1]))
      self.assertEqual(
          distribution.extended.worker_devices[0], mirrored_var.devices[0])
      self.assertEqual(
          distribution.extended.worker_devices[1], mirrored_var.devices[1])

  def testAssignAddMirroredVarReplicaContext(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(
            ds_context.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign_add(value)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(1.5, self.evaluate(mirrored_var))

  def testAssignAddMirroredVarReplicaContextWithSingleValue(self, distribution):
    def var_fn():
      return variable_scope.variable(
          1.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(1.0, self.evaluate(mirrored_var))

      def model_fn():
        return mirrored_var.assign_add(5.0)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(6.0, self.evaluate(mirrored_var))

  def testAssignSubMirroredVarCrossDeviceContext(self, distribution):
    def var_fn():
      return variable_scope.variable(5.0, name="foo")

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))
      mirrored_var_result = self.evaluate(mirrored_var.assign_sub(2.0))
      self.assertEqual(3.0, mirrored_var_result)
      self.assertEqual(3.0, self.evaluate(mirrored_var.values[0]))
      self.assertEqual(3.0, self.evaluate(mirrored_var.values[1]))
      self.assertEqual(
          distribution.extended.worker_devices[0], mirrored_var.devices[0])
      self.assertEqual(
          distribution.extended.worker_devices[1], mirrored_var.devices[1])

  def testAssignSubMirroredVarReplicaContext(self, distribution):
    def var_fn():
      return variable_scope.variable(
          5.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))

      def model_fn():
        value = math_ops.cast(
            ds_context.get_replica_context().replica_id_in_sync_group,
            mirrored_var.dtype)
        return mirrored_var.assign_sub(value)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(4.5, self.evaluate(mirrored_var))

  def testAssignSubMirroredVarReplicaContextWithSingleValue(self, distribution):
    def var_fn():
      return variable_scope.variable(
          5.0, name="foo", aggregation=variable_scope.VariableAggregation.MEAN)

    with distribution.scope():
      mirrored_var = distribution.extended.call_for_each_replica(var_fn)
      self.assertIsInstance(mirrored_var, values.MirroredVariable)
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(5.0, self.evaluate(mirrored_var))

      def model_fn():
        return mirrored_var.assign_sub(1.0)

      self.evaluate(distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn)))
      self.assertEqual(4.0, self.evaluate(mirrored_var))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class MirroredAndSyncOnReadVariableInitializerTest(test.TestCase):

  def testAssignMirroredVarInitializer(self, distribution):
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def var_fn():
        v = variable_scope.variable(1.0, name="foo")
        return v

      with distribution.scope():
        mirrored_var = distribution.extended.call_for_each_replica(var_fn)
        self.assertIsInstance(mirrored_var, values.MirroredVariable)
        self.assertFalse(self.evaluate(mirrored_var.is_initialized()))
        self.evaluate(mirrored_var.initializer)
        self.assertTrue(self.evaluate(mirrored_var.is_initialized()))

  def testAssignReplicaLocalVarInitializer(self, distribution):
    # This test is not eager compatible since in eager variables are initialized
    # upon construction instead of once the initialization op is run.
    with context.graph_mode():
      def model_fn():
        v_sum = variable_scope.variable(
            1.0,
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        self.assertIsInstance(v_sum, values.SyncOnReadVariable)
        return v_sum

      with distribution.scope():
        sync_on_read_var = distribution.extended.call_for_each_replica(
            model_fn)
        self.assertIsInstance(sync_on_read_var, values.SyncOnReadVariable)
        self.assertFalse(self.evaluate(sync_on_read_var.is_initialized()))
        self.evaluate(sync_on_read_var.initializer)
        self.assertTrue(self.evaluate(sync_on_read_var.is_initialized()))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class SyncOnReadVariableAssignTest(test.TestCase):

  def testAssignReplicaLocalVarSumAggregation(self, distribution):
    def model_fn():
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      return v_sum

    with distribution.scope():
      sync_on_read_var = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(sync_on_read_var, values.SyncOnReadVariable)
      self.evaluate(variables.global_variables_initializer())
      # Each replica has a value of 1.0 assigned to it in replica context.
      # When we read the value using `read_var` we should see the SUM of each of
      # values on each of the replicas.
      self.assertEqual(2.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))
      # Assigning 6.0 in cross replica context will assign a value of
      # 6.0/num_replicas to each replica.
      tlv_ops = sync_on_read_var.assign(6.0)
      self.evaluate(tlv_ops)
      # On reading the sync on read var we should get the assigned value back.
      # The value on all the replicas are added before being returned by
      # `read_var`.
      self.assertEqual(6.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))

  def testAssignReplicaLocalVarMeanAggregation(self, distribution):
    def model_fn():
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      return v_sum

    with distribution.scope():
      sync_on_read_var = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(sync_on_read_var, values.SyncOnReadVariable)
      self.evaluate(variables.global_variables_initializer())
      # Each replica has a value of 1.0 assigned to it in replica context.
      # When we read the value using `read_var` we should see the MEAN of values
      # on all replicas which is the value assigned in replica context.
      self.assertEqual(1.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))
      tlv_ops = sync_on_read_var.assign(6.0)
      self.evaluate(tlv_ops)
      # On reading the sync on read var we should get the MEAN of all values
      # which is equal to the value assigned.
      self.assertEqual(6.0, self.evaluate(
          distribution.extended.read_var(sync_on_read_var)))


class MockModel(object):

  def __init__(self, two_variables=False):
    self.variables = []
    self.variables.append(variable_scope.variable(1.25, name="dummy_var1"))
    if two_variables:
      self.variables.append(variable_scope.variable(2.0, name="dummy_var2"))

  def __call__(self, factor=2):
    x = factor * self.variables[0]
    if len(self.variables) > 1:
      x += self.variables[1]
    return x


class MiniModel(keras_training.Model):
  """Minimal model for mnist.

  Useful for testing and debugging on slow TPU simulators.
  """

  def __init__(self):
    super(MiniModel, self).__init__(name="")
    self.fc = keras_core.Dense(1, name="fc", kernel_initializer="ones",
                               bias_initializer="ones")

  def call(self, inputs, training=True):
    inputs = array_ops.ones([1, 10])
    return self.fc(inputs)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
        ],
        mode=["graph", "eager"]))
class MirroredStrategyDefunTest(test.TestCase):

  def _call_and_check(self, distribution, model_fn, inputs, expected_result,
                      defuns, two_variables=False):
    cpu_dev = device_util.canonicalize("CPU:0")
    gpu_dev = device_util.canonicalize("GPU:0")
    devices = [cpu_dev, gpu_dev]

    with distribution.scope():
      mock_model = MockModel(two_variables)
      self.evaluate(variables.global_variables_initializer())

      result = distribution.extended.call_for_each_replica(
          model_fn, args=[mock_model] + inputs)
      for r in range(len(devices)):
        device_result = values.select_replica(r, result)
        device_expected_result = values.select_replica(r, expected_result)
        self.assertAllClose(device_expected_result,
                            self.evaluate(device_result))

      for defun in defuns:
        # `Function`s are specialized to the current device stack, so
        # call_for_each has one trace per device. To check that the expected set
        # of variables was accessed on each trace, we first retrieve each
        # device-specific graph function.
        per_replica_graph_functions = (
            distribution.extended.call_for_each_replica(
                defun.get_concrete_function, args=[mock_model] + inputs))
        for i in range(len(devices)):
          graph_function = per_replica_graph_functions.values[i]
          # TODO(b/129555712): re-enable an assertion here that the two sets of
          # variables are the same.
          # self.assertEqual(set(graph_function.graph.variables),
          #  set(mock_model.variables))
          del graph_function

  def testVariableInDefun(self, distribution):
    @function.defun
    def times_two(mock_model):
      return mock_model()

    def model_fn(mock_model):
      return times_two(mock_model)

    self._call_and_check(distribution, model_fn, [], 2.5, [times_two])

  def testVariableInNestedDefun(self, distribution):
    @function.defun
    def times_two(mock_model):
      return mock_model()

    @function.defun
    def two_x_plus_one(mock_model):
      return times_two(mock_model) + 1

    def model_fn(mock_model):
      return two_x_plus_one(mock_model)

    self._call_and_check(distribution, model_fn, [], 3.5,
                         [times_two, two_x_plus_one])

  def testTwoVariablesInNestedDefun(self, distribution):
    @function.defun
    def fn1(mock_model):
      return mock_model()

    @function.defun
    def fn2(mock_model):
      return fn1(mock_model) + 1

    def model_fn(mock_model):
      return fn2(mock_model)

    self._call_and_check(distribution, model_fn, [], 5.5, [fn1, fn2],
                         two_variables=True)

  def testGradientTapeOverNestedDefuns(self, distribution):
    @function.defun
    def fn1(mock_model):
      return mock_model()

    @function.defun
    def fn2(mock_model):
      return fn1(mock_model) + 1

    def model_fn(mock_model):
      with backprop.GradientTape(persistent=True) as gtape:
        result = fn2(mock_model)
      grads = gtape.gradient(result,
                             [v.get() for v in mock_model.variables])
      return grads

    self._call_and_check(distribution, model_fn, [], [2.0, 1.0], [fn1, fn2],
                         two_variables=True)

  def testPassPerReplica(self, distribution):
    @function.defun
    def fn1(mock_model, factor):
      return mock_model(factor)

    factors = values.PerReplica((5.0, 3.0))
    expected_result = values.PerReplica((5.0 * 1.25, 3.0 * 1.25))
    self._call_and_check(distribution, fn1, [factors], expected_result, [fn1])

  def testTrain(self, distribution):
    with distribution.scope():
      mock_model = MiniModel()
      mock_model.call = function.defun(mock_model.call)

      def loss_fn(ctx):
        del ctx
        return mock_model(array_ops.ones([1, 10]))

      gradients_fn = backprop.implicit_grad(loss_fn)
      gradients_fn = optimizer_lib.get_filtered_grad_fn(gradients_fn)
      grads_and_vars = distribution.extended.call_for_each_replica(
          gradients_fn, args=(None,))

      optimizer = gradient_descent.GradientDescentOptimizer(0.25)
      update_ops = optimizer._distributed_apply(distribution, grads_and_vars)  # pylint: disable=protected-access

      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(update_ops)

      updated_var_values = self.evaluate(mock_model.variables)
      # All variables start at 1.0 and get two updates of 0.25.
      self.assertAllEqual(0.5 * np.ones([10, 1]), updated_var_values[0])
      self.assertAllEqual([0.5], updated_var_values[1])


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(
                    devices=mirrored_strategy.all_local_devices(),
                    cross_device_ops=cross_device_ops_lib.MultiWorkerAllReduce([
                        "/job:worker/task:0", "/job:worker/task:1"
                    ], context.num_gpus())),
                required_gpus=1)
        ],
        mode=["graph"]))
class MultiWorkerMirroredStrategyTest(
    multi_worker_test_base.MultiWorkerTestBase,
    strategy_test_lib.DistributionTestBase):

  def _configure_distribution_strategy(self, distribution):
    cluster_spec = server_lib.ClusterSpec({
        "worker": ["/job:worker/task:0", "/job:worker/task:1"]
    })
    distribution.configure(cluster_spec=cluster_spec)

  def test_num_replicas_in_sync(self, distribution):
    self._configure_distribution_strategy(distribution)
    # We calculate the total number of gpus across the workers(2) specified in
    # the cluster spec.
    self.assertEqual(context.num_gpus() * 2, distribution.num_replicas_in_sync)

  def testMinimizeLossGraph(self, distribution):
    self._configure_distribution_strategy(distribution)
    self._test_minimize_loss_graph(distribution, learning_rate=0.05)

  def testDeviceScope(self, distribution):
    """Test the device scope of multi-worker MirroredStrategy."""
    self._configure_distribution_strategy(distribution)
    with distribution.scope():
      a = constant_op.constant(1.)
      with ops.device("/cpu:0"):
        b = constant_op.constant(1.)
      self.assertEqual(a.device, "/job:worker/task:0")
      self.assertEqual(b.device, "/job:worker/task:0/device:CPU:0")

  def testMakeInputFnIteratorWithDataset(self, distribution):
    self._configure_distribution_strategy(distribution)
    dataset_fn = lambda: dataset_ops.Dataset.range(100)
    num_gpus = context.num_gpus()
    num_workers = 2

    expected_values = [[i+j for j in range(num_gpus)] * num_workers
                       for i in range(0, 100, num_gpus)]

    with context.graph_mode(), self.cached_session() as sess:
      # `expected_input_pipeline_id` is None because the input_fn will be called
      # multiple times, each with a different input_pipeline_id.
      input_fn = self._input_fn_to_test_input_context(
          dataset_fn,
          expected_num_replicas_in_sync=num_workers*num_gpus,
          expected_num_input_pipelines=num_workers,
          expected_input_pipeline_id=None)
      iterator = distribution.make_input_fn_iterator(input_fn)
      self._test_input_fn_iterator(
          iterator, distribution.extended.worker_devices, expected_values, sess)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    self._configure_distribution_strategy(distribution)
    def fn():
      dataset = dataset_ops.Dataset.range(100)
      it = dataset_ops.make_one_shot_iterator(dataset)
      return it.get_next
    num_gpus = context.num_gpus()
    num_workers = 2

    expected_values = []
    for i in range(0, 100, num_gpus):
      expected_values.append([i+j for j in range(num_gpus)] * num_workers)

    with context.graph_mode(), self.cached_session() as sess:
      # `expected_input_pipeline_id` is None because the input_fn will be called
      # multiple times, each with a different input_pipeline_id.
      input_fn = self._input_fn_to_test_input_context(
          fn,
          expected_num_replicas_in_sync=num_workers*num_gpus,
          expected_num_input_pipelines=num_workers,
          expected_input_pipeline_id=None)
      iterator = distribution.make_input_fn_iterator(input_fn)
      self._test_input_fn_iterator(
          iterator, distribution.extended.worker_devices, expected_values, sess,
          test_reinitialize=False, ignore_order=True)

  def testUpdateConfigProto(self, distribution):
    distribution.configure(cluster_spec={"worker": ["fake1", "fake2"]})

    config_proto = config_pb2.ConfigProto()
    new_config = distribution.update_config_proto(config_proto)

    # Verify isolate_session_state
    self.assertTrue(new_config.isolate_session_state)


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(
                    devices=["/job:worker/task:0/gpu:{}".format(
                        i) for i in range(context.num_gpus())]),
                required_gpus=1)
        ],
        mode=["graph"]))
class RemoteSingleWorkerMirroredStrategyGraph(
    multi_worker_test_base.SingleWorkerTestBaseGraph,
    strategy_test_lib.RemoteSingleWorkerMirroredStrategyBase):

  def _get_num_gpus(self):
    return context.num_gpus()

  def testNumReplicasInSync(self, distribution):
    self._testNumReplicasInSync(distribution)

  def testMinimizeLoss(self, distribution):
    self._testMinimizeLoss(distribution)

  def testDeviceScope(self, distribution):
    self._testDeviceScope(distribution)

  def testMakeInputFnIteratorWithDataset(self, distribution):
    self._testMakeInputFnIteratorWithDataset(distribution)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    self._testMakeInputFnIteratorWithCallable(distribution)


class MultiWorkerMirroredStrategyTestWithChief(
    multi_worker_test_base.MultiWorkerTestBase,
    strategy_test_lib.DistributionTestBase):

  @classmethod
  def setUpClass(cls):
    """Create a local cluster with 2 workers and 1 chief."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=2, num_ps=0, has_chief=True)
    cls._default_target = "grpc://" + cls._cluster_spec["chief"][0]

  def _make_cross_device_ops(self):
    return cross_device_ops_lib.MultiWorkerAllReduce(
        ["/job:chief/task:0", "/job:worker/task:0", "/job:worker/task:1"],
        context.num_gpus())

  def testMinimizeLossGraph(self):
    with context.graph_mode():
      strategy = mirrored_strategy.MirroredStrategy(
          cross_device_ops=self._make_cross_device_ops())
      strategy.configure(cluster_spec=self._cluster_spec)
      self._test_minimize_loss_graph(strategy, learning_rate=0.05)

  def testMinimizeLossGraphMirroredStrategy(self):
    with context.graph_mode():
      strategy = mirrored_strategy.MirroredStrategy(
          mirrored_strategy.all_local_devices(),
          cross_device_ops=self._make_cross_device_ops())
      strategy.configure(cluster_spec=self._cluster_spec)
      self._test_minimize_loss_graph(strategy, learning_rate=0.05)

  def testMinimizeLossGraphMirroredStrategyWithOneNode(self):
    with context.graph_mode():
      cluster_spec = {}
      cluster_spec["chief"] = self._cluster_spec["chief"]
      tf_config = {"cluster": cluster_spec}
      with test.mock.patch.dict("os.environ",
                                {"TF_CONFIG": json.dumps(tf_config)}):
        strategy = mirrored_strategy.MirroredStrategy()
        if context.num_gpus() > 0:
          self.assertIsInstance(strategy.extended._inferred_cross_device_ops,
                                cross_device_ops_lib.NcclAllReduce)
        else:
          self.assertIsInstance(strategy.extended._inferred_cross_device_ops,
                                cross_device_ops_lib.ReductionToOneDevice)
      self.skipTest("b/130551176, run the following once fixed.")
      self._test_minimize_loss_graph(strategy, learning_rate=0.05)

  def testInitializeFromTFConfig(self):
    with context.graph_mode():
      tf_config = {"cluster": self._cluster_spec}
      with test.mock.patch.dict("os.environ",
                                {"TF_CONFIG": json.dumps(tf_config)}):
        strategy = mirrored_strategy.MirroredStrategy(
            cross_device_ops=self._make_cross_device_ops())
        self.assertEqual(
            max(context.num_gpus(), 1) * 3, strategy.num_replicas_in_sync)

  def testSummaryForReplicaZeroOnly(self):
    with context.graph_mode():
      strategy = mirrored_strategy.MirroredStrategy(
          mirrored_strategy.all_local_devices(),
          cross_device_ops=self._make_cross_device_ops())
      strategy.configure(cluster_spec=self._cluster_spec)
      self._test_summary_for_replica_zero_only(strategy)


class MirroredVariableStopGradientTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_one_gpu,
          ],
          mode=["graph"]))
  def testMirroredVariableAsStopGradient(self, distribution):
    with distribution.scope():
      inp = constant_op.constant(1.0)
      x = variables.Variable(1.0)
      y = inp*x
      grads = gradients.gradients(x, y, stop_gradients=x)
      self.assertIsNone(grads[0])


class FunctionTest(test.TestCase):

  def testBackwardFuctionDevicePlacement(self):
    if context.num_gpus() < 1:
      self.skipTest("At least one GPU is required.")

    devices = [device_util.resolve("/device:GPU:0"),
               device_util.resolve("/device:CPU:0")]
    ms = mirrored_strategy.MirroredStrategy(devices)

    with ms.scope():
      w = variable_scope.variable([1.5], name="w")
      b = variable_scope.variable([0.5], name="b")

    @def_function.function
    def forward(x, w, b):
      return x * w + b
    x = constant_op.constant([1.0], name="x_useless")
    concrete_forward = forward.get_concrete_function(x, w.primary, b.primary)

    with ms.scope():
      def replica_fn():
        with backprop.GradientTape() as t:
          x = constant_op.constant([1.0], name="x")
          loss = concrete_forward(x, w.get(), b.get()) - [1.0]
          return t.gradient(loss, [w, b])

      def step_fn():
        return ms.experimental_run_v2(replica_fn)

      context.enable_run_metadata()
      g1, g2 = step_fn()
      run_metadata = context.export_run_metadata()
      context.disable_run_metadata()
      self.assertEqual(self.evaluate(g1.primary), 1.0)
      self.assertEqual(self.evaluate(g2.primary), 1.0)

      # Verify that this node runs on both devices.
      node_name = "gradients_mul_grad_mul_1_x"
      devices_for_this_node = set()
      for partition_graph in run_metadata.partition_graphs:
        for node in partition_graph.node:
          if node.name == node_name:
            devices_for_this_node.add(node.device)
      self.assertSetEqual(devices_for_this_node, set(devices))

  def testFuctionPreservesAutoGraph(self):
    config.set_logical_device_configuration(
        config.list_physical_devices("CPU")[0],
        [context.LogicalDeviceConfiguration()] * 2)
    ms = mirrored_strategy.MirroredStrategy()

    def f():
      self.assertTrue(converter_testing.is_inside_generated_code())
      return 1

    with ms.scope():
      @def_function.function
      def replica_fn():
        return f()

      ms.experimental_run_v2(replica_fn)


def _replica_id():
  replica_id = ds_context.get_replica_context().replica_id_in_sync_group
  if not isinstance(replica_id, ops.Tensor):
    replica_id = constant_op.constant(replica_id)
  return replica_id


def _replica_id_as_int():
  replica_id = ds_context.get_replica_context().replica_id_in_sync_group
  if isinstance(replica_id, ops.Tensor):
    replica_id = tensor_util.constant_value(replica_id)
  return replica_id


if __name__ == "__main__":
  test.main()
