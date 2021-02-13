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
"""Tests for the distributed values library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util as ds_test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.tracking import util as trackable_utils
from tensorflow.python.types import core
from tensorflow.python.util import nest


def _device_str(d):
  return "/device:GPU:" + str(d)


def _nested_value(d):
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


def _make_mirrored_val(init_val=5.0):
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, _ in zip(devices, ["v", "v/replica"]):
    with ops.device(d):
      v.append(constant_op.constant(init_val))
  return values_lib.Mirrored(v)


def _make_mirrored(distribution=None):
  v = []
  if distribution:
    devices = distribution.extended.worker_devices
  else:
    devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(
          variable_scope.get_variable(
              name=n, initializer=init, use_resource=True))

  if (distribution is not None) and isinstance(distribution, _TPU_STRATEGIES):
    var_cls = tpu_values.TPUMirroredVariable
  else:
    var_cls = values_lib.MirroredVariable
  mirrored = var_cls(distribution, v, variable_scope.VariableAggregation.SUM)
  return mirrored


def mirrored_and_tpu_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"])


class DistributedValuesTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueFromTensor(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    single_value = constant_op.constant(1)
    def value_fn(ctx):
      del ctx
      return single_value

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    self.assertAllEqual(
        ds_test_util.gather(distribution, distributed_values),
        constant_op.constant(1., shape=(distribution.num_replicas_in_sync)))

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueSingleNumpyArrayConstant(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    array_value = np.array([1., 2., 3.])
    def value_fn(ctx):
      del ctx
      return array_value

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    self.assertAllEqual(
        ds_test_util.gather(distribution, distributed_values).numpy(),
        [[1., 2., 3.]] * distribution.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueTupleConstant(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    tuple_value = (1., 2., 3.)
    def value_fn(ctx):
      del ctx
      return tuple_value
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    distributed_values = ds_test_util.gather(distribution, distributed_values)

    # Expected output for 2 replicas:
    # ([1.0, 1.0], [2.0, 2.0], [3.0, 3.0])
    expected = tuple([v for i in range(distribution.num_replicas_in_sync)]
                     for v in tuple_value)
    self.assertAllEqual(distributed_values, expected)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueNestedStructurePerReplica(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    tuple_value = (1., 2., 3.)
    def value_fn(ctx):
      per_replica = []
      for val in tuple_value:
        per_replica.append(val * ctx.replica_id_in_sync_group)
      return tuple(per_replica)
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    distributed_values = ds_test_util.gather(distribution, distributed_values)

    # Expected output for 2 replicas:
    # ([0.0, 1.0], [0.0, 2.0], [0.0, 3.0])
    expected = tuple([v * i for i in range(distribution.num_replicas_in_sync)]
                     for v in tuple_value)
    self.assertAllEqual(distributed_values, expected)

  # NOTE(priyag): Cannot test this with MultiWorkerMirroredStrategy because
  # collective ops do not support SparseTensors.
  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies_minus_default,
          mode=["eager"]
      ))
  def testMakeDistributedValueSpareTensor(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    def value_fn(ctx):
      del ctx
      return sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    local_results = distribution.experimental_local_results(distributed_values)
    for i in range(distribution.num_replicas_in_sync):
      self.assertAllEqual(
          sparse_ops.sparse_tensor_to_dense(local_results[i]),
          [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]])

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueExtractFromArray(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    multiple_values = range(distribution.num_replicas_in_sync)
    def value_fn(ctx):
      return multiple_values[ctx.replica_id_in_sync_group]
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    distributed_values = ds_test_util.gather(distribution, distributed_values)
    expected = range(distribution.num_replicas_in_sync)
    self.assertAllEqual(distributed_values, expected)

  @combinations.generate(
      combinations.combine(
          distribution=(strategy_combinations.all_strategies_minus_default +
                        strategy_combinations.multiworker_strategies),
          mode=["eager"]
      ))
  def testMakeDistributedValueAndRun(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    @def_function.function
    def run():
      multiple_values = range(distribution.num_replicas_in_sync)
      def value_fn(ctx):
        return multiple_values[ctx.replica_id_in_sync_group]
      distributed_values = (
          distribution.experimental_distribute_values_from_function(value_fn))

      def computation(x):
        return math_ops.square(x)

      outputs = ds_test_util.gather(
          distribution,
          distribution.run(computation, args=(distributed_values,)))
      return outputs

    results = run()

    expected = [i**2 for i in range(distribution.num_replicas_in_sync)]
    self.assertAllEqual(results, expected)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"]))
  def testMakeDistributedValueDefaultDevicePlacement(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    def value_fn(ctx):
      del ctx
      return constant_op.constant(1.0)
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    for i in range(len(distribution.extended.worker_devices)):
      self.assertAllEqual(distributed_values._values[i].device,
                          "/job:localhost/replica:0/task:0/device:CPU:0")

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ] + strategy_combinations.multiworker_strategies,
          mode=["eager"]))
  def testMakeDistributedValueExplicitDevicePlacement(self, distribution):
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")
    worker_devices = distribution.extended.worker_devices
    def value_fn(ctx):
      # In multi client setup, worker_devices is just the devices on that
      # worker.
      worker_device_id = ctx.replica_id_in_sync_group % len(worker_devices)
      with ops.device(worker_devices[worker_device_id]):
        return array_ops.identity(1.0)
    distributed_values = (
        distribution.experimental_distribute_values_from_function(value_fn))
    for i in range(len(distribution.extended.worker_devices)):
      self.assertAllEqual(distributed_values._values[i].device,
                          worker_devices[i])


class DistributedDelegateTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testGetAttr(self):
    class Foo(object):

      def __init__(self, x):
        self.x = x

    v = values_lib.DistributedDelegate((Foo(7), Foo(8)))
    self.assertEqual(7, v.x)
    with self.assertRaises(AttributeError):
      _ = v.y

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverride(self):
    v = values_lib.DistributedDelegate((7, 8))
    # v should act like int(7).
    self.assertEqual(8, v + 1)
    self.assertEqual(10, 3 + v)
    self.assertEqual(14, v + v)
    self.assertEqual(5, v - 2)
    self.assertEqual(6, 13 - v)
    self.assertEqual(0, v - v)
    self.assertEqual(14, v * 2)
    self.assertEqual(21, 3 * v)
    self.assertEqual(49, v * v)
    self.assertEqual(3.5, v / 2)
    self.assertEqual(1.5, 10.5 / v)
    self.assertEqual(3, v // 2)
    self.assertEqual(2, 15 // v)
    self.assertEqual(1, v % 2)
    self.assertEqual(2, 16 % v)
    # pylint: disable=g-generic-assert
    self.assertTrue(v < 12)
    self.assertTrue(v <= 12)
    self.assertFalse(v > 12)
    self.assertFalse(v >= 12)
    self.assertFalse(12 < v)
    self.assertFalse(12 <= v)
    self.assertTrue(12 > v)
    self.assertTrue(12 >= v)
    # pylint: enable=g-generic-assert
    self.assertEqual(3, v & 3)
    self.assertEqual(3, 11 & v)
    self.assertEqual(15, v | 8)
    self.assertEqual(23, 16 | v)
    self.assertEqual(4, v ^ 3)
    self.assertEqual(12, 11 ^ v)
    self.assertEqual(343, pow(v, 3))
    self.assertEqual(3, pow(v, 3, 10))
    self.assertEqual(128, pow(2, v))
    self.assertEqual(-7, -v)
    self.assertEqual(~7, ~v)
    self.assertEqual(7, abs(v))
    with self.assertRaises(TypeError):
      _ = v[2]

  @test_util.run_in_graph_and_eager_modes
  def testCopy(self):

    class Foo(object):

      def __init__(self, x):
        self.x = x

    v = values_lib.DistributedDelegate((Foo(7), Foo(8)))
    v_shallow_copy = copy.copy(v)
    self.assertEqual(v.x, v_shallow_copy.x)
    v_deep_copy = copy.deepcopy(v)
    self.assertEqual(v.x, v_deep_copy.x)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_one_cpu,
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.tpu_strategy,
            strategy_combinations.tpu_strategy_packed_var,
            strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu
        ],
        synchronization=[
            variables_lib.VariableSynchronization.ON_READ,
            variables_lib.VariableSynchronization.ON_WRITE,
        ],
        aggregation=[
            variables_lib.VariableAggregation.MEAN,
            variables_lib.VariableAggregation.SUM,
            variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
        ],
        mode=["graph", "eager"],
        use_var_policy=[True, False]))
class DistributedVariableTest(test.TestCase, parameterized.TestCase):

  def testExtendsVariable(self, distribution, synchronization, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          1., synchronization=synchronization, aggregation=aggregation)
    self.assertIsInstance(v, variables_lib.Variable)

  def testCheckpointing(self, distribution, synchronization, aggregation, mode):

    if (isinstance(distribution,
                   collective_all_reduce_strategy.CollectiveAllReduceStrategy)
        and mode == "graph"):
      self.skipTest("MWMS combinations tests do not work well in graph mode.")

    with distribution.scope():
      v = variables_lib.Variable(
          constant_op.constant([1., 2., 3., 4]),
          synchronization=synchronization,
          aggregation=aggregation)

    self.evaluate(v.initializer)
    before_save = self.evaluate(v.read_value())

    # Save random weights into checkpoint.
    checkpoint = trackable_utils.Checkpoint(v=v)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    with self.test_session():
      save_path = checkpoint.save(prefix)

    # Assign inverted value.
    self.evaluate(v.assign(constant_op.constant([4., 3., 2., 1.])))
    after_assign = self.evaluate(v.read_value())
    self.assertNotAllClose(before_save, after_assign)

    # Restore from the checkpoint.
    with self.test_session():
      checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    after_restore = self.evaluate(v)
    self.assertAllClose(before_save, after_restore)

  def testTraceback(self, distribution, synchronization, aggregation):
    if context.executing_eagerly():
      self.skipTest("does not apply to eager")
    with distribution.scope():
      variable_scope.get_variable(
          name="testVar",
          initializer=1.,
          use_resource=True,
          synchronization=synchronization,
          aggregation=aggregation)
      with self.assertRaisesRegex(ValueError,
                                  "Variable testVar already exists"):
        variable_scope.get_variable(
            name="testVar",
            initializer=1.,
            use_resource=True,
            synchronization=synchronization,
            aggregation=aggregation)

  def testSelectReplica(self, distribution, synchronization, aggregation):
    with distribution.scope():
      v = variables_lib.Variable(
          1., synchronization=synchronization, aggregation=aggregation)
    self.assertIs(v, distribute_utils.select_replica(0, v))

  def testIsTensorLike(self, distribution, synchronization, aggregation):
    if isinstance(distribution.extended,
                  tpu_strategy.TPUExtended) and context.executing_eagerly():
      self.skipTest("TPU doesn't support pure eager")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)
    # In cross replica context.
    self.assertIsInstance(v, core.Tensor)
    # In replica context.
    distribution.run(
        lambda v: self.assertIsInstance(v, core.Tensor), args=(v,))

  def testAssignReturnValueIsTensorLike(self, distribution, synchronization,
                                        aggregation):
    if isinstance(distribution.extended, tpu_strategy.TPUExtended):
      if context.executing_eagerly():
        self.skipTest("TPU doesn't support pure eager")
      else:
        self.skipTest("b/152076846")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)

    def assert_is_tensor_like(v):
      # We can't use Python literals because they are treated as non-distributed
      # values is not allowed when aggregation is SUM. See
      # `cross_device_ops.reduce_non_distributed_value`.
      delta = array_ops.identity(1.)
      self.assertIsInstance(v.assign(delta), core.Tensor)
      self.assertIsInstance(v.assign_sub(delta), core.Tensor)
      self.assertIsInstance(v.assign_add(delta), core.Tensor)

    # In cross replica context we return a PerReplica which is not Tensor like
    # all the time yet.
    if (synchronization == variables_lib.VariableSynchronization.ON_READ and
        aggregation != variables_lib.VariableAggregation.SUM):
      assert_is_tensor_like(v)

    # In replica context.
    distribution.run(assert_is_tensor_like, args=(v,))

  def testDeepCopy(self, distribution, synchronization,
                   aggregation):
    if not context.executing_eagerly():
      self.skipTest("deepcopy only supported in eager mode")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)
      in_dist_copy = copy.deepcopy(v)

    out_dist_copy = copy.deepcopy(v)

    def assert_is_deep_copy(v1, v2):
      self.assertIsInstance(v2, type(v1))
      self.assertEqual(v1.aggregation, v2.aggregation)
      self.assertEqual(v1.distribute_strategy, v2.distribute_strategy)
      if isinstance(v1, ps_values.AggregatingVariable):
        self.assertIsInstance(v2.get(), type(v1.get()))
        self.assertNotEqual(id(v1.get()), id(v2.get()))
      else:
        if v1._policy:
          self.assertNotEqual(id(v1._policy), id(v2._policy))  # pylint: disable=protected-access
        else:
          self.assertEqual(id(v1._policy), id(v2._policy))  # pylint: disable=protected-access
        self.assertEqual(len(v1.values), len(v2.values))
        for (v1v, v2v) in zip(v1.values, v2.values):
          self.assertEqual(v1v.device, v2v.device)
          self.assertNotEqual(id(v1v), id(v2v))
          self.assertAllEqual(self.evaluate(v1.values),
                              self.evaluate(v2.values))

    self.evaluate(variables_lib.global_variables_initializer())
    if not isinstance(distribution.extended, tpu_strategy.TPUExtended):
      distribution.run(assert_is_deep_copy, args=(v, in_dist_copy))
      distribution.run(assert_is_deep_copy, args=(v, out_dist_copy))

  def testAssignSignature(self, distribution, synchronization, aggregation):
    # This test verifies assign*() can be called in the same way as normal
    # variables.
    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)

      def assign():
        one = constant_op.constant(1.)
        v.assign(one, True, "assign", False)
        # TODO(b/154017756): SyncOnReadVariable.assign() doesn't support passing
        # value as a keyword argument.
        v.assign(one, use_locking=True, name="assign", read_value=False)
        v.assign_add(one, True, "assign", False)
        v.assign_add(one, use_locking=True, name="assign", read_value=False)
        v.assign_sub(one, True, "assign", False)
        v.assign_sub(one, use_locking=True, name="assign", read_value=False)
        # Return something for graph mode to fetch.
        return constant_op.constant(1)

      self.evaluate(variables_lib.global_variables_initializer())
      if not (synchronization == variables_lib.VariableSynchronization.ON_READ
              and aggregation == variables_lib.VariableAggregation.SUM):
        self.evaluate(distribution.experimental_local_results(assign()))
      if not (isinstance(distribution.extended, tpu_strategy.TPUExtended) and
              context.executing_eagerly()):
        self.evaluate(
            distribution.experimental_local_results(distribution.run(assign)))

  def testStrategyExtendedUpdate(self, distribution, synchronization,
                                 aggregation):
    if len(distribution.extended.parameter_devices) != 2:
      self.skipTest("n/a: needs exactly two parameter devices")
    if (synchronization == variables_lib.VariableSynchronization.ON_WRITE and
        aggregation != variables_lib.VariableAggregation.NONE):
      self.skipTest("n/a: doesn't apply to ON_WRITE variable with aggregation")
    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)
    value = values_lib.PerReplica([1., 2.])

    assign_fn = lambda var, value: var.assign(value)
    self.evaluate(distribution.extended.update(v, assign_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [1., 2.])

    assign_add_fn = lambda var, value: var.assign_add(value)
    self.evaluate(distribution.extended.update(v, assign_add_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [2., 4.])

    assign_sub_fn = lambda var, value: var.assign_sub(value)
    self.evaluate(distribution.extended.update(v, assign_sub_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [1., 2.])

    read_assign_fn = lambda var, value: var.assign_add(var.value() + var.
                                                       read_value())
    self.evaluate(
        distribution.extended.update(v, read_assign_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [3., 6.])

  def testSaveNonDistributed(self, distribution, synchronization, aggregation):
    # This test verifies that the DistributedVariable behave like the primary
    # variable when saving a non-distributed version of the model (the default).
    # The test asserts that the function traced under SaveContext has no device
    # annotations and only reference the primary component of the variable. Note
    # that please avoid capturing other eager tensors in this test to make the
    # assertion easy.

    if isinstance(distribution.extended,
                  parameter_server_strategy.ParameterServerStrategyExtended):
      self.skipTest("b/148689177: AggregatingVariable doesn't "
                    "conform to Variable interface well")

    # tf.function requires the return value to be Tensors, which is not always
    # case for properties and methods of Variable, so we simply discard the
    # return values.
    def _discard_return(f):
      f()
      return

    def _test(f, v):
      # This verifies that the function under SaveContext:
      #   - contains no device annotations.
      #   - only references the primary component of the variable.
      g = def_function.function(lambda: _discard_return(f))
      options = save_options.SaveOptions(
          experimental_variable_policy=save_options.VariablePolicy.NONE)
      with save_context.save_context(options):
        # The graph should contain no device.
        graph = g.get_concrete_function().graph
      for op in graph.get_operations():
        self.assertEqual(op.device, "", msg=str(op))
      # The function should only capture the primary variable. Note that it
      # may not have captures, e.g. v.aggregation.
      captures = list(graph.captures)
      self.assertLessEqual(len(captures), 1)
      if graph.captures:
        self.assertIs(captures[0][0], v._primary.handle)

    def _assert(cond):
      return control_flow_ops.Assert(cond, [cond])

    with distribution.scope():
      # We use four variables for convenience reasons. They have no special
      # meaning.
      # - v is used whenever possible.
      # - w is used for scatter and gather, which require the variable to be
      # non-scalar.
      # - y is used when the dtype needs to be integer. Note that aggregation
      # cannot be MEAN for integers.
      v = variables_lib.Variable(
          0.,
          synchronization=synchronization,
          aggregation=aggregation,
          trainable=True)
      w = variables_lib.Variable([0., 0., 0.],
                                 synchronization=synchronization,
                                 aggregation=aggregation,
                                 trainable=True)
      if aggregation != variables_lib.VariableAggregation.MEAN:
        y = variables_lib.Variable(
            0,
            synchronization=synchronization,
            aggregation=aggregation)

    # pylint: disable=g-long-lambda

    # tf.Variable properties.
    _test(lambda: self.assertEqual(v.aggregation, aggregation), v)
    _test(lambda: self.assertIs(v.constraint, None), v)
    # TODO(crccw): should we raise an error instead?
    _test(lambda: self.assertEqual(v.device, v._primary.device), v)
    _test(lambda: self.assertEqual(v.dtype, dtypes.float32), v)
    if not context.executing_eagerly():
      _test(lambda: self.assertIs(v.graph, v._primary.graph), v)
    if not context.executing_eagerly():
      _test(lambda: _assert(v.initial_value == 0), v)
    _test(lambda: self.assertIs(v.initializer, v._primary.initializer), v)
    _test(lambda: self.assertEqual(v.name, "Variable:0"), v)
    if not context.executing_eagerly():
      _test(lambda: self.assertIs(v.op, v._primary.op), v)
    _test(lambda: self.assertEqual(v.shape, tensor_shape.TensorShape(())), v)
    _test(lambda: self.assertEqual(v.synchronization, synchronization), v)
    _test(lambda: self.assertTrue(v.trainable, True), v)

    # tf.Variable methods.
    _test(lambda: check_ops.assert_equal_v2(v.assign(1.), 1.), v)
    _test(lambda: check_ops.assert_equal_v2(v.assign_add(1.), 2.), v)
    _test(lambda: check_ops.assert_equal_v2(v.assign_sub(1.), 1.), v)
    # TODO(b/148689177): Implement batch_scatter_update.
    # count_up_to() is skipped since it's deprecated.
    # eval() is skipped since it shouldn't called in a tf.function.
    # experimental_ref() is skipped since it's deprecated.
    # from_proto() is skipped since it shouldn't called in a tf.function.
    # TODO(b/148689177): Implement gather_nd.
    _test(
        lambda: check_ops.assert_equal_v2(v.get_shape(),
                                          tensor_shape.TensorShape(())), v)
    # initialized_value() is skipped since it shouldn't called in a tf.function.
    # load() is skipped since it shouldn't called in a tf.function.
    _test(lambda: check_ops.assert_equal_v2(v.read_value(), 1.), v)
    # ref() is skipped since it shouldn't called in a tf.function.
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_add(_make_index_slices(values=[1., 2.], indices=[0, 2])),
            [1., 0., 2.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_div(_make_index_slices(values=[4., 2.], indices=[0, 2])),
            [0.25, 0., 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_max(_make_index_slices(values=[1., 0.5], indices=[1, 2])),
            [0.25, 1., 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_min(_make_index_slices(values=[1., 0.5], indices=[0, 1])),
            [0.25, 0.5, 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_mul(_make_index_slices(values=[2., 0.5], indices=[0, 1])),
            [0.5, 0.25, 1.]), w)
    # TODO(b/148689177): Implement scatter_nd_*
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_sub(_make_index_slices(values=[2., 0.5], indices=[0, 1])),
            [-1.5, -0.25, 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_update(
                _make_index_slices(values=[2., 0.5], indices=[0, 1])),
            [2., 0.5, 1.]), w)
    # set_shape() is skipped since ResourceVariable doesn't implement it.
    # to_proto() is skipped since it shouldn't called in a tf.function.
    _test(lambda: check_ops.assert_equal_v2(v.value(), 1.), v)

    # DistributedVariable should be treated as ResourceVariable, so it needs to
    # conform to ResourceVariable interface as well.
    _test(lambda: self.assertIs(v.handle, v._primary.handle), v)

    # Convert to tensor.
    _test(lambda: check_ops.assert_equal_v2(ops.convert_to_tensor(v), 1.), v)

    # Control dependency.
    def _with_control_dep():
      with ops.control_dependencies([v.assign(1.)]):
        return array_ops.identity(1)

    _test(_with_control_dep, v)

    # Operator overloads.
    _test(lambda: check_ops.assert_equal_v2(v.assign(7.), 7.), v)
    _test(lambda: check_ops.assert_equal_v2(v + 1., 8.), v)
    _test(lambda: check_ops.assert_equal_v2(3 + v, 10.), v)
    _test(lambda: check_ops.assert_equal_v2(v + v, 14.), v)
    _test(lambda: check_ops.assert_equal_v2(v - 2., 5.), v)
    _test(lambda: check_ops.assert_equal_v2(v - v, 0.), v)
    _test(lambda: check_ops.assert_equal_v2(v * 2., 14.), v)
    _test(lambda: check_ops.assert_equal_v2(3 * v, 21.), v)
    _test(lambda: check_ops.assert_equal_v2(v * v, 49.), v)
    _test(
        lambda: check_ops.assert_equal_v2(
            math_ops.cast(v / 2., dtypes.float32), 3.5), v)
    _test(
        lambda: check_ops.assert_equal_v2(
            math_ops.cast(14. / v, dtypes.float32), 2.), v)
    _test(lambda: _assert(v < 12.), v)
    _test(lambda: _assert(v <= 12.), v)
    _test(lambda: _assert(not v > 12.), v)
    _test(lambda: _assert(not v >= 12.), v)
    _test(lambda: _assert(not 12. < v), v)
    _test(lambda: _assert(not 12. <= v), v)
    _test(lambda: _assert(12. > v), v)
    _test(lambda: _assert(12. >= v), v)
    _test(lambda: check_ops.assert_near_v2(pow(v, 3.), 343.), v)
    _test(lambda: check_ops.assert_near_v2(pow(2., v), 128.), v)
    _test(lambda: check_ops.assert_equal_v2(abs(v), 7.), v)

    # Operator overloads that only works for integers.
    if aggregation != variables_lib.VariableAggregation.MEAN:
      _test(lambda: check_ops.assert_equal_v2(y.assign(7), 7), y)
      _test(lambda: check_ops.assert_equal_v2(y // 2, 3), y)
      _test(lambda: check_ops.assert_equal_v2(15 // y, 2), y)
      _test(lambda: check_ops.assert_equal_v2(y % 2, 1), y)
      _test(lambda: check_ops.assert_equal_v2(16 % y, 2), y)
      _test(lambda: check_ops.assert_equal_v2(y & 3, 3), y)
      _test(lambda: check_ops.assert_equal_v2(3 & y, 3), y)
      _test(lambda: check_ops.assert_equal_v2(y | 8, 15), y)
      _test(lambda: check_ops.assert_equal_v2(16 | y, 23), y)
      _test(lambda: check_ops.assert_equal_v2(y ^ 3, 4), y)
      _test(lambda: check_ops.assert_equal_v2(11 ^ y, 12), y)
      _test(lambda: check_ops.assert_equal_v2(-y, -7), y)
      _test(lambda: check_ops.assert_equal_v2(~y, ~7), y)

    # Index.
    if isinstance(distribution.extended, tpu_strategy.TPUExtended):
      # TODO(b/161572567): slice assignment doesn't work for TPU.
      _test(lambda: check_ops.assert_equal_v2(w[0], 2.), w)
    else:
      _test(lambda: check_ops.assert_equal_v2(w[0].assign(1.), [1., 0.5, 1.]),
            w)
      _test(lambda: check_ops.assert_equal_v2(w[0], 1.), w)

    # pylint: enable=g-long-lambda

  def testUnsaveable(self, distribution, synchronization, aggregation, mode):
    if isinstance(distribution.extended,
                  parameter_server_strategy.ParameterServerStrategyExtended):
      self.skipTest("n/a: not appliable to AggregatingVariable")
    if (isinstance(distribution,
                   collective_all_reduce_strategy.CollectiveAllReduceStrategy)
        and mode == "graph"):
      self.skipTest("MWMS combinations tests do not work well in graph mode.")
    with distribution.scope():
      v = variables_lib.Variable([1., 1.],
                                 synchronization=synchronization,
                                 aggregation=aggregation)

    with self.cached_session():
      self.evaluate(variables_lib.global_variables_initializer())

    export_dir = self.get_temp_dir()

    def _assert_unsaveable(f):
      # Ignore if it cannot be traced. Certain combinations are not supported or
      # yet or not allowed.
      try:
        f = def_function.function(f).get_concrete_function()
      except (NotImplementedError, ValueError):
        return
      with self.assertRaisesRegex(ValueError, "f_with_input_signature"):
        save.save(v, export_dir, signatures=f)

    _assert_unsaveable(lambda: v.assign(ops.convert_to_tensor([1., 1.])))
    _assert_unsaveable(lambda: v.assign_add(ops.convert_to_tensor([1., 1.])))
    _assert_unsaveable(lambda: v.assign_sub(ops.convert_to_tensor([1., 1.])))
    _assert_unsaveable(lambda: v.scatter_add(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_sub(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_mul(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_div(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_min(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_max(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_update(_make_index_slices([1.], [0])))
    # Reading a ON_READ variable should be unsaveable if either:
    # 1) CollectiveAllReduceStrategy, and aggregation is MEAN/SUM.
    # 2) aggregation is SUM.
    if (synchronization == variables_lib.VariableSynchronization.ON_READ and
        (aggregation == variables_lib.VariableAggregation.SUM or
         (isinstance(distribution.extended,
                     collective_all_reduce_strategy.CollectiveAllReduceExtended)
          and aggregation == variables_lib.VariableAggregation.MEAN))):
      _assert_unsaveable(v.read_value)
      _assert_unsaveable(v.value)
      _assert_unsaveable(lambda: ops.convert_to_tensor(v))
    else:
      # Otherwise reading a variable should be saveable.

      @def_function.function
      def f():
        v.read_value()
        v.value()
        return ops.convert_to_tensor(v)

      with self.cached_session():
        save.save(v, export_dir, signatures=f.get_concrete_function())


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_one_cpu,
            strategy_combinations.tpu_strategy,
        ],
        mode=["eager"]))
class PackedDistributedVariableTest(test.TestCase, parameterized.TestCase):

  def testPackedVariable(self, distribution):
    with distribution.scope():
      v0 = variables_lib.Variable(0.)
    self.assertIsNone(v0._packed_var)

    distribution._enable_packed_variable_in_eager_mode = True
    with distribution.scope():
      v1 = variables_lib.Variable(0)
      self.assertIsInstance(v1._packed_var, packed.PackedDistributedVariable)

    devices = v1._devices
    for i in range(1, len(devices)):
      with distribute_lib.ReplicaContext(distribution, i):
        v1.assign(i)
    val = v1._get()
    self.assertIsInstance(val, packed.PackedVarAndDevice)
    self.assertEqual(val.device, devices[0])
    self.assertEqual(self.evaluate(val.read_value()), 0)
    for i in range(0, len(devices)):
      with distribute_lib.ReplicaContext(distribution, i):
        val = v1._get()
        self.assertIsInstance(val, packed.PackedVarAndDevice)
        self.assertEqual(val.device, devices[i])
        self.assertEqual(self.evaluate(val.read_value()), i)

  def testIgnorePackedVariableInSaveContext(self, distribution):
    distribution._enable_packed_variable_in_eager_mode = True
    with distribution.scope():
      v = variables_lib.Variable(0)
      self.assertIsInstance(
          v._packed_variable, packed.PackedDistributedVariable)

    options = save_options.SaveOptions()
    with save_context.save_context(options):
      self.assertIsNone(v._packed_variable)


class MirroredVariableTest(test.TestCase, parameterized.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    mirrored = _make_mirrored()
    v = mirrored.values[0]
    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    mirrored = values_lib.MirroredVariable(
        None, (v,), variable_scope.VariableAggregation.MEAN)

    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)


class MirroredVariableSaveRestoreTest(test.TestCase, parameterized.TestCase):

  def _assign_mirrored(self, v, new):
    for var, n in zip(v.values, new):
      self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  def _save_mirrored(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored(distribution)

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path = self._save(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.))

      # Saves the current value of var, 3.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3. to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3., self.evaluate(var))

  def _restore_mirrored(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored(distribution)
      v = mirrored.values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [7., 8.])

      # Restores the saved value of 3. to both variables.
      saver = saver_lib.Saver(var_list=[mirrored])
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreMirroredOneGraph(self, distribution):
    with self.cached_session() as sess:
      mirrored = _make_mirrored(distribution)
      v = mirrored  .values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path, saver = self._save_return_saver(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])

      # Restores the saved value of 3. to both variables.
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveMirroredRestoreMirrored(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored(distribution)
    self._restore_mirrored(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveMirroredRestoreNormal(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreMirrored(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_mirrored(save_path, distribution)


_TPU_STRATEGIES = (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1)


def _make_replica_local(method, strategy=None):
  if strategy is None:
    devices = ("/device:GPU:0", "/device:CPU:0")
  else:
    devices = strategy.extended.worker_devices

  v = []
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(variable_scope.get_variable(
          name=n, initializer=init, use_resource=True))

  if (strategy is not None) and isinstance(strategy, _TPU_STRATEGIES):
    var_cls = tpu_values.TPUSyncOnReadVariable
  else:
    var_cls = values_lib.SyncOnReadVariable
  replica_local = var_cls(strategy, v, method)
  return v, replica_local


class SyncOnReadVariableTest(test.TestCase, parameterized.TestCase):

  def _assign_replica_local(self, v, new):
    for var, n in zip(v, new):
      with ops.device(var.device):
        self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")
    v, replica_local = _make_replica_local(
        variable_scope.VariableAggregation.SUM)

    self.assertEqual(v[0].constraint, replica_local.constraint)
    self.assertEqual(v[0].name, replica_local.name)
    self.assertEqual(v[0].dtype, replica_local.dtype)
    self.assertEqual(v[0].shape, replica_local.shape)
    self.assertEqual(variable_scope.VariableAggregation.SUM,
                     replica_local.aggregation)

  @test_util.run_v2_only
  def testCanPassToDefFun(self):
    @def_function.function
    def add1(x):
      return x + 1

    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    replica_local = values_lib.SyncOnReadVariable(
        None, (v,), variable_scope.VariableAggregation.MEAN)
    self.assertEqual(2., self.evaluate(add1(replica_local)))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testTensorConversion(self, distribution):
    with context.graph_mode():
      _, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)
      converted = ops.convert_to_tensor(replica_local, as_ref=False)
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

      converted = ops.convert_to_tensor(replica_local, as_ref=True)
      # Resources variable are converted to tensors as well when as_ref is True.
      self.assertIsInstance(converted, ops.Tensor)
      self.assertEqual(converted.dtype, replica_local.dtype)

  @combinations.generate(combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ], mode=["eager"]))
  def testValueInCrossReplicaContext(self, distribution):
    value_list, replica_local = _make_replica_local(
        variable_scope.VariableAggregation.ONLY_FIRST_REPLICA, distribution)

    self.assertIsInstance(replica_local.value(), ops.Tensor)
    self.assertEqual(self.evaluate(replica_local.value()),
                     self.evaluate(value_list[0].value()))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreReplicaLocalSumOneGraph(self, distribution):
    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 7.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 7. which gets divided equally
        # between the variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreReplicaLocalMeanOneGraph(self, distribution):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    with self.cached_session() as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5.
        save_path, saver = self._save_return_saver(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])

        # Restores the saved value of 3.5 to both variables.
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _save_replica_local_mean(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [3., 4.])

      with distribution.scope():
        # Saves the current value of (v[0] + v[1])/2, 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
    return save_path

  def _save_replica_local_sum(self, distribution):
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [1.5, 2.])

      with distribution.scope():
        # Saves the current value of v[0] + v[1], 3.5
        save_path = self._save(sess, replica_local)

        # Change the values between save and restore.
        self._assign_replica_local(v, [5., 6.])
    return save_path

  def _save_normal(self):
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.5))

      # Saves the current value of var, 3.5.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3.5 to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3.5, self.evaluate(var))

  def _restore_replica_local_mean(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.MEAN, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([3.5, 3.5], self.evaluate([v[0], v[1]]))

  def _restore_replica_local_sum(self, save_path, distribution):
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      v, replica_local = _make_replica_local(
          variable_scope.VariableAggregation.SUM, distribution)

      # Overwrite the initial values.
      self._assign_replica_local(v, [7., 8.])

      with distribution.scope():
        # Restores the saved value of 3.5 to both variables.
        saver = saver_lib.Saver(var_list=[replica_local])
        saver.restore(sess, save_path)
        self.assertEqual([1.75, 1.75], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalRestoreReplicaLocalMean(self, distribution):
    save_path = self._save_replica_local_mean(distribution)
    self._restore_replica_local_mean(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalRestoreReplicaLocalSum(self, distribution):
    save_path = self._save_replica_local_sum(distribution)
    self._restore_replica_local_sum(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalMeanRestoreNormal(self, distribution):
    save_path = self._save_replica_local_mean(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveReplicaLocalSumRestoreNormal(self, distribution):
    save_path = self._save_replica_local_sum(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreReplicaLocalMean(self, distribution):
    save_path = self._save_normal()
    self._restore_replica_local_mean(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreReplicaLocalSum(self, distribution):
    save_path = self._save_normal()
    self._restore_replica_local_sum(save_path, distribution)


class MirroredTest(test.TestCase):

  def testAddOp(self):
    if context.num_gpus() < 1:
      self.skipTest("A GPU is not available for this test.")
    mirrored_val = _make_mirrored_val(init_val=3.)

    self.assertEqual(self.evaluate(constant_op.constant(6.)),
                     self.evaluate(mirrored_val + mirrored_val))
    self.assertEqual(self.evaluate(constant_op.constant(4.)),
                     self.evaluate(mirrored_val + 1))
    self.assertEqual(self.evaluate(mirrored_val + 1),
                     self.evaluate(math_ops.add(mirrored_val, 1)))
    self.assertEqual(type(mirrored_val + 1),
                     type(math_ops.add(mirrored_val, 1)))


class PerReplicaTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testTypeSpec(self):
    vals = (constant_op.constant(1.),)
    per_replica = values_lib.PerReplica(vals)

    spec = per_replica._type_spec
    self.assertEqual(spec._value_specs,
                     (tensor_spec.TensorSpec([], dtypes.float32),))

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testTypeSpecRoundTrip(self):
    vals = (constant_op.constant(1.),)
    per_replica = values_lib.PerReplica(vals)

    spec = per_replica._type_spec
    tensor_list = spec._to_components(per_replica)
    reconstructed = spec._from_components(tensor_list)

    self.assertAllEqual(per_replica.values, reconstructed.values)

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testTypeSpecNest(self):
    vals = (constant_op.constant(1.), constant_op.constant([5., 6.0]),)
    per_replica = values_lib.PerReplica(vals)

    # Note: nest.map_structure exercises nest.flatten and
    # nest.pack_sequence_as.
    result = nest.map_structure(
        lambda t: t + 10, per_replica, expand_composites=True)

    self.assertLen(result.values, 2)
    self.assertAllEqual(result.values[0], 11.)
    self.assertAllEqual(result.values[1], [15., 16.0])

  @test_util.run_in_graph_and_eager_modes
  def testIsGraphTensor(self):
    per_replica = values_lib.PerReplica((constant_op.constant(1.),))
    for t in nest.flatten(per_replica, expand_composites=True):
      self.assertEqual(hasattr(t, "graph"), not context.executing_eagerly())

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testDoesNotTriggerFunctionTracing(self):
    traces = []

    @def_function.function
    def f(x):
      traces.append(None)  # Only happens on trace.
      return x

    per_replica = values_lib.PerReplica((constant_op.constant(1.),))

    # Trace once.
    f(per_replica)
    self.assertNotEmpty(traces)
    del traces[:]

    per_replica_spec = per_replica._type_spec
    for _ in range(5):
      vals = per_replica_spec._to_components(per_replica)
      vals = [v * 2 for v in vals]
      per_replica = per_replica_spec._from_components(vals)

      output = f(per_replica)
      self.assertIsInstance(output, values_lib.PerReplica)
      self.assertAllEqual(output._values, per_replica._values)
      self.assertEmpty(traces)  # Make sure we're not re-tracing `f`.

  @combinations.generate(combinations.combine(mode=["eager"]))
  def testFunctionCanReturnPerReplica(self):
    f = def_function.function(lambda x: x)
    x = values_lib.PerReplica((constant_op.constant(1.),))
    y = f(x)
    self.assertIsNot(x, y)
    nest.map_structure(self.assertAllEqual, x, y, expand_composites=True)
    self.assertEqual(x._type_spec, y._type_spec)

  @test_util.run_in_graph_and_eager_modes
  def testCondWithTensorValues(self):
    per_replica_1 = values_lib.PerReplica((constant_op.constant("a"),))
    per_replica_2 = values_lib.PerReplica((constant_op.constant(["b", "c"]),))
    condition = array_ops.placeholder_with_default(True, [])

    result = control_flow_ops.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.run_in_graph_and_eager_modes
  def testCondWithValuesConvertibleToTensor(self):
    per_replica_1 = values_lib.PerReplica(("a",))
    per_replica_2 = values_lib.PerReplica(("b",))
    condition = array_ops.placeholder_with_default(True, [])

    result = control_flow_ops.cond(
        condition, lambda: per_replica_1, lambda: per_replica_2)

    self.assertLen(result.values, 1)
    self.assertAllEqual(result.values[0], "a")

  @test_util.build_as_function_and_v1_graph
  def testCondWithValuesNotConvertibleToTensor(self):
    per_replica_1 = values_lib.PerReplica(({"a"},))
    per_replica_2 = values_lib.PerReplica(({"b", "c"},))
    condition = array_ops.placeholder(dtypes.bool, [])

    with self.assertRaisesRegex(TypeError, "Could not build a TypeSpec for"):
      control_flow_ops.cond(
          condition, lambda: per_replica_1, lambda: per_replica_2)


def _make_index_slices(values, indices, dense_shape=None):
  if dense_shape:
    dense_shape = array_ops.identity(dense_shape)
  return indexed_slices.IndexedSlices(
      array_ops.identity(values), array_ops.identity(indices), dense_shape)


if __name__ == "__main__":
  ds_test_util.main()
