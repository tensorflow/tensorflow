# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the distributed variables library."""

import copy
import os

from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
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
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.types import core


def _device_str(d):
  return "/device:GPU:" + str(d)


def _nested_value(d):
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


def mirrored_and_tpu_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"])


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_one_cpu,
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
            strategy_combinations.tpu_strategy,
            strategy_combinations.tpu_strategy_packed_var,
            strategy_combinations.tpu_strategy_spmd,
            strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
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
    distribution.run(lambda v: self.assertIsInstance(v, core.Tensor), args=(v,))

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

  def testDeepCopy(self, distribution, synchronization, aggregation):
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
          self.assertAllEqual(
              self.evaluate(v1.values), self.evaluate(v2.values))

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
            0, synchronization=synchronization, aggregation=aggregation)

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
    _test(lambda: self.assertEqual(v.trainable, True), v)

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
    if not distribution.extended._use_merge_call():
      self.skipTest("Unsupported combination.")
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
         (not distribution.extended._use_merge_call()) or
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
      self.assertIsInstance(v._packed_variable,
                            packed.PackedDistributedVariable)

    options = save_options.SaveOptions()
    with save_context.save_context(options):
      self.assertIsNone(v._packed_variable)


def _make_index_slices(values, indices, dense_shape=None):
  if dense_shape:
    dense_shape = array_ops.identity(dense_shape)
  return indexed_slices.IndexedSlices(
      array_ops.identity(values), array_ops.identity(indices), dense_shape)


if __name__ == "__main__":
  ds_test_util.main()
