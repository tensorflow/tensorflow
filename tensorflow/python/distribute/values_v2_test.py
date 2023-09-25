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
"""Tests for the distributed values library."""

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import values_v2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib


class _VariableInterfaceTestBase(test.TestCase, parameterized.TestCase):
  # This test verifies that DistributedVariable/AutoSyncVariable conforms to
  # Variable and ResourceVariable interface, i.e. the methods and properties are
  # all defined. It verifies methods and properties that have the same code path
  # under different replicas/devices as well. It is not intended to verify
  # methods and properties that behave differently under different
  # replicas/devices; those should be covered separate tests.

  def create_variable(self, initial_value=1., **kwargs):
    raise NotImplementedError

  @property
  def devices(self):
    return ["CPU:0", "CPU:1"]

  # ==== Begin Variable interface ===
  # Please follow the same order as methods and properties defined in
  # tf.Variable.

  def testStringify(self):
    v = self.create_variable()
    self.assertIsInstance(v.__str__(), str)
    self.assertIsInstance(v.__repr__(), str)

  def testDenseRead(self):
    v = self.create_variable(1.)
    self.assertEqual(v.value(), 1.)
    self.assertEqual(v.read_value(), 1.)

  def testShape(self):
    v = self.create_variable([1.])
    self.assertEqual(v.shape, (1,))
    self.assertEqual(v.get_shape(), (1,))
    v.set_shape((1,))
    with self.assertRaisesRegex(ValueError, "not compatible"):
      v.set_shape((1, 1))

  @combinations.generate(combinations.combine(trainable=[True, False]))
  def testTrainable(self, trainable):
    v = self.create_variable(trainable=trainable)
    self.assertEqual(v.trainable, trainable)

  @combinations.generate(
      combinations.combine(synchronization=[
          variables_lib.VariableSynchronization.ON_READ,
          variables_lib.VariableSynchronization.ON_WRITE,
          variables_lib.VariableSynchronization.AUTO,
          variables_lib.VariableSynchronization.NONE,
      ]))
  def testSynchronization(self, synchronization):
    v = self.create_variable(synchronization=synchronization)
    self.assertEqual(v.synchronization, synchronization)

  @combinations.generate(
      combinations.combine(aggregation=[
          variables_lib.VariableAggregation.MEAN,
          variables_lib.VariableAggregation.SUM,
          variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
          variables_lib.VariableAggregation.NONE,
      ]))
  def testAggregation(self, aggregation):
    v = self.create_variable(aggregation=aggregation)
    self.assertEqual(v.aggregation, aggregation)

  @combinations.generate(combinations.combine(mode="graph"))
  def testEval(self):
    v = self.create_variable(1.)
    with self.cached_session():
      self.evaluate(variables_lib.global_variables_initializer())
      self.assertEqual(v.eval(), 1.)

  def testInitialValueEager(self):
    v = self.create_variable(1.)
    with self.assertRaises(RuntimeError):
      v.initial_value  # pylint: disable=pointless-statement

  @combinations.generate(combinations.combine(mode="graph"))
  def testInitialValueGraph(self):
    v = self.create_variable(1.)
    self.assertEqual(self.evaluate(v.initial_value), 1.)

  def testConstraint(self):
    v = self.create_variable(constraint=lambda x: x + 1.)
    self.assertEqual(v.constraint(1.), 2.)

  def testDenseUpdate(self):
    v = self.create_variable(1.)
    self.assertEqual(
        v.assign(2., use_locking=True, name="assign", read_value=True), 2.)
    self.assertIsNone(v.assign(3., read_value=False))
    self.assertEqual(v, 3.)
    self.assertEqual(
        v.assign_add(1., use_locking=True, name="assign_add", read_value=True),
        4.)
    self.assertIsNone(v.assign_add(1., read_value=False))
    self.assertEqual(v, 5.)
    self.assertEqual(
        v.assign_sub(1., use_locking=True, name="assign_sub", read_value=True),
        4.)
    self.assertIsNone(v.assign_sub(1., read_value=False))
    self.assertEqual(v, 3.)

    @def_function.function
    def f():
      self.assertIsInstance(v.assign(1., read_value=False), ops.Operation)
      self.assertIsInstance(v.assign_add(1., read_value=False), ops.Operation)
      self.assertIsInstance(v.assign_sub(1., read_value=False), ops.Operation)

    f()

  def testSparseUpdate(self):
    v = self.create_variable([0., 0., 0.])
    self.assertAllEqual(
        v.scatter_add(
            _make_index_slices(values=[1., 2.], indices=[0, 2]),
            use_locking=True,
            name="add"), [1., 0., 2.])
    self.assertAllEqual(
        v.scatter_div(
            _make_index_slices(values=[4., 2.], indices=[0, 2]),
            use_locking=True,
            name="div"), [0.25, 0., 1.])
    self.assertAllEqual(
        v.scatter_max(
            _make_index_slices(values=[1., 0.5], indices=[1, 2]),
            use_locking=True,
            name="max"), [0.25, 1., 1.])
    self.assertAllEqual(
        v.scatter_min(
            _make_index_slices(values=[1., 0.5], indices=[0, 1]),
            use_locking=True,
            name="min"), [0.25, 0.5, 1.])
    self.assertAllEqual(
        v.scatter_mul(
            _make_index_slices(values=[2., 0.5], indices=[0, 1]),
            use_locking=True,
            name="mul"), [0.5, 0.25, 1.])
    self.assertAllEqual(
        v.scatter_sub(
            _make_index_slices(values=[2., 0.5], indices=[0, 1]),
            use_locking=True,
            name="sub"), [-1.5, -0.25, 1.])
    self.assertAllEqual(
        v.scatter_update(
            _make_index_slices(values=[2., 0.5], indices=[0, 1]),
            use_locking=True,
            name="update"), [2., 0.5, 1.])
    self.assertAllEqual(
        v.batch_scatter_update(
            _make_index_slices(values=[1., 1.5], indices=[0, 1]),
            use_locking=True,
            name="update"), [1., 1.5, 1.])

  def testSparseNdUpdate(self):
    v = self.create_variable([0., 0., 0., 0.])
    self.assertAllEqual(
        v.scatter_nd_sub([[3], [1]], [1., 2.], name="sub"), [0., -2., 0., -1.])
    self.assertAllEqual(
        v.scatter_nd_add([[2], [0]], [1., 2.], name="add"), [2., -2., 1., -1.])
    self.assertAllEqual(
        v.scatter_nd_update([[1], [3]], [3., 3.], name="update"),
        [2., 3., 1., 3.])

  def testSparseRead(self):
    v = self.create_variable([[1., 2.], [3., 4.]])
    self.assertAllEqual(
        v.sparse_read([1, 0], name="read"), [[3., 4.], [1., 2.]])
    self.assertAllEqual(
        v.gather_nd([[1, 0], [0, 1]], name="gather_nd"), [3., 2.])

  def testTensorConversion(self):
    v = self.create_variable([1.])
    self.assertEqual(ops.convert_to_tensor(v), [1.])

  def testHash(self):
    v = self.create_variable()
    w = self.create_variable()
    d = {}
    with self.assertRaises(TypeError):
      d[v] = 1
    d[v.ref()] = 1
    self.assertEqual(d[v.ref()], 1)
    self.assertNotIn(w.ref(), d)

  @combinations.generate(combinations.combine(mode="graph"))
  def testHashGraph(self):
    v = self.create_variable()
    w = self.create_variable()
    d = {v: 1}
    self.assertEqual(d[v], 1)
    self.assertNotIn(w, d)

  def testEquality(self):
    v = self.create_variable(1.)
    w = self.create_variable(2.)
    x = self.create_variable(1.)
    self.assertEqual(v, x)
    self.assertNotEqual(v, w)

  @combinations.generate(combinations.combine(mode="graph"))
  def testEqualityGraph(self):
    # In legacy graph mode, tensor equality is object equality
    v = self.create_variable(1.)
    w = self.create_variable(1.)
    self.assertNotEqual(v, w)
    self.assertEqual(v, v)

  def testIteration(self):
    v = self.create_variable([1.])
    self.assertEqual([1.], list(iter(v)))

  def testProperties(self):
    v = self.create_variable()
    self.assertIsInstance(v.name, str)
    # _shared_name is also part of the interface. E.g. it's used in optimizer to
    # determine slot variable key.
    self.assertIsInstance(v._shared_name, str)
    self.assertIsNone(v.initializer)
    self.assertIsInstance(v.device, str)
    self.assertEqual(v.dtype, dtypes.float32)
    with self.assertRaises(AttributeError):
      v.op  # pylint: disable=pointless-statement
    with self.assertRaises(AttributeError):
      v.graph  # pylint: disable=pointless-statement

  @combinations.generate(combinations.combine(mode="graph"))
  def testPropertiesGraph(self):
    v = self.create_variable()
    self.assertIsInstance(v.initializer, ops.Operation)
    self.assertIsInstance(v.op, ops.Operation)
    self.assertIsInstance(v.graph, ops.Graph)

  def testProtoConversion(self):
    # to_proto and from_proto are not supported.
    v = self.create_variable([1, 2])
    with self.assertRaises(TypeError):
      v.to_proto()
    with self.assertRaises(TypeError):
      v.from_proto(variable_def=None)

  def testSaveSliceInfo(self):
    v = self.create_variable()
    slice_info = variables_lib.Variable.SaveSliceInfo()
    v._set_save_slice_info(slice_info)
    self.assertIs(v._get_save_slice_info(), slice_info)
    # Some code accesses _save_slice_info directly without using the getter.
    self.assertIs(v._save_slice_info, slice_info)

  def testOperatorOverride(self):
    v = self.create_variable(7)
    self.assertEqual(v + 1, 8)
    self.assertEqual(3 + v, 10)
    self.assertEqual(v + v, 14)
    self.assertEqual(v - 2, 5)
    self.assertEqual(13 - v, 6)
    self.assertEqual(v - v, 0)
    self.assertEqual(v * 2, 14)
    self.assertEqual(3 * v, 21)
    self.assertEqual(v * v, 49)
    self.assertEqual(v / 2, 3.5)
    self.assertEqual(14 / v, 2.)
    self.assertEqual(v // 2, 3)
    self.assertEqual(15 // v, 2)
    self.assertEqual(v % 2, 1)
    self.assertEqual(16 % v, 2)
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
    self.assertEqual(v & 3, 3)
    self.assertEqual(11 & v, 3)
    self.assertEqual(v | 8, 15)
    self.assertEqual(16 | v, 23)
    self.assertEqual(v ^ 3, 4)
    self.assertEqual(11 ^ v, 12)
    self.assertEqual(pow(v, 3), 343)
    # TODO(b/178748613): pow(v, 3, 10) fails.
    self.assertEqual(pow(2, v), 128)
    self.assertEqual(-v, -7)
    self.assertEqual(~v, ~7)
    self.assertEqual(abs(v), 7)

  def testSlice(self):
    v = self.create_variable([1., 2., 3.])
    self.assertEqual(v[1], 2.)
    v[2].assign(4.)
    self.assertAllEqual(v, [1., 2., 4.])

  # ==== End Variable interface ===

  # ==== Begin ResourceVariable interface ===
  def testHandle(self):
    v = self.create_variable()
    self.assertIsInstance(v.handle, tensor.Tensor)
    self.assertEqual(v.handle.dtype, dtypes.resource)

  def testInGraphMode(self):
    # This is protected but used in a lot of places internally.
    v = self.create_variable()
    self.assertFalse(v._in_graph_mode)

  def testUniqueId(self):
    # This is used in optimizer as part of slot variable key.
    v = self.create_variable()
    w = self.create_variable()
    self.assertNotEqual(v._unique_id, w._unique_id)

  def testIsResourceVariable(self):
    v = self.create_variable()
    self.assertTrue(resource_variable_ops.is_resource_variable(v))
  # ==== End ResourceVariable interface ===

  @combinations.generate(combinations.combine(mode="graph"))
  def testAsGraphElement(self):
    g = ops.Graph()
    with g.as_default():
      v = self.create_variable(1.)
      g.finalize()
      self.evaluate(v.initializer)
      # _as_graph_element shouldn't create new operations.
      self.assertEqual(self.evaluate(v._as_graph_element()), 1.)


class DistributedVariableInterfaceTest(_VariableInterfaceTestBase):

  def create_variable(self, initial_value=1., **kwargs):
    variables = []
    for device in self.devices:
      with ops.device(device):
        variables.append(
            variables_lib.Variable(initial_value, **kwargs))
    return values_v2.DistributedVariable(variables)


# Prevent the base class from running.
del _VariableInterfaceTestBase


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.tpu_strategy,
            strategy_combinations.mirrored_strategy_with_two_cpus,
            strategy_combinations.mirrored_strategy_with_two_gpus,
        ],
        enable_packed_handle=[True, False],
        tf_function=[combinations.tf_function, combinations.no_tf_function]))
class DistributedVariableTest(test.TestCase, parameterized.TestCase):

  def create_variable(self, strategy, initial_value, enable_packed_handle,
                      **kwargs):
    variables = []
    for device in strategy.extended.parameter_devices:
      with ops.device(device):
        variables.append(variables_lib.Variable(initial_value, **kwargs))
    return values_v2.DistributedVariable(
        variables, enable_packed_handle=enable_packed_handle)

  def assertReplica(self, distributed_var, values):
    for var, value in zip(distributed_var._variables, values):
      self.assertAllEqual(var, value)

  def testRead(self, strategy, enable_packed_handle, tf_function):
    v = self.create_variable(strategy, 0., enable_packed_handle)

    with ops.device(strategy.extended.parameter_devices[0]):
      v.assign(1.)
    with ops.device(strategy.extended.parameter_devices[1]):
      v.assign(2.)

    @tf_function
    def read_device0():
      with ops.device(strategy.extended.parameter_devices[0]):
        return v.read_value(), v.value()

    @tf_function
    def read_device1():
      with ops.device(strategy.extended.parameter_devices[1]):
        return v.read_value(), v.value()

    @tf_function
    def read_other_device():
      with ops.device("CPU:0"):
        return v.read_value(), v.value()

    self.assertAllEqual(read_device0(), [1., 1.])
    self.assertAllEqual(read_device1(), [2., 2.])
    self.assertAllEqual(read_other_device(), [1., 1.])

  def testAssign(self, strategy, enable_packed_handle, tf_function):
    v = self.create_variable(strategy, 0., enable_packed_handle)

    @tf_function
    def update_device0():
      with ops.device(strategy.extended.parameter_devices[0]):
        v.assign(1.)

    @tf_function
    def update_device1():
      with ops.device(strategy.extended.parameter_devices[1]):
        v.assign(2.)

    update_device0()
    update_device1()
    self.assertReplica(v, [1., 2.])

    with ops.device("CPU:0"):
      # Update the primary replica.
      v.assign(3.)
      self.assertReplica(v, [3., 2.])

  def testStrategyRun(self, strategy, enable_packed_handle, tf_function):
    if (test_util.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest("tpu doesn't support eager")
    v = self.create_variable(strategy, 0., enable_packed_handle)

    @tf_function
    def update(per_replica):
      v.assign(per_replica)

    @tf_function
    def read():
      return v.read_value()

    strategy.run(
        update, args=(test_util.create_per_replica(strategy, [1., 2.]),))
    self.assertReplica(v, [1., 2.])
    self.assertAllEqual(
        test_util.gather(strategy, strategy.run(read)), [1., 2.])


def _make_index_slices(values, indices, dense_shape=None):
  if dense_shape:
    dense_shape = array_ops.identity(dense_shape)
  return indexed_slices.IndexedSlices(
      array_ops.identity(values), array_ops.identity(indices), dense_shape)


if __name__ == "__main__":
  test_util.main()
