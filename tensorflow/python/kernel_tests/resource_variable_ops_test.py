# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.resource_variable_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import momentum
from tensorflow.python.training import saver
from tensorflow.python.training import training_util
from tensorflow.python.util import compat


class ResourceVariableOpsTest(test_util.TensorFlowTestCase):

  def tearDown(self):
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEqual(0, len(gc.garbage))

  def testHandleDtypeShapeMatch(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
      with self.assertRaises(ValueError):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant(0.0, dtype=dtypes.float32)).run()
      with self.assertRaises(ValueError):
        resource_variable_ops.assign_variable_op(handle,
                                                 constant_op.constant(
                                                     [0],
                                                     dtype=dtypes.int32)).run()
      resource_variable_ops.assign_variable_op(handle,
                                               constant_op.constant(
                                                   0,
                                                   dtype=dtypes.int32)).run()

  def testGPUInt64(self):
    if not context.context().num_gpus():
      return
    with context.eager_mode(), context.device("gpu:0"):
      v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.int64)
      self.assertAllEqual(1, v.numpy())

  def testEagerNameNotIdentity(self):
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0, name="a")
      v1 = resource_variable_ops.ResourceVariable(2.0, name="a")
      self.assertAllEqual(v0.numpy(), 1.0)
      self.assertAllEqual(v1.numpy(), 2.0)

  def testEagerNameNotNeeded(self):
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0)
      self.assertAllEqual(v0.numpy(), 1.0)

  def testReadVariableDtypeMismatchEager(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      resource_variable_ops.assign_variable_op(handle, 1)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Trying to read variable with wrong dtype. "
                                   "Expected float got int32."):
        _ = resource_variable_ops.read_variable_op(handle, dtype=dtypes.float32)

  def testEagerInitializedValue(self):
    with context.eager_mode():
      variable = resource_variable_ops.ResourceVariable(1.0, name="eager-init")
      self.assertAllEqual(variable.numpy(), 1.0)
      self.assertAllEqual(variable.initialized_value().numpy(), 1.0)

  def testEagerBool(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(False, name="bool_test")
      self.assertAllEqual(bool(v), False)

  def testDifferentAssignGraph(self):
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)
    ops.reset_default_graph()
    v.assign(2.0)  # Note: this fails if we run convert_to_tensor on not the
                   # variable graph.

  def testFetchHandle(self):
    with self.test_session():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      self.assertGreater(len(handle.eval()), 0)

  def testCachedValueReadBeforeWrite(self):
    with self.test_session() as sess:
      v = resource_variable_ops.ResourceVariable(0.0, caching_device="cpu:0")
      sess.run(v.initializer)
      value, _ = sess.run([v, v.assign_add(1.0)])
      self.assertAllEqual(value, 0.0)

  def testAssignVariableDtypeMismatchEager(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant([1]))
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Trying to assign variable with wrong "
                                   "dtype. Expected int32 got float."):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([1.], dtype=dtypes.float32))

  def testUnprintableHandle(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      self.assertIn("<unprintable>", str(handle))
      self.assertIn("<unprintable>", repr(handle))

  @test_util.run_in_graph_and_eager_modes()
  def testDtypeSurvivesIdentity(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    id_handle = array_ops.identity(handle)
    self.evaluate(resource_variable_ops.assign_variable_op(
        id_handle, constant_op.constant(0, dtype=dtypes.int32)))

  @test_util.run_in_graph_and_eager_modes()
  def testCreateRead(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    value = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertAllEqual(1, value)

  @test_util.run_in_graph_and_eager_modes()
  def testManyAssigns(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    create = resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32))
    with ops.control_dependencies([create]):
      first_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32)
    with ops.control_dependencies([first_read]):
      write = resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(2, dtype=dtypes.int32))
    with ops.control_dependencies([write]):
      second_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32)
    f, s = self.evaluate([first_read, second_read])
    self.assertEqual(f, 1)
    self.assertEqual(s, 2)

  @test_util.run_in_graph_and_eager_modes()
  def testAssignAdd(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    self.evaluate(resource_variable_ops.assign_add_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    read = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertEqual(read, 2)

  @test_util.run_in_graph_and_eager_modes()
  def testScatterAdd(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_add(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterSub(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterMul(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant([[5]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterDiv(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterMin(self):
    with ops.device("cpu:0"):
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      self.evaluate(
          resource_variable_ops.assign_variable_op(handle,
                                                   constant_op.constant(
                                                       [[6]],
                                                       dtype=dtypes.int32)))
      self.evaluate(
          resource_variable_ops.resource_scatter_min(handle, [0],
                                                     constant_op.constant(
                                                         [[3]],
                                                         dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  def testMetagraph(self):
    with ops.Graph().as_default():
      with variable_scope.variable_scope("foo", use_resource=True):
        a = variable_scope.get_variable("a", initializer=10.0)

      momentum.MomentumOptimizer(
          learning_rate=0.001, momentum=0.1).minimize(
              a,
              colocate_gradients_with_ops=True,
              global_step=training_util.get_or_create_global_step())

      graph = ops.get_default_graph()
      meta_graph_def = saver.export_meta_graph(graph=graph)

    with ops.Graph().as_default():
      saver.import_meta_graph(meta_graph_def, import_scope="")
      meta_graph_two = saver.export_meta_graph(graph=graph)
    self.assertEqual(meta_graph_def, meta_graph_two)

  @test_util.run_in_graph_and_eager_modes()
  def testScatterMax(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterAddScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_add(
            handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterSubScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterMulScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant(5, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterDivScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterMinScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_min(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes()
  def testScatterMaxScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)))
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])

  def testScatterUpdateString(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.string, shape=[1, 1])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant([["a"]], dtype=dtypes.string)))
    self.evaluate(resource_variable_ops.resource_scatter_update(
        handle, [0], constant_op.constant([["b"]], dtype=dtypes.string)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.string)
    self.assertEqual(compat.as_bytes(self.evaluate(read)[0][0]),
                     compat.as_bytes("b"))

  def testScatterUpdateStringScalar(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.string, shape=[1, 1])
    self.evaluate(
        resource_variable_ops.assign_variable_op(handle,
                                                 constant_op.constant(
                                                     [["a"]],
                                                     dtype=dtypes.string)))
    self.evaluate(
        resource_variable_ops.resource_scatter_update(handle, [0],
                                                      constant_op.constant(
                                                          "b",
                                                          dtype=dtypes.string)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.string)
    self.assertEqual(
        compat.as_bytes(self.evaluate(read)[0][0]), compat.as_bytes("b"))

  # TODO(alive): get this to work in Eager mode.
  def testGPU(self):
    with self.test_session(use_gpu=True):
      abc = variable_scope.get_variable(
          "abc",
          shape=[1],
          initializer=init_ops.ones_initializer(),
          use_resource=True)

      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(
          self.evaluate(
              resource_variable_ops.var_is_initialized_op(abc.handle)),
          True)

  def testScatterBool(self):
    with context.eager_mode():
      ref = resource_variable_ops.ResourceVariable(
          [False, True, False], trainable=False)
      indices = math_ops.range(3)
      updates = constant_op.constant([True, True, True])
      state_ops.scatter_update(ref, indices, updates)
      self.assertAllEqual(ref.read_value(), [True, True, True])

  @test_util.run_in_graph_and_eager_modes()
  def testConstraintArg(self):
    constraint = lambda x: x
    v = resource_variable_ops.ResourceVariable(
        initial_value=lambda: 1, constraint=constraint, name="var0")
    self.assertEqual(v.constraint, constraint)

    constraint = 0
    with self.assertRaises(ValueError):
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, constraint=constraint, name="var1")

  # TODO(alive): how should this work in Eager mode?
  def testInitFn(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, dtype=dtypes.float32)
      self.assertEqual(v.handle.op.colocation_groups(),
                       v.initializer.inputs[1].op.colocation_groups())

  def testHandleNumpy(self):
    with context.eager_mode():
      with self.assertRaises(ValueError):
        resource_variable_ops.ResourceVariable(
            1.0, name="handle-numpy").handle.numpy()

  def testCountUpTo(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(0, name="upto")
      self.assertAllEqual(v.count_up_to(1), 0)
      with self.assertRaises(errors.OutOfRangeError):
        v.count_up_to(1)

  def testCountUpToFunction(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(0, name="upto")
      self.assertAllEqual(state_ops.count_up_to(v, 1), 0)
      with self.assertRaises(errors.OutOfRangeError):
        state_ops.count_up_to(v, 1)

  @test_util.run_in_graph_and_eager_modes()
  def testInitFnDtype(self):
    v = resource_variable_ops.ResourceVariable(
        initial_value=lambda: 1, dtype=dtypes.float32, name="var0")
    self.assertEqual(dtypes.float32, v.value().dtype)

  @test_util.run_in_graph_and_eager_modes()
  def testInitFnNoDtype(self):
    v = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                               name="var2")
    self.assertEqual(dtypes.int32, v.value().dtype)

  @test_util.run_in_graph_and_eager_modes()
  def testInitializeAllVariables(self):
    v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.float32,
                                               name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(1.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testOperatorOverload(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(2.0, self.evaluate(v + v))

  @test_util.run_in_graph_and_eager_modes()
  def testAssignMethod(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign(2.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign(3.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign(4.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testLoad(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    v.load(2.0)
    self.assertEqual(2.0, self.evaluate(v.value()))

  def testVariableDefInitializedInstances(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      v_def = resource_variable_ops.ResourceVariable(
          initial_value=constant_op.constant(3.0)).to_proto()

    with ops.Graph().as_default(), self.test_session() as sess:
      # v describes a VariableDef-based variable without an initial value.
      v = resource_variable_ops.ResourceVariable(variable_def=v_def)
      self.assertEqual(3.0, sess.run(v.initialized_value()))

      # initialized_value should not rerun the initializer_op if the variable
      # has already been initialized elsewhere.
      sess.run(v.assign(1.0))
      self.assertEqual(1.0, v.initialized_value().eval())

    v_def.ClearField("initial_value_name")
    with ops.Graph().as_default(), self.test_session() as sess:
      # Restoring a legacy VariableDef proto that does not have
      # initial_value_name set should still work.
      v = resource_variable_ops.ResourceVariable(variable_def=v_def)
      # We should also be able to re-export the variable to a new meta graph.
      self.assertProtoEquals(v_def, v.to_proto())
      # But attempts to use initialized_value will result in errors.
      with self.assertRaises(ValueError):
        sess.run(v.initialized_value())

  def testTrainableInProto(self):
    with ops.Graph().as_default():
      non_trainable_variable = resource_variable_ops.ResourceVariable(
          trainable=False,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          False,
          resource_variable_ops.ResourceVariable(
              variable_def=non_trainable_variable.to_proto())
          .trainable)
      trainable_variable = resource_variable_ops.ResourceVariable(
          trainable=True,
          initial_value=constant_op.constant(10.0))
      self.assertEqual(
          True,
          resource_variable_ops.ResourceVariable(
              variable_def=trainable_variable.to_proto())
          .trainable)

  @test_util.run_in_graph_and_eager_modes()
  def testSparseRead(self):
    with self.test_session():
      init_value = np.reshape(np.arange(np.power(4, 3)), (4, 4, 4))
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant(init_value, dtype=dtypes.int32), name="var3")
      self.evaluate(variables.global_variables_initializer())

      value = self.evaluate(v.sparse_read([0, 3, 1, 2]))
      self.assertAllEqual(init_value[[0, 3, 1, 2], ...], value)

  def testToFromProto(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()

      w = resource_variable_ops.ResourceVariable.from_proto(v.to_proto())
      self.assertEquals(2, math_ops.add(w, 1).eval())

      self.assertEquals(v._handle, w._handle)
      self.assertEquals(v._graph_element, w._graph_element)

  @test_util.run_in_graph_and_eager_modes()
  def testAssignAddMethod(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_add(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_add(1.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_add(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testAssignSubMethod(self):
    v = resource_variable_ops.ResourceVariable(3.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_sub(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_sub(1.0, read_value=True)
    self.assertEqual(1.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_sub(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(0.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testDestroyResource(self):
    v = resource_variable_ops.ResourceVariable(3.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(3.0, self.evaluate(v.value()))
    self.evaluate(resource_variable_ops.destroy_resource_op(v.handle))
    with self.assertRaises(errors.FailedPreconditionError):
      self.evaluate(v.value())
    # Handle to a resource not actually created.
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    # Should raise no exception
    self.evaluate(resource_variable_ops.destroy_resource_op(
        handle, ignore_lookup_error=True))

  def testAssignDifferentShapes(self):
    with self.test_session() as sess, variable_scope.variable_scope(
        "foo", use_resource=True):
      var = variable_scope.get_variable("x", shape=[1, 1], dtype=dtypes.float32)
      placeholder = array_ops.placeholder(dtypes.float32)
      assign = var.assign(placeholder)
      sess.run(
          [assign],
          feed_dict={placeholder: np.zeros(shape=[2, 2], dtype=np.float32)})

  def testAssignDifferentShapesEager(self):
    with context.eager_mode():
      with variable_scope.variable_scope("foo"):
        var = variable_scope.get_variable("x", shape=[1, 1],
                                          dtype=dtypes.float32)
        with self.assertRaisesRegexp(ValueError,
                                     "Shapes.*and.*are incompatible"):
          assign = var.assign(np.zeros(shape=[2, 2]))
          self.evaluate(assign)

  def testDtypeAfterFromProto(self):
    v = resource_variable_ops.ResourceVariable(2.0)
    w = resource_variable_ops.ResourceVariable.from_proto(v.to_proto())
    self.assertIsInstance(w.dtype, dtypes.DType)
    self.assertEqual(v.dtype, w.dtype)

  # TODO(alive): get caching to work in eager mode.
  def testCachingDevice(self):
    with ops.device("/job:server/task:1"):
      v = resource_variable_ops.ResourceVariable(
          2.0, caching_device="/job:localhost")
      self.assertEqual("/job:localhost", v.value().device)
      with self.assertRaises(ValueError):
        _ = v.value().op.get_attr("_class")

    with ops.colocate_with(v.op):
      w = resource_variable_ops.ResourceVariable(
          2.0, caching_device="/job:localhost")
      self.assertEqual("/job:localhost", w.value().device)
      with self.assertRaises(ValueError):
        _ = w.value().op.get_attr("_class")

  def testSharedName(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(300.0, name="var4")
      variables.global_variables_initializer().run()

      w = resource_variable_ops.var_handle_op(
          dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="var4",
          # Needed in Eager since we get a unique container name by default.
          container=ops.get_default_graph()._container)
      w_read = resource_variable_ops.read_variable_op(w, v.dtype.base_dtype)
      self.assertEqual(300.0, w_read.eval())

      x = resource_variable_ops.var_handle_op(
          dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="var5",
          container=ops.get_default_graph()._container)
      with self.assertRaisesOpError("Resource .*/var5/.* does not exist"):
        resource_variable_ops.read_variable_op(x, v.dtype.base_dtype).eval()

  def testSharedNameWithNamescope(self):
    with self.test_session():
      with ops.name_scope("foo"):
        v = resource_variable_ops.ResourceVariable(300.0, name="var6")
        self.assertEqual("foo/var6", v._shared_name)  # pylint: disable=protected-access
        self.assertEqual("foo/var6:0", v.name)
        self.evaluate(variables.global_variables_initializer())

      w = resource_variable_ops.var_handle_op(
          dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="foo/var6",
          # Needed in Eager since we get a unique container name by default.
          container=ops.get_default_graph()._container)
      w_read = resource_variable_ops.read_variable_op(w, v.dtype.base_dtype)
      self.assertEqual(300.0, self.evaluate(w_read))

  @test_util.run_in_graph_and_eager_modes()
  def testShape(self):
    v = resource_variable_ops.ResourceVariable(
        name="var4", initial_value=array_ops.ones(shape=[10, 20, 35]))
    self.assertEqual("(10, 20, 35)", str(v.shape))
    self.assertEqual("(10, 20, 35)", str(v.get_shape()))
    self.assertEqual("(10, 20, 35)", str(v.value().shape))
    self.assertEqual("(3, 20, 35)", str(v.sparse_read([0, 1, 2]).shape))
    if not context.executing_eagerly():
      self.assertEqual(
          "<unknown>",
          str(v.sparse_read(array_ops.placeholder(dtypes.int32)).shape))

  def testSetInitialValue(self):
    with self.test_session():
      # Initialize variable with a value different from the initial value passed
      # in the constructor.
      v = resource_variable_ops.ResourceVariable(2.0)
      v.initializer.run(feed_dict={v.initial_value: 3.0})
      self.assertEqual(3.0, v.value().eval())

  def testControlFlowInitialization(self):
    """Expects an error if an initializer is in a control-flow scope."""

    def cond(i, _):
      return i < 10

    def body(i, _):
      zero = array_ops.zeros([], dtype=dtypes.int32)
      v = resource_variable_ops.ResourceVariable(initial_value=zero)
      return (i + 1, v.read_value())

    with self.assertRaisesRegexp(ValueError, "inside a control-flow"):
      control_flow_ops.while_loop(cond, body, [0, 0])

  def testVariableEager(self):
    with context.eager_mode():
      init = array_ops.ones(shape=[10, 20, 35], dtype=dtypes.int32)
      constraint = lambda x: x
      with ops.name_scope("foo"):
        v = resource_variable_ops.ResourceVariable(
            name="var7",
            initial_value=init,
            caching_device="cpu:0",
            constraint=constraint)
      # Test properties
      self.assertEqual(dtypes.int32, v.dtype)
      self.assertEqual("foo/var7:0", v.name)
      self.assertAllEqual([10, 20, 35], v.shape.as_list())
      self.assertTrue(isinstance(v.handle, ops.EagerTensor))
      self.assertEqual(constraint, v.constraint)
      self.assertAllEqual(init.numpy(), v.read_value().numpy())
      self.assertAllEqual(init.numpy(), v.value().numpy())

      # Callable init.
      callable_init = lambda: init * 2
      v2 = resource_variable_ops.ResourceVariable(
          initial_value=callable_init, name="var7")
      self.assertEqual("var7:0", v2.name)
      self.assertAllEqual(2 * init.numpy(), v2.read_value().numpy())

      # Test assign_add.
      new_v2_val = v2.assign_add(v.read_value())
      self.assertAllEqual(v.read_value().numpy() * 3, new_v2_val.numpy())

      # Test assign_sub.
      new_v2_val = v2.assign_sub(v.read_value())
      self.assertAllEqual(v.read_value().numpy() * 2, new_v2_val.numpy())

      # Test assign.
      v2.assign(v.read_value())
      self.assertAllEqual(v.read_value().numpy(), v2.read_value().numpy())

      # Test load
      v2.load(2 * v.read_value())
      self.assertAllEqual(2 * v.read_value().numpy(), v2.read_value().numpy())

      # Test convert_to_tensor
      t = ops.convert_to_tensor(v)
      self.assertAllEqual(t.numpy(), v.read_value().numpy())

      # Test operations
      self.assertAllEqual((v * 2).numpy(), (v + v).numpy())

  def testContainerEager(self):
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                                  name="same")
      with ops.container("different"):
        v2 = resource_variable_ops.ResourceVariable(initial_value=lambda: 0,
                                                    name="same")
      v2.assign(2)
      self.assertEqual(1, v1.read_value().numpy())
      self.assertEqual(2, v2.read_value().numpy())

  def testDestruction(self):
    with context.eager_mode():
      var = resource_variable_ops.ResourceVariable(initial_value=1.0,
                                                   name="var8")
      var_handle = var._handle
      del var
      with self.assertRaisesRegexp(errors.NotFoundError,
                                   r"Resource .* does not exist."):
        resource_variable_ops.destroy_resource_op(var_handle,
                                                  ignore_lookup_error=False)

  def testScatterUpdate(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="update")
      state_ops.scatter_update(v, [1], [3.0])
      self.assertAllEqual([1.0, 3.0], v.numpy())

  def testScatterAddStateOps(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="add")
      state_ops.scatter_add(v, [1], [3])
      self.assertAllEqual([1.0, 5.0], v.numpy())

  def testScatterNdAddStateOps(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(
          [1, 1, 1, 1, 1, 1, 1, 1], dtype=dtypes.float32, name="add")
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, 12, 1, 11, 10, 1, 1, 13])
      state_ops.scatter_nd_add(v, indices, updates)
      self.assertAllClose(expected, v.numpy())

  def testScatterUpdateCast(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="update")
      state_ops.scatter_update(v, [1], [3])
      self.assertAllEqual([1.0, 3.0], v.numpy())

  @test_util.run_in_graph_and_eager_modes()
  def testScatterUpdateInvalidArgs(self):
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3], name="update")
    # The exact error and message differ between graph construction (where the
    # error is realized during shape inference at graph construction time) and
    # eager execution (where the error is realized during kernel execution).
    with self.assertRaisesRegexp(Exception, r"shape.*2.*3"):
      state_ops.scatter_update(v, [0, 1], [0, 1, 2])


if __name__ == "__main__":
  test.main()
