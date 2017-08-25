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
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ResourceVariableOpsTest(test_util.TensorFlowTestCase):

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

  def testReadVariableDtypeMismatch(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Trying to read variable with wrong dtype. "
                                   "Expected float got int32."):
        _ = resource_variable_ops.read_variable_op(handle, dtype=dtypes.float32)

  def testAssignVariableDtypeMismatch(self):
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

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testScatterAdd(self):
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant([[1]], dtype=dtypes.int32)))
    self.evaluate(resource_variable_ops.resource_scatter_add(
        handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

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

  # TODO(alive): fix bug in convert_to_tensor; get this to work in Eager.
  def testConstraintArg(self):
    constraint = lambda x: x
    v = resource_variable_ops.ResourceVariable(
        initial_value=lambda: 1, constraint=constraint)
    self.assertEqual(v.constraint, constraint)

    constraint = 0
    with self.assertRaises(ValueError):
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, constraint=constraint)

  # TODO(alive): how should this work in Eager mode?
  def testInitFn(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, dtype=dtypes.float32)
      self.assertEqual(v.handle.op.colocation_groups(),
                       v.initializer.inputs[1].op.colocation_groups())

  # TODO(alive): fix bug in convert_to_tensor; get this to work in Eager.
  def testInitFnDtype(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, dtype=dtypes.float32)
      self.assertEqual(dtypes.float32, v.value().dtype)

  # TODO(alive): fix bug in convert_to_tensor; get this to work in Eager.
  def testInitFnNoDtype(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(initial_value=lambda: 1)
      self.assertEqual(dtypes.int32, v.value().dtype)

  @test_util.run_in_graph_and_eager_modes()
  def testInitializeAllVariables(self):
    v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.float32)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(1.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testOperatorOverload(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(2.0, self.evaluate(v + v))

  @test_util.run_in_graph_and_eager_modes()
  def testAssignMethod(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign(2.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testLoad(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())
    v.load(2.0)
    self.assertEqual(2.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testSparseRead(self):
    with self.test_session():
      init_value = np.reshape(np.arange(np.power(4, 3)), (4, 4, 4))
      v = resource_variable_ops.ResourceVariable(
          constant_op.constant(init_value, dtype=dtypes.int32), name="var0")
      self.evaluate(variables.global_variables_initializer())

      value = self.evaluate(v.sparse_read([0, 3, 1, 2]))
      self.assertAllEqual(init_value[[0, 3, 1, 2], ...], value)

  def testToFromProto(self):
    with self.test_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      variables.global_variables_initializer().run()

      w = resource_variable_ops.ResourceVariable.from_proto(v.to_proto())
      self.assertEquals(2, math_ops.add(w, 1).eval())

  @test_util.run_in_graph_and_eager_modes()
  def testAssignAddMethod(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_add(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testAssignSubMethod(self):
    v = resource_variable_ops.ResourceVariable(3.0)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_sub(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes()
  def testDestroyResource(self):
    v = resource_variable_ops.ResourceVariable(3.0)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(3.0, self.evaluate(v.value()))
    self.evaluate(resource_variable_ops.destroy_resource_op(v.handle))
    with self.assertRaises(errors.NotFoundError):
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
      with self.assertRaisesRegexp(ValueError, "No attr named '_class'"):
        _ = v.value().op.get_attr("_class")

    with ops.colocate_with(v.op):
      w = resource_variable_ops.ResourceVariable(
          2.0, caching_device="/job:localhost")
      self.assertEqual("/job:localhost", w.value().device)
      with self.assertRaisesRegexp(ValueError, "No attr named '_class'"):
        _ = w.value().op.get_attr("_class")

  @test_util.run_in_graph_and_eager_modes()
  def testSharedName(self):
    v = resource_variable_ops.ResourceVariable(300.0, name="var1")
    self.evaluate(variables.global_variables_initializer())

    w = resource_variable_ops.var_handle_op(
        dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="var1")
    w_read = resource_variable_ops.read_variable_op(w, v.dtype.base_dtype)
    self.assertEqual(300.0, self.evaluate(w_read))

    x = resource_variable_ops.var_handle_op(
        dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="var2")
    if context.in_graph_mode():
      with self.assertRaisesOpError("Resource .*/var2/.* does not exist"):
        x_read = resource_variable_ops.read_variable_op(x, v.dtype.base_dtype)
        self.evaluate(x_read)
    else:
      with self.assertRaisesRegexp(errors.NotFoundError,
                                   "Attempted to read a nonexistent variable."):
        _ = resource_variable_ops.read_variable_op(x, v.dtype.base_dtype)

  @test_util.run_in_graph_and_eager_modes()
  def testSharedNameWithNamescope(self):
    with ops.name_scope("foo"):
      v = resource_variable_ops.ResourceVariable(300.0, name="var3")
      self.evaluate(variables.global_variables_initializer())

    w = resource_variable_ops.var_handle_op(
        dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="foo/var3")
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
    if context.in_graph_mode():
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
            name="var5",
            initial_value=init,
            caching_device="cpu:0",
            constraint=constraint)
      # Test properties
      self.assertEqual(dtypes.int32, v.dtype)
      self.assertEqual("foo/var5:0", v.name)
      self.assertAllEqual([10, 20, 35], v.shape.as_list())
      self.assertAllEqual(init.device, v.device)
      self.assertTrue(isinstance(v.handle, ops.EagerTensor))
      self.assertEqual(constraint, v.constraint)
      self.assertAllEqual(init.numpy(), v.read_value().numpy())
      self.assertAllEqual(init.numpy(), v.value().numpy())

      # Callable init.
      callable_init = lambda: init * 2
      v2 = resource_variable_ops.ResourceVariable(
          initial_value=callable_init, name="var6")
      self.assertEqual("var6:0", v2.name)
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


if __name__ == "__main__":
  test.main()
