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

import copy
import gc
import os
import pickle
import re

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import tensor_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
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


@test_util.with_control_flow_v2
class ResourceVariableOpsTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):

  def tearDown(self):
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEmpty(gc.garbage)
    super(ResourceVariableOpsTest, self).tearDown()

  @test_util.run_deprecated_v1
  def testHandleDtypeShapeMatch(self):
    with self.cached_session():
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

  @test_util.run_gpu_only
  def testGPUInt64(self):
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
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          "Trying to read variable with wrong dtype. "
          "Expected float got int32"):
        _ = resource_variable_ops.read_variable_op(handle, dtype=dtypes.float32)

  def testEagerInitializedValue(self):
    with context.eager_mode():
      variable = resource_variable_ops.ResourceVariable(1.0, name="eager-init")
      self.assertAllEqual(variable.numpy(), 1.0)
      self.assertAllEqual(variable.initialized_value().numpy(), 1.0)

  def testInitializeVariableUsingInitializedValue(self):
    var1 = resource_variable_ops.ResourceVariable(1.0, name="var1")
    var2 = resource_variable_ops.ResourceVariable(var1.initialized_value(),
                                                  name="var2")
    self.assertAllEqual(var2.initialized_value(), 1.0)

  def testEagerBool(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(False, name="bool_test")
      self.assertAllEqual(bool(v), False)

  def testEagerDeepCopy(self):
    with context.eager_mode():
      init_value = np.ones((4, 4, 4))
      variable = resource_variable_ops.ResourceVariable(init_value,
                                                        name="init")

      copied_variable = copy.deepcopy(variable)
      self.assertEqual(variable.name, copied_variable.name)
      self.assertEqual(variable.shape, copied_variable.shape)
      self.assertEqual(variable.device, copied_variable.device)

      # The copied variable should have the same value as the original.
      self.assertAllEqual(variable.numpy(), copied_variable.numpy())

      # Updates to the copy should not be reflected in the original.
      copied_variable.assign(4 * np.ones((4, 4, 4)))
      self.assertNotAllEqual(variable.numpy(), copied_variable.numpy())

  @test_util.run_deprecated_v1
  def testGraphDeepCopy(self):
    with self.cached_session():
      init_value = np.ones((4, 4, 4))
      variable = resource_variable_ops.ResourceVariable(init_value,
                                                        name="init")
      with self.assertRaises(NotImplementedError):
        copy.deepcopy(variable)

  @test_util.run_in_graph_and_eager_modes
  def testStridedSliceAssign(self):
    v = resource_variable_ops.ResourceVariable([1.0, 2.0])
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v[0].assign(2.0))
    self.assertAllEqual(self.evaluate(v), [2.0, 2.0])

  @test_util.run_in_graph_and_eager_modes
  def testVariableShape(self):
    v = resource_variable_ops.ResourceVariable([1., 1.])
    self.assertAllEqual(
        tensor_util.constant_value(
            resource_variable_ops.variable_shape(v.handle)),
        [2])

  @test_util.run_deprecated_v1
  def testDifferentAssignGraph(self):
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)
    ops.reset_default_graph()
    v.assign(2.0)  # Note: this fails if we run convert_to_tensor on not the
    # variable graph.

  @test_util.run_deprecated_v1
  def testFetchHandle(self):
    with self.cached_session():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      self.assertNotEmpty(handle.eval())

  @test_util.run_deprecated_v1
  def testCachedValueReadBeforeWrite(self):
    with self.cached_session() as sess:
      v = resource_variable_ops.ResourceVariable(0.0, caching_device="cpu:0")
      self.evaluate(v.initializer)
      value, _ = sess.run([v, v.assign_add(1.0)])
      self.assertAllEqual(value, 0.0)

  def testAssignVariableDtypeMismatchEager(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant([1]))
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError, "Trying to assign variable with wrong "
          "dtype. Expected int32 got float"):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([1.], dtype=dtypes.float32))

  def testRepr(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(1)
      text = "%r" % v
      self.assertEqual(
          "<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=1>", text)

  def testReprUnavailable(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(1)

      # Monkey-patch this variable to not have an available value
      def broken_read():
        raise ValueError("This doesn't work")

      v.read_value = broken_read
      text = "%r" % v
      self.assertEqual("<tf.Variable 'Variable:0' shape=() dtype=int32,"
                       " numpy=<unavailable>>", text)

  def testUnprintableHandle(self):
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo")
      self.assertIn("<unprintable>", str(handle))
      self.assertIn("<unprintable>", repr(handle))

  @test_util.run_in_graph_and_eager_modes
  def testDtypeSurvivesIdentity(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    id_handle = array_ops.identity(handle)
    self.evaluate(resource_variable_ops.assign_variable_op(
        id_handle, constant_op.constant(0, dtype=dtypes.int32)))

  def testUnreadOpName(self):
    v = resource_variable_ops.ResourceVariable(1.0)
    self.assertNotEqual(v.name, v.assign_add(1.0).name)

  @test_util.run_in_graph_and_eager_modes
  def testCreateRead(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    value = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertAllEqual(1, value)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  def testAssignAdd(self):
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    self.evaluate(resource_variable_ops.assign_add_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)))
    read = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32))
    self.assertEqual(read, 2)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  def testGradientGatherNd(self):
    v = resource_variable_ops.ResourceVariable(
        np.random.uniform(size=[2, 2]), dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      l = array_ops.gather_nd(v, [[1, 1]])
      l = math_ops.reduce_sum(l)

    grads = tape.gradient(l, v)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(self.evaluate(grads), [[0., 0.], [0., 1.]])

  @test_util.run_deprecated_v1
  def testDefaultGradientDtype(self):
    v = resource_variable_ops.ResourceVariable(
        np.random.uniform(size=[2, 2]), dtype=dtypes.float64)

    c = constant_op.constant(1.)
    identity = array_ops.identity_n([c, v.handle])
    # TODO(b/137403775): Remove this.
    custom_gradient.copy_handle_data(v.handle, identity[1])

    g = gradients_impl.gradients(identity[0], [c, v.handle])
    self.assertEqual(g[1].dtype, dtypes.float64)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(g[1], [[0., 0.], [0., 0.]])

  @test_util.run_deprecated_v1
  def testUnconnectedGradientZeros(self):
    b = resource_variable_ops.ResourceVariable(initial_value=[[3., 4.]])
    c = constant_op.constant(0.)
    g = gradients_impl.gradients(c, [b], unconnected_gradients="zero")[0]
    self.assertAllEqual(g.shape.as_list(), [1, 2])

  @test_util.run_deprecated_v1
  def testGradientCondInWhileLoop(self):
    v = resource_variable_ops.ResourceVariable(initial_value=1.0)
    def cond(i, unused_x):
      return i < 1

    def body(i, x):
      def true():
        return x + v
      def false():
        return 2.0 * v
      return i + 1, control_flow_ops.cond(i > 0, true, false)

    _, x = control_flow_ops.while_loop(cond, body, [0, 0.0])
    # Computing gradients does not produce an exception:
    g = gradients_impl.gradients(x, v)
    self.evaluate(variables.global_variables_initializer())
    # Only the false branch is taken so the gradient is 2.
    self.assertAllEqual(g[0], 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testGradientGatherNdIndexedSlices(self):
    v = resource_variable_ops.ResourceVariable(
        np.random.uniform(size=[2, 2]), dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      l = array_ops.gather_nd(v, [[1], [1]])
      l = math_ops.reduce_sum(l)

    grads = tape.gradient(l, v)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(self.evaluate(grads.values), [[1., 1.], [1., 1.]])

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  def testEagerPickle(self):
    with context.eager_mode():
      tmp_dir = self.get_temp_dir()
      fname = os.path.join(tmp_dir, "var.pickle")
      with open(fname, "wb") as f:
        v = resource_variable_ops.ResourceVariable(
            10.0,
            dtype=dtypes.float16,
            name="v")
        pickle.dump(v, f)

      with open(fname, "rb") as f:
        new_v = pickle.load(f)
        self.assertEqual(new_v.name, v.name)
        self.assertEqual(new_v.shape, v.shape)
        self.assertEqual(new_v.dtype, v.dtype)
        self.assertEqual(new_v.trainable, v.trainable)
        self.assertAllEqual(new_v.numpy(), v.numpy())

  @test_util.run_in_graph_and_eager_modes
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

  def testUseResource(self):
    v = variables.VariableV1(1.0, use_resource=True)
    self.assertIsInstance(v, resource_variable_ops.ResourceVariable)

  def testEagerNoUseResource(self):
    with context.eager_mode():
      v = variables.Variable(1.0)
      self.assertIsInstance(v, resource_variable_ops.ResourceVariable)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  def testScatterAddVariableMethod(self):
    v = resource_variable_ops.ResourceVariable([0.0, 1.5], name="add")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_add(ops.IndexedSlices(indices=[1], values=[2.5])))
    self.assertAllEqual([0.0, 4.0], self.evaluate(v))

  @test_util.run_in_graph_and_eager_modes
  def testScatterSubVariableMethod(self):
    v = resource_variable_ops.ResourceVariable([0.0, 2.5], name="sub")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_sub(ops.IndexedSlices(indices=[1], values=[1.5])))
    self.assertAllEqual([0.0, 1.0], self.evaluate(v))

  @test_util.run_in_graph_and_eager_modes
  def testScatterMaxVariableMethod(self):
    v = resource_variable_ops.ResourceVariable([0.0, 4.0], name="max1")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_max(ops.IndexedSlices(indices=[1], values=[5.0])))
    self.assertAllEqual([0.0, 5.0], self.evaluate(v))

    v = resource_variable_ops.ResourceVariable([0.0, 3.5], name="max2")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_max(ops.IndexedSlices(indices=[1], values=[2.0])))
    self.assertAllEqual([0.0, 3.5], self.evaluate(v))

  @test_util.run_in_graph_and_eager_modes
  def testScatterMinVariableMethod(self):
    v = resource_variable_ops.ResourceVariable([0.0, 4.0], name="min1")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_min(ops.IndexedSlices(indices=[1], values=[5.0])))
    self.assertAllEqual([0.0, 4.0], self.evaluate(v))

    v = resource_variable_ops.ResourceVariable([0.0, 3.5], name="min2")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_min(ops.IndexedSlices(indices=[1], values=[2.0])))
    self.assertAllEqual([0.0, 2.0], self.evaluate(v))

  @test_util.run_in_graph_and_eager_modes
  def testScatterMulVariableMethod(self):
    v = resource_variable_ops.ResourceVariable([0.0, 4.0], name="mul")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_mul(ops.IndexedSlices(indices=[1], values=[3.0])))
    self.assertAllEqual([0.0, 12.0], self.evaluate(v))

  @test_util.run_in_graph_and_eager_modes
  def testScatterDivVariableMethod(self):
    v = resource_variable_ops.ResourceVariable([0.0, 6.0], name="div")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_div(ops.IndexedSlices(indices=[1], values=[2.0])))
    self.assertAllEqual([0.0, 3.0], self.evaluate(v))

  @test_util.run_in_graph_and_eager_modes
  def testScatterUpdateVariableMethod(self):
    v = resource_variable_ops.ResourceVariable([0.0, 6.0], name="update")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(
        v.scatter_update(ops.IndexedSlices(indices=[1], values=[3.0])))
    self.assertAllEqual([0.0, 3.0], self.evaluate(v))

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
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
    with test_util.use_gpu():
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

  @test_util.run_in_graph_and_eager_modes
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
  @test_util.run_deprecated_v1
  def testInitFn(self):
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(
          initial_value=lambda: 1, dtype=dtypes.float32)
      self.assertEqual(v.handle.op.colocation_groups(),
                       v.initializer.inputs[1].op.colocation_groups())

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

  @test_util.run_in_graph_and_eager_modes
  def testInitFnDtype(self):
    v = resource_variable_ops.ResourceVariable(
        initial_value=lambda: 1, dtype=dtypes.float32, name="var0")
    self.assertEqual(dtypes.float32, v.value().dtype)

  @test_util.run_in_graph_and_eager_modes
  def testInitFnNoDtype(self):
    v = resource_variable_ops.ResourceVariable(initial_value=lambda: 1,
                                               name="var2")
    self.assertEqual(dtypes.int32, v.value().dtype)

  @test_util.run_in_graph_and_eager_modes
  def testInitializeAllVariables(self):
    v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.float32,
                                               name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(1.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testOperatorOverload(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(2.0, self.evaluate(v + v))

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  def testLoad(self):
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    v.load(2.0)
    self.assertEqual(2.0, self.evaluate(v.value()))

  def testShapePassedToGradient(self):
    with ops.Graph().as_default():
      @custom_gradient.custom_gradient
      def differentiable_scatter_update(handle, indices, values):
        with ops.control_dependencies([
            resource_variable_ops.resource_scatter_update(
                handle, indices, values)]):
          new_handle = array_ops.identity(handle)

        def grad(dresult):
          self.assertIsNotNone(
              tensor_util.constant_value(dresult.dense_shape))
          return [dresult, None, None]

        return new_handle, grad

      var = variable_scope.get_variable(
          "foo", shape=[20], initializer=init_ops.zeros_initializer,
          dtype=dtypes.float64, use_resource=True)

      indices = math_ops.range(10)
      updates = math_ops.range(9, -1, -1, dtype=dtypes.float64)
      new_handle = differentiable_scatter_update(var.handle, indices, updates)
      gathered = resource_variable_ops.resource_gather(
          new_handle, indices, dtype=var.dtype)
      gradients_impl.gradients([gathered], [updates])

  def testToFromProtoCachedValue(self):
    with ops.Graph().as_default():
      v_def = resource_variable_ops.ResourceVariable(
          initial_value=constant_op.constant(3.0)).to_proto()
      v_prime = resource_variable_ops.ResourceVariable(variable_def=v_def)
      self.assertIsNone(getattr(v_prime, "_cached_value", None))

      other_v_def = resource_variable_ops.ResourceVariable(
          caching_device="cpu:0",
          initial_value=constant_op.constant(3.0)).to_proto()
      other_v_prime = resource_variable_ops.ResourceVariable(
          variable_def=other_v_def)
      self.assertIsNotNone(other_v_prime._cached_value)

  def testVariableDefInitializedInstances(self):
    with ops.Graph().as_default(), self.cached_session():
      v_def = resource_variable_ops.ResourceVariable(
          initial_value=constant_op.constant(3.0)).to_proto()

    with ops.Graph().as_default(), self.cached_session():
      # v describes a VariableDef-based variable without an initial value.
      v = resource_variable_ops.ResourceVariable(variable_def=v_def)
      self.assertEqual(3.0, self.evaluate(v.initialized_value()))

      # initialized_value should not rerun the initializer_op if the variable
      # has already been initialized elsewhere.
      self.evaluate(v.assign(1.0))
      self.assertEqual(1.0, v.initialized_value().eval())

    v_def.ClearField("initial_value_name")
    with ops.Graph().as_default(), self.cached_session():
      # Restoring a legacy VariableDef proto that does not have
      # initial_value_name set should still work.
      v = resource_variable_ops.ResourceVariable(variable_def=v_def)
      # We should also be able to re-export the variable to a new meta graph.
      self.assertProtoEquals(v_def, v.to_proto())
      # But attempts to use initialized_value will result in errors.
      with self.assertRaises(ValueError):
        self.evaluate(v.initialized_value())

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

  @test_util.run_in_graph_and_eager_modes
  def testSparseRead(self):
    init_value = np.reshape(np.arange(np.power(4, 3)), (4, 4, 4))
    v = resource_variable_ops.ResourceVariable(
        constant_op.constant(init_value, dtype=dtypes.int32), name="var3")
    self.evaluate(variables.global_variables_initializer())

    value = self.evaluate(v.sparse_read([0, 3, 1, 2]))
    self.assertAllEqual(init_value[[0, 3, 1, 2], ...], value)

  @test_util.run_in_graph_and_eager_modes
  def testGatherNd(self):
    init_value = np.reshape(np.arange(np.power(4, 3)), (4, 4, 4))
    v = resource_variable_ops.ResourceVariable(
        constant_op.constant(init_value, dtype=dtypes.int32), name="var3")
    self.evaluate(variables.global_variables_initializer())

    value_op = v.gather_nd([[0, 0], [1, 2], [3, 3]])
    self.assertAllEqual([3, 4], value_op.shape)
    value = self.evaluate(value_op)
    self.assertAllEqual([[0, 1, 2, 3], [24, 25, 26, 27], [60, 61, 62, 63]],
                        value)

    value_op = v.gather_nd([[0, 0, 0], [1, 2, 3], [3, 3, 3]])
    self.assertAllEqual([3], value_op.shape)
    value = self.evaluate(value_op)
    self.assertAllEqual([0, 27, 63], value)

  @test_util.run_deprecated_v1
  def testToFromProto(self):
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(1.0)
      self.evaluate(variables.global_variables_initializer())

      w = resource_variable_ops.ResourceVariable.from_proto(v.to_proto())
      self.assertEqual(2, math_ops.add(w, 1).eval())

      self.assertEqual(v._handle, w._handle)
      self.assertEqual(v._graph_element, w._graph_element)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
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

  @test_util.run_deprecated_v1
  def testAssignDifferentShapes(self):
    with self.cached_session() as sess, variable_scope.variable_scope(
        "foo", use_resource=True):
      var = variable_scope.get_variable("x", shape=[1, 1], dtype=dtypes.float32)
      placeholder = array_ops.placeholder(dtypes.float32)
      assign = var.assign(placeholder)
      sess.run(
          [assign],
          feed_dict={placeholder: np.zeros(shape=[2, 2], dtype=np.float32)})

  def testAssignDifferentShapesEagerNotAllowed(self):
    with context.eager_mode():
      with variable_scope.variable_scope("foo"):
        var = variable_scope.get_variable("x", shape=[1, 1],
                                          dtype=dtypes.float32)
        with self.assertRaisesRegexp(ValueError,
                                     "Shapes.*and.*are incompatible"):
          assign = var.assign(np.zeros(shape=[2, 2]))
          self.evaluate(assign)

  @test_util.disable_xla("XLA doesn't allow changing shape at assignment, as "
                         "dictated by tf2xla/xla_resource.cc:SetTypeAndShape")
  @test_util.run_in_graph_and_eager_modes
  def testAssignDifferentShapesAllowed(self):
    var = resource_variable_ops.ResourceVariable(
        initial_value=np.zeros(shape=[1, 1]),
        shape=tensor_shape.TensorShape(None))
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(np.zeros(shape=[1, 1]), var.read_value())
    self.evaluate(var.assign(np.zeros(shape=[2, 2])))
    self.assertAllEqual(np.zeros(shape=[2, 2]), var.read_value())

  @test_util.run_in_graph_and_eager_modes
  def testAssignReturnsVariable(self):
    var = resource_variable_ops.ResourceVariable(1.)
    self.evaluate(variables.global_variables_initializer())
    assigned = var.assign(2.)
    self.assertIsInstance(assigned, resource_variable_ops.BaseResourceVariable)
    assigned = assigned.assign(3.)
    self.assertEqual(self.evaluate(assigned), 3.)
    self.assertEqual(self.evaluate(var), 3.)

    self.assertEqual(self.evaluate(var.assign_add(1.).assign_add(1.)), 5)
    self.assertEqual(self.evaluate(var.assign_sub(1.).assign_sub(1.)), 3)

    var = resource_variable_ops.ResourceVariable([1., 2.])
    self.evaluate(variables.global_variables_initializer())
    slices = ops.IndexedSlices(indices=[1], values=[2])
    def assert_eq(tensor, vals):
      self.assertAllEqual(self.evaluate(tensor), vals)
    assert_eq(var.scatter_add(slices).scatter_add(slices), [1., 6.])
    assert_eq(var.scatter_sub(slices).scatter_sub(slices), [1., 2.])
    slices2 = ops.IndexedSlices(indices=[0], values=[3])
    assert_eq(var.scatter_max(slices2).scatter_add(slices), [3., 4.])
    assert_eq(var.scatter_add(slices).scatter_min(slices), [3., 2.])
    assert_eq(var.scatter_mul(slices).scatter_mul(slices), [3., 8.])
    assert_eq(var.scatter_div(slices).scatter_div(slices), [3., 2.])
    assert_eq(
        var.scatter_nd_update([[1]], [4.]).scatter_nd_add([[0]], [2.])
        .scatter_nd_sub([[1]], [3]),
        [5., 1.])
    assert_eq(var, [5., 1.])

    batch_var = resource_variable_ops.ResourceVariable(array_ops.ones((2, 2)))
    self.evaluate(variables.global_variables_initializer())
    batch_slices1 = ops.IndexedSlices(indices=[[1], [0]], values=[[2], [2]])
    batch_slices2 = ops.IndexedSlices(indices=[[1], [1]], values=[[3], [3]])
    assert_eq(
        batch_var.batch_scatter_update(batch_slices1)
        .batch_scatter_update(batch_slices2),
        [[1, 3], [2, 3]])

  @test_util.run_in_graph_and_eager_modes
  def testInitValueWrongShape(self):
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"not compatible with"):
      var = resource_variable_ops.ResourceVariable(
          initial_value=np.zeros(shape=[3]),
          shape=[4])
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(var.read_value())

  @test_util.run_deprecated_v1
  def testDtypeAfterFromProto(self):
    v = resource_variable_ops.ResourceVariable(2.0)
    w = resource_variable_ops.ResourceVariable.from_proto(v.to_proto())
    self.assertIsInstance(w.dtype, dtypes.DType)
    self.assertEqual(v.dtype, w.dtype)

  # TODO(alive): get caching to work in eager mode.
  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testSharedName(self):
    with self.cached_session():
      v = resource_variable_ops.ResourceVariable(300.0, name="var4")
      self.evaluate(variables.global_variables_initializer())

      w = resource_variable_ops.var_handle_op(
          dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="var4",
          # Needed in Eager since we get a unique container name by default.
          container=ops.get_default_graph()._container)
      w_read = resource_variable_ops.read_variable_op(w, v.dtype.base_dtype)
      self.assertEqual(300.0, self.evaluate(w_read))

      x = resource_variable_ops.var_handle_op(
          dtype=v.dtype.base_dtype, shape=v.get_shape(), shared_name="var5",
          container=ops.get_default_graph()._container)
      with self.assertRaisesOpError(
          "(Resource .*/var5/.* does not exist|Read of uninitialized variable)"
      ):
        resource_variable_ops.read_variable_op(x, v.dtype.base_dtype).eval()

  @test_util.run_deprecated_v1
  def testSharedNameWithNamescope(self):
    with self.cached_session():
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

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.run_deprecated_v1
  def testSetInitialValue(self):
    with self.cached_session():
      # Initialize variable with a value different from the initial value passed
      # in the constructor.
      v = resource_variable_ops.ResourceVariable(2.0)
      v.initializer.run(feed_dict={v.initial_value: 3.0})
      self.assertEqual(3.0, v.value().eval())

  @test_util.run_v1_only("b/120545219")
  def testControlFlowInitialization(self):
    """Expects an error if an initializer is in a control-flow scope."""

    def cond(i, _):
      return i < 10

    def body(i, _):
      zero = array_ops.zeros([], dtype=dtypes.int32)
      v = resource_variable_ops.ResourceVariable(initial_value=zero)
      return (i + 1, v.read_value())

    with self.assertRaisesRegexp(ValueError, "initializer"):
      control_flow_ops.while_loop(cond, body, [0, 0])

  def testVariableEager(self):
    with context.eager_mode():
      init = array_ops.ones(shape=[10, 20, 35], dtype=dtypes.int32)
      constraint = lambda x: x
      with ops.name_scope("foo", skip_on_eager=False):
        v = resource_variable_ops.ResourceVariable(
            name="var7",
            initial_value=init,
            caching_device="cpu:0",
            constraint=constraint)
      # Test properties
      self.assertEqual(dtypes.int32, v.dtype)
      self.assertEqual("foo/var7:0", v.name)
      self.assertAllEqual([10, 20, 35], v.shape.as_list())
      self.assertIsInstance(v.handle, ops.EagerTensor)
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

  def testScatterSubStateOps(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="sub")
      state_ops.scatter_sub(v, [1], [3])
      self.assertAllEqual([1.0, -1.0], v.numpy())

  def testScatterUpdateVariant(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([
          list_ops.empty_tensor_list(
              element_dtype=dtypes.float32, element_shape=[])
      ])
      v.scatter_update(
          ops.IndexedSlices(
              list_ops.tensor_list_from_tensor([1., 2.], element_shape=[]), 0))
      self.assertAllEqual(
          list_ops.tensor_list_get_item(v[0], 0, element_dtype=dtypes.float32),
          1.)

  def testGroupDoesntForceRead(self):
    with ops.Graph().as_default():
      v = resource_variable_ops.ResourceVariable(1.0)
      assign = v.assign_add(1.0)
      g = control_flow_ops.group([assign])
      self.assertEqual(g.control_inputs[0].type, "AssignAddVariableOp")

  def testScatterNdAddStateOps(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(
          [1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.float32, name="add")
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, 13, 3, 14, 14, 6, 7, 20])
      state_ops.scatter_nd_add(v, indices, updates)
      self.assertAllClose(expected, v.numpy())

  @test_util.run_in_graph_and_eager_modes
  def testUnreadVariableInsideFunction(self):
    v = resource_variable_ops.ResourceVariable(1.0)

    @def_function.function
    def assign():
      v.assign(1.0)

    graph = assign.get_concrete_function().graph
    self.assertTrue(all(x.type != "ReadVariableOp"
                        for x in graph.get_operations()))

  def testScatterNdSubStateOps(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(
          [1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.float32, name="sub")
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, -9, 3, -6, -4, 6, 7, -4])
      state_ops.scatter_nd_sub(v, indices, updates)
      self.assertAllClose(expected, v.numpy())

  def testScatterUpdateCast(self):
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable([1.0, 2.0], name="update")
      state_ops.scatter_update(v, [1], [3])
      self.assertAllEqual([1.0, 3.0], v.numpy())

  @test_util.run_in_graph_and_eager_modes
  def testScatterUpdateInvalidArgs(self):
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3], name="update")
    # The exact error and message differ between graph construction (where the
    # error is realized during shape inference at graph construction time) and
    # eager execution (where the error is realized during kernel execution).
    with self.assertRaisesRegexp(Exception, r"shape.*2.*3"):
      state_ops.scatter_update(v, [0, 1], [0, 1, 2])

  @test_util.run_in_graph_and_eager_modes
  def testAssignIncompatibleShape(self):
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
    self.evaluate(v.initializer)
    pattern = re.compile("shapes must be equal", re.IGNORECASE)
    with self.assertRaisesRegexp(Exception, pattern):
      self.evaluate(v.assign_add(1))

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testCopyToGraphUninitialized(self):
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
    copy_to_graph = ops.Graph()
    with copy_to_graph.as_default():  # Intentionally testing v1 behavior
      copied = resource_variable_ops.copy_to_graph_uninitialized(v)
      self.assertEqual(v.name, copied.name)
      self.assertIsNone(copied.initializer)

  def create_variant_shape_and_type_data(self):
    variant_shape_and_type_data = (
        cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData())
    variant_shape_and_type_data.is_set = True
    stored_shape = tensor_shape.TensorShape([None, 4]).as_proto()
    stored_dtype = dtypes.float32.as_datatype_enum
    # NOTE(ebrevdo): shape_and_type lacks append() in some versions of protobuf.
    variant_shape_and_type_data.shape_and_type.extend([
        cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
            shape=stored_shape, dtype=stored_dtype)])
    return variant_shape_and_type_data

  @def_function.function
  def create_constant_variant(self, value):
    value = constant_op.constant(
        tensor_pb2.TensorProto(
            dtype=dtypes.variant.as_datatype_enum,
            tensor_shape=tensor_shape.TensorShape([]).as_proto(),
            variant_val=[
                tensor_pb2.VariantTensorDataProto(
                    # Match registration in variant_op_registry.cc
                    type_name=b"int",
                    metadata=np.array(value, dtype=np.int32).tobytes())
            ]))
    return value

  # TODO(ebrevdo): Add run_in_graph_and_eager_modes once we can create
  # EagerTensor constants with TensorProto inputs.
  @test_util.run_in_graph_and_eager_modes()
  def testVariantInitializer(self):
    variant_shape_and_type_data = self.create_variant_shape_and_type_data()
    value = self.create_constant_variant(3)
    initializer = array_ops.fill([3], value)
    resource_variable_ops._set_handle_shapes_and_types(  # pylint: disable=protected-access
        initializer, variant_shape_and_type_data,
        graph_mode=not context.executing_eagerly())
    v = resource_variable_ops.ResourceVariable(initializer)
    read = array_ops.identity(v)
    read_variant_shape_and_type = (
        resource_variable_ops.get_eager_safe_handle_data(read))
    self.assertEqual(
        read_variant_shape_and_type, variant_shape_and_type_data)
    gather = v.sparse_read([0])
    gather_variant_shape_and_type = (
        resource_variable_ops.get_eager_safe_handle_data(gather))
    self.assertEqual(
        gather_variant_shape_and_type, variant_shape_and_type_data)
    # Make sure initializer runs.
    if not context.executing_eagerly():
      self.evaluate(v.initializer)
      self.evaluate(read.op)
      self.evaluate(gather.op)

  @parameterized.parameters([
      # batch_dims=0 (equivalent to tf.gather)
      dict(  # 2D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[2, 1], [0, 3]],
          expected=[[8, 7], [6, 9]]),
      dict(  # 3D indices
          batch_dims=0,
          params=[6, 7, 8, 9],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[9, 7], [8, 6]], [[6, 9], [8, 8]]]),
      dict(  # 4D indices
          batch_dims=0,
          params=[8, 9],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[8, 9], [9, 8]], [[8, 8], [9, 9]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),

      # batch_dims=indices.shape.ndims - 1 (equivalent to
      # tf.compat.v1.batch_gather)
      dict(  # 2D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=2,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),
      dict(  # 2D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[2, 1], [0, 3]],
          expected=[[12, 11], [20, 23]]),
      dict(  # 3D indices (2 batch dims)
          batch_dims=2,
          params=[[[100, 101], [110, 111]], [[200, 201], [210, 211]]],
          indices=[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
          expected=[[[100, 101], [111, 110]], [[200, 200], [211, 211]]]),

      # 0 < batch_dims < indices.shape.ndims - 1
      dict(  # 3D indices (1 batch dim)
          batch_dims=1,
          params=[[10, 11, 12, 13], [20, 21, 22, 23]],
          indices=[[[3, 1], [2, 0]], [[0, 3], [2, 2]]],
          expected=[[[13, 11], [12, 10]], [[20, 23], [22, 22]]]),
      dict(  # 4D indices (1 batch dim)
          batch_dims=1,
          params=[[6, 7], [8, 9]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[6, 7], [7, 6]], [[6, 6], [7, 7]]],
                    [[[9, 9], [8, 8]], [[8, 9], [9, 8]]]]),
      dict(  # 4D indices (2 batch dims)
          batch_dims=2,
          params=[[[2, 3], [4, 5]], [[6, 7], [8, 9]]],
          indices=[[[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                   [[[1, 1], [0, 0]], [[0, 1], [1, 0]]]],
          expected=[[[[2, 3], [3, 2]], [[4, 4], [5, 5]]],
                    [[[7, 7], [6, 6]], [[8, 9], [9, 8]]]]),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testGatherWithBatchDims(self, params, indices, batch_dims, expected):
    var = resource_variable_ops.ResourceVariable(params, name="var0")
    with ops.control_dependencies([var.initializer]):
      result = resource_variable_ops.resource_gather(
          var.handle, indices, dtype=var.dtype, batch_dims=batch_dims)
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=0,
          output_shape=[2, 3, 8, 9, 10, 3, 4, 5, 6, 7]
          # = indices.shape + params.shape[1:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=1,
          output_shape=[2, 3, 8, 9, 10, 4, 5, 6, 7]
          # = params.shape[:1] + indices.shape[1:] + params.shape[2:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 8, 9, 10],
          batch_dims=2,
          output_shape=[2, 3, 8, 9, 10, 5, 6, 7]
          # = params.shape[:2] + indices.shape[2:] + params.shape[3:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 4, 9, 10],
          batch_dims=3,
          output_shape=[2, 3, 4, 9, 10, 6, 7]
          # = params.shape[:3] + indices.shape[3:] + params.shape[4:]
      ),
      dict(
          params_shape=[2, 3, 4, 5, 6, 7],
          indices_shape=[2, 3, 4, 5, 10],
          batch_dims=4,
          output_shape=[2, 3, 4, 5, 10, 7]
          # = params.shape[:4] + indices.shape[4:] + params.shape[5:]
      ),
  ])
  @test_util.run_in_graph_and_eager_modes
  def testGatherWithBatchDimsMatchesTensor(self, params_shape, indices_shape,
                                           batch_dims, output_shape):
    """Checks that gather with batch_dims returns the correct shape."""
    # Generate a `params` tensor with the indicated shape.
    params_size = np.prod(params_shape)
    params = np.reshape(np.arange(params_size, dtype=np.int32), params_shape)

    # Generate an `indices` tensor with the indicated shape, where each index
    # is within the appropriate range.
    indices_size = np.prod(indices_shape)
    indices = np.reshape(np.arange(indices_size, dtype=np.int32), indices_shape)
    indices = indices % params_shape[batch_dims]

    var = resource_variable_ops.ResourceVariable(params, name="var0")
    with ops.control_dependencies([var.initializer]):
      expected = array_ops.gather(
          var.read_value(), indices, batch_dims=batch_dims)
      result = resource_variable_ops.resource_gather(
          var.handle, indices, dtype=var.dtype, batch_dims=batch_dims)

    self.assertAllEqual(output_shape, result.shape.as_list())
    self.assertAllEqual(expected, result)

  @parameterized.parameters([
      dict(dtype=dtypes.bool),
      dict(dtype=dtypes.int64),
      dict(dtype=dtypes.half),
      dict(dtype=dtypes.float32),
      dict(dtype=dtypes.double),
  ])
  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testGatherWithDTypes(self, dtype):
    if dtype == dtypes.bool:
      params = constant_op.constant([False, True, False, True])
      expected = constant_op.constant([[False, True], [False, True]])
    else:
      params = constant_op.constant([6, 7, 8, 9], dtype=dtype)
      expected = constant_op.constant([[8, 7], [6, 9]], dtype=dtype)
    indices = constant_op.constant([[2, 1], [0, 3]])
    var = resource_variable_ops.ResourceVariable(params, name="var0")
    with ops.control_dependencies([var.initializer]):
      result = resource_variable_ops.resource_gather(
          var.handle, indices, dtype=dtype)
    self.assertAllEqual(expected, result)


class PerReplicaResourceHandleTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(PerReplicaResourceHandleTest, self).setUp()
    cpus = config.list_physical_devices("CPU")
    # Set 2 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
    ])

  def testAllowedDevices(self):
    device0 = "/job:localhost/replica:0/task:0/device:CPU:0"
    device1 = "/job:localhost/replica:0/task:0/device:CPU:1"
    value0 = 1
    value1 = 2
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[], allowed_devices=[device0, device1])
      with ops.device(device0):
        assign0 = resource_variable_ops.assign_variable_op(handle, value0)
      with ops.device(device1):
        assign1 = resource_variable_ops.assign_variable_op(handle, value1)
      with ops.control_dependencies([assign0, assign1]):
        with ops.device(device0):
          read0 = resource_variable_ops.read_variable_op(
              handle, dtype=dtypes.int32)
        with ops.device(device1):
          read1 = resource_variable_ops.read_variable_op(
              handle, dtype=dtypes.int32)

      self.assertAllEqual(value0, read0)
      self.assertAllEqual(value1, read1)


if __name__ == "__main__":
  test.main()
