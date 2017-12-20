# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.test_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import threading

import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class TestUtilTest(test_util.TensorFlowTestCase):

  def test_assert_ops_in_graph(self):
    with self.test_session():
      constant_op.constant(["hello", "taffy"], name="hello")
      test_util.assert_ops_in_graph({"hello": "Const"}, ops.get_default_graph())

    self.assertRaises(ValueError, test_util.assert_ops_in_graph,
                      {"bye": "Const"}, ops.get_default_graph())

    self.assertRaises(ValueError, test_util.assert_ops_in_graph,
                      {"hello": "Variable"}, ops.get_default_graph())

  def test_assert_equal_graph_def(self):
    with ops.Graph().as_default() as g:
      def_empty = g.as_graph_def()
      constant_op.constant(5, name="five")
      constant_op.constant(7, name="seven")
      def_57 = g.as_graph_def()
    with ops.Graph().as_default() as g:
      constant_op.constant(7, name="seven")
      constant_op.constant(5, name="five")
      def_75 = g.as_graph_def()
    # Comparing strings is order dependent
    self.assertNotEqual(str(def_57), str(def_75))
    # assert_equal_graph_def doesn't care about order
    test_util.assert_equal_graph_def(def_57, def_75)
    # Compare two unequal graphs
    with self.assertRaisesRegexp(AssertionError,
                                 r"^Found unexpected node 'seven"):
      test_util.assert_equal_graph_def(def_57, def_empty)

  def testIsGoogleCudaEnabled(self):
    # The test doesn't assert anything. It ensures the py wrapper
    # function is generated correctly.
    if test_util.IsGoogleCudaEnabled():
      print("GoogleCuda is enabled")
    else:
      print("GoogleCuda is disabled")

  def testAssertProtoEqualsStr(self):

    graph_str = "node { name: 'w1' op: 'params' }"
    graph_def = graph_pb2.GraphDef()
    text_format.Merge(graph_str, graph_def)

    # test string based comparison
    self.assertProtoEquals(graph_str, graph_def)

    # test original comparison
    self.assertProtoEquals(graph_def, graph_def)

  def testAssertProtoEqualsAny(self):
    # Test assertProtoEquals with a protobuf.Any field.
    meta_graph_def_str = """
    meta_info_def {
      meta_graph_version: "outer"
      any_info {
        [type.googleapis.com/tensorflow.MetaGraphDef] {
          meta_info_def {
            meta_graph_version: "inner"
          }
        }
      }
    }
    """
    meta_graph_def_outer = meta_graph_pb2.MetaGraphDef()
    meta_graph_def_outer.meta_info_def.meta_graph_version = "outer"
    meta_graph_def_inner = meta_graph_pb2.MetaGraphDef()
    meta_graph_def_inner.meta_info_def.meta_graph_version = "inner"
    meta_graph_def_outer.meta_info_def.any_info.Pack(meta_graph_def_inner)
    self.assertProtoEquals(meta_graph_def_str, meta_graph_def_outer)
    self.assertProtoEquals(meta_graph_def_outer, meta_graph_def_outer)

    # Check if the assertion failure message contains the content of
    # the inner proto.
    with self.assertRaisesRegexp(AssertionError,
                                 r'meta_graph_version: "inner"'):
      self.assertProtoEquals("", meta_graph_def_outer)

  def testNDArrayNear(self):
    a1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a3 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    self.assertTrue(self._NDArrayNear(a1, a2, 1e-5))
    self.assertFalse(self._NDArrayNear(a1, a3, 1e-5))

  def testCheckedThreadSucceeds(self):

    def noop(ev):
      ev.set()

    event_arg = threading.Event()

    self.assertFalse(event_arg.is_set())
    t = self.checkedThread(target=noop, args=(event_arg,))
    t.start()
    t.join()
    self.assertTrue(event_arg.is_set())

  def testCheckedThreadFails(self):

    def err_func():
      return 1 // 0

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("integer division or modulo by zero" in str(fe.exception))

  def testCheckedThreadWithWrongAssertionFails(self):
    x = 37

    def err_func():
      self.assertTrue(x < 10)

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("False is not true" in str(fe.exception))

  def testMultipleThreadsWithOneFailure(self):

    def err_func(i):
      self.assertTrue(i != 7)

    threads = [
        self.checkedThread(
            target=err_func, args=(i,)) for i in range(10)
    ]
    for t in threads:
      t.start()
    for i, t in enumerate(threads):
      if i == 7:
        with self.assertRaises(self.failureException):
          t.join()
      else:
        t.join()

  def _WeMustGoDeeper(self, msg):
    with self.assertRaisesOpError(msg):
      with ops.Graph().as_default():
        node_def = ops._NodeDef("op_type", "name")
        node_def_orig = ops._NodeDef("op_type_orig", "orig")
        op_orig = ops.Operation(node_def_orig, ops.get_default_graph())
        op = ops.Operation(node_def, ops.get_default_graph(),
                           original_op=op_orig)
        raise errors.UnauthenticatedError(node_def, op, "true_err")

  def testAssertRaisesOpErrorDoesNotPassMessageDueToLeakedStack(self):
    with self.assertRaises(AssertionError):
      self._WeMustGoDeeper("this_is_not_the_error_you_are_looking_for")

    self._WeMustGoDeeper("true_err")
    self._WeMustGoDeeper("name")
    self._WeMustGoDeeper("orig")

  def testAllCloseScalars(self):
    self.assertAllClose(7, 7 + 1e-8)
    with self.assertRaisesRegexp(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(7, 7 + 1e-5)

  def testAllCloseDictToNonDict(self):
    with self.assertRaisesRegexp(ValueError, r"Can't compare dict to non-dict"):
      self.assertAllClose(1, {"a": 1})
    with self.assertRaisesRegexp(ValueError, r"Can't compare dict to non-dict"):
      self.assertAllClose({"a": 1}, 1)

  def testAllCloseDicts(self):
    a = 7
    b = (2., 3.)
    c = np.ones((3, 2, 4)) * 7.
    expected = {"a": a, "b": b, "c": c}

    # Identity.
    self.assertAllClose(expected, expected)
    self.assertAllClose(expected, dict(expected))

    # With each item removed.
    for k in expected:
      actual = dict(expected)
      del actual[k]
      with self.assertRaisesRegexp(AssertionError, r"mismatched keys"):
        self.assertAllClose(expected, actual)

    # With each item changed.
    with self.assertRaisesRegexp(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(expected, {"a": a + 1e-5, "b": b, "c": c})
    with self.assertRaisesRegexp(AssertionError, r"Shape mismatch"):
      self.assertAllClose(expected, {"a": a, "b": b + (4.,), "c": c})
    c_copy = np.array(c)
    c_copy[1, 1, 1] += 1e-5
    with self.assertRaisesRegexp(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(expected, {"a": a, "b": b, "c": c_copy})

  def testAllCloseNestedDicts(self):
    a = {"a": 1, "b": 2, "nested": {"d": 3, "e": 4}}
    with self.assertRaisesRegexp(
        TypeError,
        r"inputs could not be safely coerced to any supported types"):
      self.assertAllClose(a, a)

  def testArrayNear(self):
    a = [1, 2]
    b = [1, 2, 5]
    with self.assertRaises(AssertionError):
      self.assertArrayNear(a, b, 0.001)
    a = [1, 2]
    b = [[1, 2], [3, 4]]
    with self.assertRaises(TypeError):
      self.assertArrayNear(a, b, 0.001)
    a = [1, 2]
    b = [1, 2]
    self.assertArrayNear(a, b, 0.001)

  def testForceGPU(self):
    with self.assertRaises(errors.InvalidArgumentError):
      with self.test_session(force_gpu=True):
        # this relies on us not having a GPU implementation for assert, which
        # seems sensible
        x = constant_op.constant(True)
        y = [15]
        control_flow_ops.Assert(x, y).run()

  def testAssertAllCloseAccordingToType(self):
    # test float64
    self.assertAllCloseAccordingToType(
        np.asarray([1e-8], dtype=np.float64),
        np.asarray([2e-8], dtype=np.float64),
        rtol=1e-8, atol=1e-8
    )

    with (self.assertRaises(AssertionError)):
      self.assertAllCloseAccordingToType(
          np.asarray([1e-7], dtype=np.float64),
          np.asarray([2e-7], dtype=np.float64),
          rtol=1e-8, atol=1e-8
      )

    # test float32
    self.assertAllCloseAccordingToType(
        np.asarray([1e-7], dtype=np.float32),
        np.asarray([2e-7], dtype=np.float32),
        rtol=1e-8, atol=1e-8,
        float_rtol=1e-7, float_atol=1e-7
    )

    with (self.assertRaises(AssertionError)):
      self.assertAllCloseAccordingToType(
          np.asarray([1e-6], dtype=np.float32),
          np.asarray([2e-6], dtype=np.float32),
          rtol=1e-8, atol=1e-8,
          float_rtol=1e-7, float_atol=1e-7
      )

    # test float16
    self.assertAllCloseAccordingToType(
        np.asarray([1e-4], dtype=np.float16),
        np.asarray([2e-4], dtype=np.float16),
        rtol=1e-8, atol=1e-8,
        float_rtol=1e-7, float_atol=1e-7,
        half_rtol=1e-4, half_atol=1e-4
    )

    with (self.assertRaises(AssertionError)):
      self.assertAllCloseAccordingToType(
          np.asarray([1e-3], dtype=np.float16),
          np.asarray([2e-3], dtype=np.float16),
          rtol=1e-8, atol=1e-8,
          float_rtol=1e-7, float_atol=1e-7,
          half_rtol=1e-4, half_atol=1e-4
      )

  def testRandomSeed(self):
    a = random.randint(1, 1000)
    a_np_rand = np.random.rand(1)
    with self.test_session():
      a_rand = random_ops.random_normal([1]).eval()
    # ensure that randomness in multiple testCases is deterministic.
    self.setUp()
    b = random.randint(1, 1000)
    b_np_rand = np.random.rand(1)
    with self.test_session():
      b_rand = random_ops.random_normal([1]).eval()
    self.assertEqual(a, b)
    self.assertEqual(a_np_rand, b_np_rand)
    self.assertEqual(a_rand, b_rand)

  @test_util.run_in_graph_and_eager_modes()
  def test_callable_evaluate(self):
    def model():
      return resource_variable_ops.ResourceVariable(
          name="same_name",
          initial_value=1) + 1
    with context.eager_mode():
      self.assertEqual(2, self.evaluate(model))

  @test_util.run_in_graph_and_eager_modes()
  def test_nested_tensors_evaluate(self):
    expected = {"a": 1, "b": 2, "nested": {"d": 3, "e": 4}}
    nested = {"a": constant_op.constant(1),
              "b": constant_op.constant(2),
              "nested": {"d": constant_op.constant(3),
                         "e": constant_op.constant(4)}}

    self.assertEqual(expected, self.evaluate(nested))


class GarbageCollectionTest(test_util.TensorFlowTestCase):

  def test_no_reference_cycle_decorator(self):

    class ReferenceCycleTest(object):

      def __init__(inner_self):  # pylint: disable=no-self-argument
        inner_self.assertEqual = self.assertEqual  # pylint: disable=invalid-name

      @test_util.assert_no_garbage_created
      def test_has_cycle(self):
        a = []
        a.append(a)

      @test_util.assert_no_garbage_created
      def test_has_no_cycle(self):
        pass

    with self.assertRaises(AssertionError):
      ReferenceCycleTest().test_has_cycle()

    ReferenceCycleTest().test_has_no_cycle()

  def test_no_leaked_tensor_decorator(self):

    class LeakedTensorTest(object):

      def __init__(inner_self):  # pylint: disable=no-self-argument
        inner_self.assertEqual = self.assertEqual  # pylint: disable=invalid-name

      @test_util.assert_no_new_tensors
      def test_has_leak(self):
        self.a = constant_op.constant([3.])

      @test_util.assert_no_new_tensors
      def test_has_no_leak(self):
        constant_op.constant([3.])

    with self.assertRaisesRegexp(AssertionError, "Tensors not deallocated"):
      LeakedTensorTest().test_has_leak()

    LeakedTensorTest().test_has_no_leak()


@test_util.with_c_api
class IsolationTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_variable_reuse_exception(self):
    with test_util.IsolateTest(), session.Session():
      first_container_variable = resource_variable_ops.ResourceVariable(
          name="first_container_variable",
          initial_value=1)
      if context.in_graph_mode():
        self.evaluate([variables.global_variables_initializer()])
    with test_util.IsolateTest():
      if context.in_graph_mode():
        with self.assertRaises(RuntimeError):
          self.evaluate(first_container_variable.read_value())
      else:
        with self.assertRaises(ValueError):
          first_container_variable.read_value()

  @test_util.run_in_graph_and_eager_modes()
  def test_variable_reuse_exception_nested(self):
    with test_util.IsolateTest(), session.Session():
      first_container_variable = resource_variable_ops.ResourceVariable(
          name="first_container_variable",
          initial_value=1)
      if context.in_graph_mode():
        self.evaluate([variables.global_variables_initializer()])
      with test_util.IsolateTest(), session.Session():
        if context.in_graph_mode():
          with self.assertRaises(RuntimeError):
            self.evaluate(first_container_variable.read_value())
        else:
          with self.assertRaises(ValueError):
            first_container_variable.read_value()

  @test_util.run_in_graph_and_eager_modes()
  def test_no_sharing(self):
    with test_util.IsolateTest(), session.Session():
      first_container_variable = resource_variable_ops.ResourceVariable(
          name="same_name",
          initial_value=1)
      if context.in_graph_mode():
        self.evaluate([variables.global_variables_initializer()])
      with test_util.IsolateTest(), session.Session():
        second_container_variable = resource_variable_ops.ResourceVariable(
            name="same_name",
            initial_value=2)
        if context.in_graph_mode():
          self.evaluate([variables.global_variables_initializer()])
        self.assertEqual(
            2, self.evaluate(second_container_variable.read_value()))
      self.assertEqual(1, self.evaluate(first_container_variable.read_value()))

  def test_graph_mode_isolation(self):
    with context.graph_mode():
      # Even if we've (accidentally) called IsolateTest in Graph mode, it should
      # provide Eager isolation.
      with test_util.IsolateTest():
        with context.eager_mode():
          first_container_variable = resource_variable_ops.ResourceVariable(
              name="first_container_variable",
              initial_value=1)
      with context.eager_mode():
        with self.assertRaises(ValueError):
          first_container_variable.read_value()

if __name__ == "__main__":
  googletest.main()
