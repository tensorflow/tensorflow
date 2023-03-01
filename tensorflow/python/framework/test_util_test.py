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

import collections
import copy
import math
import random
import threading
import time
import unittest
import weakref

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util.protobuf import compare_test_pb2


class TestUtilTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def test_assert_ops_in_graph(self):
    with ops.Graph().as_default():
      constant_op.constant(["hello", "taffy"], name="hello")
      test_util.assert_ops_in_graph({"hello": "Const"}, ops.get_default_graph())

      self.assertRaises(ValueError, test_util.assert_ops_in_graph,
                        {"bye": "Const"}, ops.get_default_graph())

      self.assertRaises(ValueError, test_util.assert_ops_in_graph,
                        {"hello": "Variable"}, ops.get_default_graph())

  @test_util.run_deprecated_v1
  def test_session_functions(self):
    with self.test_session() as sess:
      sess_ref = weakref.ref(sess)
      with self.cached_session(graph=None, config=None) as sess2:
        # We make sure that sess2 is sess.
        assert sess2 is sess
        # We make sure we raise an exception if we use cached_session with
        # different values.
        with self.assertRaises(ValueError):
          with self.cached_session(graph=ops.Graph()) as sess2:
            pass
        with self.assertRaises(ValueError):
          with self.cached_session(force_gpu=True) as sess2:
            pass
    # We make sure that test_session will cache the session even after the
    # with scope.
    assert not sess_ref()._closed
    with self.session() as unique_sess:
      unique_sess_ref = weakref.ref(unique_sess)
      with self.session() as sess2:
        assert sess2 is not unique_sess
    # We make sure the session is closed when we leave the with statement.
    assert unique_sess_ref()._closed

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
    with self.assertRaisesRegex(AssertionError,
                                r"^Found unexpected node '{{node seven}}"):
      test_util.assert_equal_graph_def(def_57, def_empty)

  def test_assert_equal_graph_def_hash_table(self):
    def get_graph_def():
      with ops.Graph().as_default() as g:
        x = constant_op.constant([2, 9], name="x")
        keys = constant_op.constant([1, 2], name="keys")
        values = constant_op.constant([3, 4], name="values")
        default = constant_op.constant(-1, name="default")
        table = lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(keys, values), default)
        _ = table.lookup(x)
      return g.as_graph_def()
    def_1 = get_graph_def()
    def_2 = get_graph_def()
    # The unique shared_name of each table makes the graph unequal.
    with self.assertRaisesRegex(AssertionError, "hash_table_"):
      test_util.assert_equal_graph_def(def_1, def_2,
                                       hash_table_shared_name=False)
    # That can be ignored. (NOTE: modifies GraphDefs in-place.)
    test_util.assert_equal_graph_def(def_1, def_2,
                                     hash_table_shared_name=True)

  def testIsGoogleCudaEnabled(self):
    # The test doesn't assert anything. It ensures the py wrapper
    # function is generated correctly.
    if test_util.IsGoogleCudaEnabled():
      print("GoogleCuda is enabled")
    else:
      print("GoogleCuda is disabled")

  def testIsMklEnabled(self):
    # This test doesn't assert anything.
    # It ensures the py wrapper function is generated correctly.
    if test_util.IsMklEnabled():
      print("MKL is enabled")
    else:
      print("MKL is disabled")

  @test_util.disable_asan("Skip test if ASAN is enabled.")
  def testDisableAsan(self):
    self.assertFalse(pywrap_sanitizers.is_asan_enabled())

  @test_util.disable_msan("Skip test if MSAN is enabled.")
  def testDisableMsan(self):
    self.assertFalse(pywrap_sanitizers.is_msan_enabled())

  @test_util.disable_tsan("Skip test if TSAN is enabled.")
  def testDisableTsan(self):
    self.assertFalse(pywrap_sanitizers.is_tsan_enabled())

  @test_util.disable_ubsan("Skip test if UBSAN is enabled.")
  def testDisableUbsan(self):
    self.assertFalse(pywrap_sanitizers.is_ubsan_enabled())

  @test_util.run_in_graph_and_eager_modes
  def testAssertProtoEqualsStr(self):

    graph_str = "node { name: 'w1' op: 'params' }"
    graph_def = graph_pb2.GraphDef()
    text_format.Parse(graph_str, graph_def)

    # test string based comparison
    self.assertProtoEquals(graph_str, graph_def)

    # test original comparison
    self.assertProtoEquals(graph_def, graph_def)

  @test_util.run_in_graph_and_eager_modes
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
    with self.assertRaisesRegex(AssertionError, r'meta_graph_version: "inner"'):
      self.assertProtoEquals("", meta_graph_def_outer)

  def test_float_relative_tolerance(self):
    pb1 = compare_test_pb2.Floats(float_=65061.0420, double_=164107.8938)
    pb2 = compare_test_pb2.Floats(float_=65061.0322, double_=164107.9087)
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )

  def test_float_relative_tolerance_inf(self):
    pb1 = compare_test_pb2.Floats(float_=float("inf"))
    pb2 = compare_test_pb2.Floats(float_=float("inf"))
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  def test_float_relative_tolerance_denormal(self):
    pb1 = compare_test_pb2.Floats(float_=math.ulp(0.0))
    pb2 = compare_test_pb2.Floats(float_=math.ulp(0.0))
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  def test_repeated_float_relative_tolerance(self):
    pb1 = compare_test_pb2.RepeatedFloats(
        float_=(x for x in [1.0, 2.0, 65061.042])
    )
    pb2 = compare_test_pb2.RepeatedFloats(
        float_=(x for x in [1.0, 2.0, 65061.0322])
    )
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  def test_nested_float_relative_tolerance(self):
    pb1 = compare_test_pb2.NestedFloats()
    pb2 = compare_test_pb2.NestedFloats()
    pb1.floats.float_ = 65061.0420
    pb2.floats.float_ = 65061.0322
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  def test_map_float_relative_tolerance(self):
    pb1 = compare_test_pb2.MapFloats()
    pb2 = compare_test_pb2.MapFloats()
    pb1.int_to_floats[1].float_ = 65061.0420
    pb2.int_to_floats[1].float_ = 65061.0322
    pb1.int_to_floats[1].double_ = 164107.8938
    pb2.int_to_floats[1].double_ = 164107.9087
    self.assertRaises(
        AssertionError,
        self.assertProtoEquals,
        pb1,
        pb2,
        relative_tolerance=1e-7,
    )
    self.assertProtoEquals(pb1, pb2, relative_tolerance=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testNDArrayNear(self):
    a1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a3 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    self.assertTrue(self._NDArrayNear(a1, a2, 1e-5))
    self.assertFalse(self._NDArrayNear(a1, a3, 1e-5))

  @test_util.run_in_graph_and_eager_modes
  def testCheckedThreadSucceeds(self):

    def noop(ev):
      ev.set()

    event_arg = threading.Event()

    self.assertFalse(event_arg.is_set())
    t = self.checkedThread(target=noop, args=(event_arg,))
    t.start()
    t.join()
    self.assertTrue(event_arg.is_set())

  @test_util.run_in_graph_and_eager_modes
  def testCheckedThreadFails(self):

    def err_func():
      return 1 // 0

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("integer division or modulo by zero" in str(fe.exception))

  @test_util.run_in_graph_and_eager_modes
  def testCheckedThreadWithWrongAssertionFails(self):
    x = 37

    def err_func():
      self.assertTrue(x < 10)

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("False is not true" in str(fe.exception))

  @test_util.run_in_graph_and_eager_modes
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
        node_def = ops._NodeDef("IntOutput", "name")
        node_def_orig = ops._NodeDef("IntOutput", "orig")
        op_orig = ops.Operation.from_node_def(
            node_def_orig, ops.get_default_graph()
        )
        op = ops.Operation.from_node_def(
            node_def, ops.get_default_graph(), original_op=op_orig
        )
        raise errors.UnauthenticatedError(node_def, op, "true_err")

  @test_util.run_in_graph_and_eager_modes
  def testAssertRaisesOpErrorDoesNotPassMessageDueToLeakedStack(self):
    with self.assertRaises(AssertionError):
      self._WeMustGoDeeper("this_is_not_the_error_you_are_looking_for")

    self._WeMustGoDeeper("true_err")
    self._WeMustGoDeeper("name")
    self._WeMustGoDeeper("orig")

  @parameterized.named_parameters(
      dict(testcase_name="tensors", ragged_tensors=False),
      dict(testcase_name="ragged_tensors", ragged_tensors=True))
  @test_util.run_in_graph_and_eager_modes
  def testAllCloseTensors(self, ragged_tensors: bool):
    a_raw_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    a = constant_op.constant(a_raw_data)
    b = math_ops.add(1, constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    if ragged_tensors:
      a = ragged_tensor.RaggedTensor.from_tensor(a)
      b = ragged_tensor.RaggedTensor.from_tensor(b)

    self.assertAllClose(a, b)
    self.assertAllClose(a, a_raw_data)

    a_dict = {"key": a}
    b_dict = {"key": b}
    self.assertAllClose(a_dict, b_dict)

    x_list = [a, b]
    y_list = [a_raw_data, b]
    self.assertAllClose(x_list, y_list)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseScalars(self):
    self.assertAllClose(7, 7 + 1e-8)
    with self.assertRaisesRegex(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(7, 7 + 1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseList(self):
    with self.assertRaisesRegex(AssertionError, r"not close dif"):
      self.assertAllClose([0], [1])

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseDictToNonDict(self):
    with self.assertRaisesRegex(ValueError, r"Can't compare dict to non-dict"):
      self.assertAllClose(1, {"a": 1})
    with self.assertRaisesRegex(ValueError, r"Can't compare dict to non-dict"):
      self.assertAllClose({"a": 1}, 1)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseNamedtuples(self):
    a = 7
    b = (2., 3.)
    c = np.ones((3, 2, 4)) * 7.
    expected = {"a": a, "b": b, "c": c}
    my_named_tuple = collections.namedtuple("MyNamedTuple", ["a", "b", "c"])

    # Identity.
    self.assertAllClose(expected, my_named_tuple(a=a, b=b, c=c))
    self.assertAllClose(
        my_named_tuple(a=a, b=b, c=c), my_named_tuple(a=a, b=b, c=c))

  @test_util.run_in_graph_and_eager_modes
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
      with self.assertRaisesRegex(AssertionError, r"mismatched keys"):
        self.assertAllClose(expected, actual)

    # With each item changed.
    with self.assertRaisesRegex(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(expected, {"a": a + 1e-5, "b": b, "c": c})
    with self.assertRaisesRegex(AssertionError, r"Shape mismatch"):
      self.assertAllClose(expected, {"a": a, "b": b + (4.,), "c": c})
    c_copy = np.array(c)
    c_copy[1, 1, 1] += 1e-5
    with self.assertRaisesRegex(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(expected, {"a": a, "b": b, "c": c_copy})

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseListOfNamedtuples(self):
    my_named_tuple = collections.namedtuple("MyNamedTuple", ["x", "y"])
    l1 = [
        my_named_tuple(x=np.array([[2.3, 2.5]]), y=np.array([[0.97, 0.96]])),
        my_named_tuple(x=np.array([[3.3, 3.5]]), y=np.array([[0.98, 0.99]]))
    ]
    l2 = [
        ([[2.3, 2.5]], [[0.97, 0.96]]),
        ([[3.3, 3.5]], [[0.98, 0.99]]),
    ]
    self.assertAllClose(l1, l2)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseNestedStructure(self):
    a = {"x": np.ones((3, 2, 4)) * 7, "y": (2, [{"nested": {"m": 3, "n": 4}}])}
    self.assertAllClose(a, a)

    b = copy.deepcopy(a)
    self.assertAllClose(a, b)

    # Test mismatched values
    b["y"][1][0]["nested"]["n"] = 4.2
    with self.assertRaisesRegex(AssertionError,
                                r"\[y\]\[1\]\[0\]\[nested\]\[n\]"):
      self.assertAllClose(a, b)

  @test_util.run_in_graph_and_eager_modes
  def testAssertDictEqual(self):
    a = 7
    b = (2., 3.)
    c = np.ones((3, 2, 4)) * 7.
    d = "testing123"
    expected = {"a": a, "b": b, "c": c, "d": d}
    actual = {"a": a, "b": b, "c": constant_op.constant(c), "d": d}

    self.assertDictEqual(expected, expected)
    self.assertDictEqual(expected, actual)

  @test_util.run_in_graph_and_eager_modes
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

  @test_util.skip_if(True)  # b/117665998
  def testForceGPU(self):
    with self.assertRaises(errors.InvalidArgumentError):
      with self.test_session(force_gpu=True):
        # this relies on us not having a GPU implementation for assert, which
        # seems sensible
        x = constant_op.constant(True)
        y = [15]
        control_flow_ops.Assert(x, y).run()

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllCloseAccordingToType(self):
    # test plain int
    self.assertAllCloseAccordingToType(1, 1, rtol=1e-8, atol=1e-8)

    # test float64
    self.assertAllCloseAccordingToType(
        np.asarray([1e-8], dtype=np.float64),
        np.asarray([2e-8], dtype=np.float64),
        rtol=1e-8, atol=1e-8
    )

    self.assertAllCloseAccordingToType(
        constant_op.constant([1e-8], dtype=dtypes.float64),
        constant_op.constant([2e-8], dtype=dtypes.float64),
        rtol=1e-8,
        atol=1e-8)

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

    self.assertAllCloseAccordingToType(
        constant_op.constant([1e-7], dtype=dtypes.float32),
        constant_op.constant([2e-7], dtype=dtypes.float32),
        rtol=1e-8,
        atol=1e-8,
        float_rtol=1e-7,
        float_atol=1e-7)

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

    self.assertAllCloseAccordingToType(
        constant_op.constant([1e-4], dtype=dtypes.float16),
        constant_op.constant([2e-4], dtype=dtypes.float16),
        rtol=1e-8,
        atol=1e-8,
        float_rtol=1e-7,
        float_atol=1e-7,
        half_rtol=1e-4,
        half_atol=1e-4)

    with (self.assertRaises(AssertionError)):
      self.assertAllCloseAccordingToType(
          np.asarray([1e-3], dtype=np.float16),
          np.asarray([2e-3], dtype=np.float16),
          rtol=1e-8, atol=1e-8,
          float_rtol=1e-7, float_atol=1e-7,
          half_rtol=1e-4, half_atol=1e-4
      )

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllEqual(self):
    i = variables.Variable([100] * 3, dtype=dtypes.int32, name="i")
    j = constant_op.constant([20] * 3, dtype=dtypes.int32, name="j")
    k = math_ops.add(i, j, name="k")

    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual([100] * 3, i)
    self.assertAllEqual([120] * 3, k)
    self.assertAllEqual([20] * 3, j)

    with self.assertRaisesRegex(AssertionError, r"not equal lhs"):
      self.assertAllEqual([0] * 3, k)

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllEqual(self):
    i = variables.Variable([100], dtype=dtypes.int32, name="i")
    j = constant_op.constant([20], dtype=dtypes.int32, name="j")
    k = math_ops.add(i, j, name="k")

    self.evaluate(variables.global_variables_initializer())
    self.assertNotAllEqual([100] * 3, i)
    self.assertNotAllEqual([120] * 3, k)
    self.assertNotAllEqual([20] * 3, j)

    with self.assertRaisesRegex(
        AssertionError, r"two values are equal at all elements. $"):
      self.assertNotAllEqual([120], k)

    with self.assertRaisesRegex(
        AssertionError, r"two values are equal at all elements.*extra message"):
      self.assertNotAllEqual([120], k, msg="extra message")

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllClose(self):
    # Test with arrays
    self.assertNotAllClose([0.1], [0.2])
    with self.assertRaises(AssertionError):
      self.assertNotAllClose([-1.0, 2.0], [-1.0, 2.0])

    # Test with tensors
    x = constant_op.constant([1.0, 1.0], name="x")
    y = math_ops.add(x, x)

    self.assertAllClose([2.0, 2.0], y)
    self.assertNotAllClose([0.9, 1.0], x)

    with self.assertRaises(AssertionError):
      self.assertNotAllClose([1.0, 1.0], x)

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllCloseRTol(self):
    # Test with arrays
    with self.assertRaises(AssertionError):
      self.assertNotAllClose([1.1, 2.1], [1.0, 2.0], rtol=0.2)

    # Test with tensors
    x = constant_op.constant([1.0, 1.0], name="x")
    y = math_ops.add(x, x)

    self.assertAllClose([2.0, 2.0], y)

    with self.assertRaises(AssertionError):
      self.assertNotAllClose([0.9, 1.0], x, rtol=0.2)

  @test_util.run_in_graph_and_eager_modes
  def testAssertNotAllCloseATol(self):
    # Test with arrays
    with self.assertRaises(AssertionError):
      self.assertNotAllClose([1.1, 2.1], [1.0, 2.0], atol=0.2)

    # Test with tensors
    x = constant_op.constant([1.0, 1.0], name="x")
    y = math_ops.add(x, x)

    self.assertAllClose([2.0, 2.0], y)

    with self.assertRaises(AssertionError):
      self.assertNotAllClose([0.9, 1.0], x, atol=0.2)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllGreaterLess(self):
    x = constant_op.constant([100.0, 110.0, 120.0], dtype=dtypes.float32)
    y = constant_op.constant([10.0] * 3, dtype=dtypes.float32)
    z = math_ops.add(x, y)

    self.assertAllClose([110.0, 120.0, 130.0], z)

    self.assertAllGreater(x, 95.0)
    self.assertAllLess(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllGreater(x, 105.0)
    with self.assertRaises(AssertionError):
      self.assertAllGreater(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllLess(x, 115.0)
    with self.assertRaises(AssertionError):
      self.assertAllLess(x, 95.0)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllGreaterLessEqual(self):
    x = constant_op.constant([100.0, 110.0, 120.0], dtype=dtypes.float32)
    y = constant_op.constant([10.0] * 3, dtype=dtypes.float32)
    z = math_ops.add(x, y)

    self.assertAllEqual([110.0, 120.0, 130.0], z)

    self.assertAllGreaterEqual(x, 95.0)
    self.assertAllLessEqual(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllGreaterEqual(x, 105.0)
    with self.assertRaises(AssertionError):
      self.assertAllGreaterEqual(x, 125.0)

    with self.assertRaises(AssertionError):
      self.assertAllLessEqual(x, 115.0)
    with self.assertRaises(AssertionError):
      self.assertAllLessEqual(x, 95.0)

  def testAssertAllInRangeWithNonNumericValuesFails(self):
    s1 = constant_op.constant("Hello, ", name="s1")
    c = constant_op.constant([1 + 2j, -3 + 5j], name="c")
    b = constant_op.constant([False, True], name="b")

    with self.assertRaises(AssertionError):
      self.assertAllInRange(s1, 0.0, 1.0)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(c, 0.0, 1.0)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(b, 0, 1)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRange(self):
    x = constant_op.constant([10.0, 15.0], name="x")
    self.assertAllInRange(x, 10, 15)

    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, 15, open_lower_bound=True)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, 15, open_upper_bound=True)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(
          x, 10, 15, open_lower_bound=True, open_upper_bound=True)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeScalar(self):
    x = constant_op.constant(10.0, name="x")
    nan = constant_op.constant(np.nan, name="nan")
    self.assertAllInRange(x, 5, 15)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(nan, 5, 15)

    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, 15, open_lower_bound=True)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 1, 2)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeErrorMessageEllipses(self):
    x_init = np.array([[10.0, 15.0]] * 12)
    x = constant_op.constant(x_init, name="x")
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 5, 10)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeDetectsNaNs(self):
    x = constant_op.constant(
        [[np.nan, 0.0], [np.nan, np.inf], [np.inf, np.nan]], name="x")
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 0.0, 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInRangeWithInfinities(self):
    x = constant_op.constant([10.0, np.inf], name="x")
    self.assertAllInRange(x, 10, np.inf)
    with self.assertRaises(AssertionError):
      self.assertAllInRange(x, 10, np.inf, open_upper_bound=True)

  @test_util.run_in_graph_and_eager_modes
  def testAssertAllInSet(self):
    b = constant_op.constant([True, False], name="b")
    x = constant_op.constant([13, 37], name="x")

    self.assertAllInSet(b, [False, True])
    self.assertAllInSet(b, (False, True))
    self.assertAllInSet(b, {False, True})
    self.assertAllInSet(x, [0, 13, 37, 42])
    self.assertAllInSet(x, (0, 13, 37, 42))
    self.assertAllInSet(x, {0, 13, 37, 42})

    with self.assertRaises(AssertionError):
      self.assertAllInSet(b, [False])
    with self.assertRaises(AssertionError):
      self.assertAllInSet(x, (42,))

  @test_util.run_in_graph_and_eager_modes
  def testAssertShapeEqualSameInputTypes(self):
    # Test with arrays
    array_a = np.random.rand(3, 1)
    array_b = np.random.rand(3, 1)
    array_c = np.random.rand(4, 2)

    self.assertShapeEqual(array_a, array_b)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(array_a, array_c)

    # Test with tensors
    tensor_x = random_ops.random_uniform((5, 2, 1))
    tensor_y = random_ops.random_uniform((5, 2, 1))
    tensor_z = random_ops.random_uniform((2, 4))

    self.assertShapeEqual(tensor_x, tensor_y)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(tensor_x, tensor_z)

  @test_util.run_in_graph_and_eager_modes
  def testAssertShapeEqualMixedInputTypes(self):

    # Test mixed multi-dimensional inputs
    array_input = np.random.rand(4, 3, 2)
    tensor_input = random_ops.random_uniform((4, 3, 2))
    tensor_input_2 = random_ops.random_uniform((10, 5))

    self.assertShapeEqual(array_input, tensor_input)
    self.assertShapeEqual(tensor_input, array_input)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(array_input, tensor_input_2)

    # Test with scalar inputs
    array_input = np.random.rand(1)
    tensor_input = random_ops.random_uniform((1,))
    tensor_input_2 = random_ops.random_uniform((3, 1))

    self.assertShapeEqual(array_input, tensor_input)
    self.assertShapeEqual(tensor_input, array_input)
    with self.assertRaises(AssertionError):
      self.assertShapeEqual(array_input, tensor_input_2)

  def testAssertShapeEqualDynamicShapes(self):

    array_a = np.random.rand(4)
    values = [1, 1, 2, 3, 4, 4]

    # Dynamic shape should be resolved in eager execution.
    with context.eager_mode():
      tensor_b = array_ops.unique(values)[0]
      self.assertShapeEqual(array_a, tensor_b)

    # Shape comparison should fail when a graph is traced but not evaluated.
    with context.graph_mode():
      tensor_c = array_ops.unique(values)[0]
      with self.assertRaises(AssertionError):
        self.assertShapeEqual(array_a, tensor_c)

  def testRandomSeed(self):
    # Call setUp again for WithCApi case (since it makes a new default graph
    # after setup).
    # TODO(skyewm): remove this when C API is permanently enabled.
    with context.eager_mode():
      self.setUp()
      a = random.randint(1, 1000)
      a_np_rand = np.random.rand(1)
      a_rand = random_ops.random_normal([1])
      # ensure that randomness in multiple testCases is deterministic.
      self.setUp()
      b = random.randint(1, 1000)
      b_np_rand = np.random.rand(1)
      b_rand = random_ops.random_normal([1])
      self.assertEqual(a, b)
      self.assertEqual(a_np_rand, b_np_rand)
      self.assertAllEqual(a_rand, b_rand)

  def testIndexedSlices(self):
    with context.eager_mode():
      self.evaluate(
          indexed_slices.IndexedSlices(
              constant_op.constant(1.0), constant_op.constant(0.0)))

  @test_util.run_in_graph_and_eager_modes
  def test_callable_evaluate(self):
    def model():
      return resource_variable_ops.ResourceVariable(
          name="same_name",
          initial_value=1) + 1
    with context.eager_mode():
      self.assertEqual(2, self.evaluate(model))

  @test_util.run_in_graph_and_eager_modes
  def test_nested_tensors_evaluate(self):
    expected = {"a": 1, "b": 2, "nested": {"d": 3, "e": 4}}
    nested = {"a": constant_op.constant(1),
              "b": constant_op.constant(2),
              "nested": {"d": constant_op.constant(3),
                         "e": constant_op.constant(4)}}

    self.assertEqual(expected, self.evaluate(nested))

  def test_run_in_graph_and_eager_modes(self):
    l = []
    def inc(self, with_brackets):
      del self  # self argument is required by run_in_graph_and_eager_modes.
      mode = "eager" if context.executing_eagerly() else "graph"
      with_brackets = "with_brackets" if with_brackets else "without_brackets"
      l.append((with_brackets, mode))

    f = test_util.run_in_graph_and_eager_modes(inc)
    f(self, with_brackets=False)
    f = test_util.run_in_graph_and_eager_modes()(inc)  # pylint: disable=assignment-from-no-return
    f(self, with_brackets=True)

    self.assertEqual(len(l), 4)
    self.assertEqual(set(l), {
        ("with_brackets", "graph"),
        ("with_brackets", "eager"),
        ("without_brackets", "graph"),
        ("without_brackets", "eager"),
    })

  def test_get_node_def_from_graph(self):
    graph_def = graph_pb2.GraphDef()
    node_foo = graph_def.node.add()
    node_foo.name = "foo"
    self.assertIs(test_util.get_node_def_from_graph("foo", graph_def), node_foo)
    self.assertIsNone(test_util.get_node_def_from_graph("bar", graph_def))

  def test_run_in_eager_and_graph_modes_test_class(self):
    msg = "`run_in_graph_and_eager_modes` only supports test methods.*"
    with self.assertRaisesRegex(ValueError, msg):

      @test_util.run_in_graph_and_eager_modes()
      class Foo(object):
        pass
      del Foo  # Make pylint unused happy.

  def test_run_in_eager_and_graph_modes_skip_graph_runs_eager(self):
    modes = []
    def _test(self):
      if not context.executing_eagerly():
        self.skipTest("Skipping in graph mode")
      modes.append("eager" if context.executing_eagerly() else "graph")
    test_util.run_in_graph_and_eager_modes(_test)(self)
    self.assertEqual(modes, ["eager"])

  def test_run_in_eager_and_graph_modes_skip_eager_runs_graph(self):
    modes = []
    def _test(self):
      if context.executing_eagerly():
        self.skipTest("Skipping in eager mode")
      modes.append("eager" if context.executing_eagerly() else "graph")
    test_util.run_in_graph_and_eager_modes(_test)(self)
    self.assertEqual(modes, ["graph"])

  def test_run_in_graph_and_eager_modes_setup_in_same_mode(self):
    modes = []
    mode_name = lambda: "eager" if context.executing_eagerly() else "graph"

    class ExampleTest(test_util.TensorFlowTestCase):

      def runTest(self):
        pass

      def setUp(self):
        modes.append("setup_" + mode_name())

      @test_util.run_in_graph_and_eager_modes
      def testBody(self):
        modes.append("run_" + mode_name())

    e = ExampleTest()
    e.setUp()
    e.testBody()

    self.assertEqual(modes[1:2], ["run_graph"])
    self.assertEqual(modes[2:], ["setup_eager", "run_eager"])

  @parameterized.named_parameters(dict(testcase_name="argument",
                                       arg=True))
  @test_util.run_in_graph_and_eager_modes
  def test_run_in_graph_and_eager_works_with_parameterized_keyword(self, arg):
    self.assertEqual(arg, True)

  @combinations.generate(combinations.combine(arg=True))
  @test_util.run_in_graph_and_eager_modes
  def test_run_in_graph_and_eager_works_with_combinations(self, arg):
    self.assertEqual(arg, True)

  def test_build_as_function_and_v1_graph(self):

    class GraphModeAndFunctionTest(parameterized.TestCase):

      def __init__(inner_self):  # pylint: disable=no-self-argument
        super(GraphModeAndFunctionTest, inner_self).__init__()
        inner_self.graph_mode_tested = False
        inner_self.inside_function_tested = False

      def runTest(self):
        del self

      @test_util.build_as_function_and_v1_graph
      def test_modes(inner_self):  # pylint: disable=no-self-argument
        if ops.inside_function():
          self.assertFalse(inner_self.inside_function_tested)
          inner_self.inside_function_tested = True
        else:
          self.assertFalse(inner_self.graph_mode_tested)
          inner_self.graph_mode_tested = True

    test_object = GraphModeAndFunctionTest()
    test_object.test_modes_v1_graph()
    test_object.test_modes_function()
    self.assertTrue(test_object.graph_mode_tested)
    self.assertTrue(test_object.inside_function_tested)

  @test_util.run_in_graph_and_eager_modes
  def test_consistent_random_seed_in_assert_all_equal(self):
    random_seed.set_seed(1066)
    index = random_ops.random_shuffle([0, 1, 2, 3, 4], seed=2021)
    # This failed when `a` and `b` were evaluated in separate sessions.
    self.assertAllEqual(index, index)

  def test_with_forward_compatibility_horizons(self):

    tested_codepaths = set()
    def some_function_with_forward_compat_behavior():
      if compat.forward_compatible(2050, 1, 1):
        tested_codepaths.add("future")
      else:
        tested_codepaths.add("present")

    @test_util.with_forward_compatibility_horizons(None, [2051, 1, 1])
    def some_test(self):
      del self  # unused
      some_function_with_forward_compat_behavior()

    some_test(None)
    self.assertEqual(tested_codepaths, set(["present", "future"]))


class SkipTestTest(test_util.TensorFlowTestCase):

  def _verify_test_in_set_up_or_tear_down(self):
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError,
                                   ["foo bar", "test message"]):
        raise ValueError("test message")
    try:
      with self.assertRaisesRegex(ValueError, "foo bar"):
        with test_util.skip_if_error(self, ValueError, "test message"):
          raise ValueError("foo bar")
    except unittest.SkipTest:
      raise RuntimeError("Test is not supposed to skip.")

  def setUp(self):
    super(SkipTestTest, self).setUp()
    self._verify_test_in_set_up_or_tear_down()

  def tearDown(self):
    super(SkipTestTest, self).tearDown()
    self._verify_test_in_set_up_or_tear_down()

  def test_skip_if_error_should_skip(self):
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError, "test message"):
        raise ValueError("test message")

  def test_skip_if_error_should_skip_with_list(self):
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError,
                                   ["foo bar", "test message"]):
        raise ValueError("test message")

  def test_skip_if_error_should_skip_without_expected_message(self):
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError):
        raise ValueError("test message")

  def test_skip_if_error_should_skip_without_error_message(self):
    with self.assertRaises(unittest.SkipTest):
      with test_util.skip_if_error(self, ValueError):
        raise ValueError()

  def test_skip_if_error_should_raise_message_mismatch(self):
    try:
      with self.assertRaisesRegex(ValueError, "foo bar"):
        with test_util.skip_if_error(self, ValueError, "test message"):
          raise ValueError("foo bar")
    except unittest.SkipTest:
      raise RuntimeError("Test is not supposed to skip.")

  def test_skip_if_error_should_raise_no_message(self):
    try:
      with self.assertRaisesRegex(ValueError, ""):
        with test_util.skip_if_error(self, ValueError, "test message"):
          raise ValueError()
    except unittest.SkipTest:
      raise RuntimeError("Test is not supposed to skip.")


# Its own test case to reproduce variable sharing issues which only pop up when
# setUp() is overridden and super() is not called.
class GraphAndEagerNoVariableSharing(test_util.TensorFlowTestCase):

  def setUp(self):
    pass  # Intentionally does not call TensorFlowTestCase's super()

  @test_util.run_in_graph_and_eager_modes
  def test_no_variable_sharing(self):
    variable_scope.get_variable(
        name="step_size",
        initializer=np.array(1e-5, np.float32),
        use_resource=True,
        trainable=False)


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

  @test_util.run_in_graph_and_eager_modes
  def test_no_leaked_tensor_decorator(self):

    class LeakedTensorTest(object):

      def __init__(inner_self):  # pylint: disable=no-self-argument
        inner_self.assertEqual = self.assertEqual  # pylint: disable=invalid-name

      @test_util.assert_no_new_tensors
      def test_has_leak(self):
        self.a = constant_op.constant([3.], name="leak")

      @test_util.assert_no_new_tensors
      def test_has_no_leak(self):
        constant_op.constant([3.], name="no-leak")

    with self.assertRaisesRegex(AssertionError, "Tensors not deallocated"):
      LeakedTensorTest().test_has_leak()

    LeakedTensorTest().test_has_no_leak()

  def test_no_new_objects_decorator(self):

    class LeakedObjectTest(unittest.TestCase):

      def __init__(self, *args, **kwargs):
        super(LeakedObjectTest, self).__init__(*args, **kwargs)
        self.accumulation = []

      @unittest.expectedFailure
      @test_util.assert_no_new_pyobjects_executing_eagerly
      def test_has_leak(self):
        self.accumulation.append([1.])

      @test_util.assert_no_new_pyobjects_executing_eagerly
      def test_has_no_leak(self):
        self.not_accumulating = [1.]

    self.assertTrue(LeakedObjectTest("test_has_leak").run().wasSuccessful())
    self.assertTrue(LeakedObjectTest("test_has_no_leak").run().wasSuccessful())


class RunFunctionsEagerlyInV2Test(test_util.TensorFlowTestCase,
                                  parameterized.TestCase):
  @parameterized.named_parameters(
      [("_RunEagerly", True), ("_RunGraph", False)])
  def test_run_functions_eagerly(self, run_eagerly):  # pylint: disable=g-wrong-blank-lines
    results = []

    @def_function.function
    def add_two(x):
      for _ in range(5):
        x += 2
        results.append(x)
      return x

    with test_util.run_functions_eagerly(run_eagerly):
      add_two(constant_op.constant(2.))
      if context.executing_eagerly():
        if run_eagerly:
          self.assertTrue(isinstance(t, ops.EagerTensor) for t in results)
        else:
          self.assertTrue(isinstance(t, ops.Tensor) for t in results)
      else:
        self.assertTrue(isinstance(t, ops.Tensor) for t in results)


class SyncDevicesTest(test_util.TensorFlowTestCase):

  def tearDown(self):
    super().tearDown()
    config.set_synchronous_execution(True)

  def test_sync_device_cpu(self):
    with context.eager_mode(), ops.device("/CPU:0"):
      config.set_synchronous_execution(False)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertLess(time.time() - start, 1.0)
      test_util.sync_devices()
      self.assertGreater(time.time() - start, 1.0)

      config.set_synchronous_execution(True)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertGreaterEqual(time.time() - start, 1.0)
      start = time.time()
      test_util.sync_devices()
      self.assertLess(time.time() - start, 1.0)

  def test_sync_device_gpu(self):
    if not test_util.is_gpu_available(min_cuda_compute_capability=(7, 0)):
      # sleep_op requires compute capability 7.0
      self.skipTest("Requires GPU with compute capability 7.0")

    with context.eager_mode(), ops.device("/GPU:0"):
      config.set_synchronous_execution(False)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertLess(time.time() - start, 1.0)
      test_util.sync_devices()
      self.assertGreater(time.time() - start, 1.0)

      config.set_synchronous_execution(True)
      start = time.time()
      test_ops.sleep_op(sleep_seconds=1)
      self.assertLess(time.time() - start, 1.0)
      start = time.time()
      test_util.sync_devices()
      self.assertGreaterEqual(time.time() - start, 1.0)

  def test_sync_devices_graph_mode_error(self):
    with context.graph_mode():
      with self.assertRaisesRegex(
          RuntimeError, r"sync_devices\(\) must only be called in Eager mode"
      ):
        test_util.sync_devices()


if __name__ == "__main__":
  googletest.main()
