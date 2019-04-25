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

import collections
import copy
import random
import threading
import weakref

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops  # pylint: disable=unused-import
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class TestUtilTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @test_util.run_deprecated_v1
  def test_assert_ops_in_graph(self):
    with self.test_session():
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
    with self.assertRaisesRegexp(AssertionError,
                                 r"^Found unexpected node '{{node seven}}"):
      test_util.assert_equal_graph_def(def_57, def_empty)

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

  @test_util.run_in_graph_and_eager_modes
  def testAssertProtoEqualsStr(self):

    graph_str = "node { name: 'w1' op: 'params' }"
    graph_def = graph_pb2.GraphDef()
    text_format.Merge(graph_str, graph_def)

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
    with self.assertRaisesRegexp(AssertionError,
                                 r'meta_graph_version: "inner"'):
      self.assertProtoEquals("", meta_graph_def_outer)

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
        op_orig = ops.Operation(node_def_orig, ops.get_default_graph())
        op = ops.Operation(node_def, ops.get_default_graph(),
                           original_op=op_orig)
        raise errors.UnauthenticatedError(node_def, op, "true_err")

  @test_util.run_in_graph_and_eager_modes
  def testAssertRaisesOpErrorDoesNotPassMessageDueToLeakedStack(self):
    with self.assertRaises(AssertionError):
      self._WeMustGoDeeper("this_is_not_the_error_you_are_looking_for")

    self._WeMustGoDeeper("true_err")
    self._WeMustGoDeeper("name")
    self._WeMustGoDeeper("orig")

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseTensors(self):
    a_raw_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    a = constant_op.constant(a_raw_data)
    b = math_ops.add(1, constant_op.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
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
    with self.assertRaisesRegexp(AssertionError, r"Not equal to tolerance"):
      self.assertAllClose(7, 7 + 1e-5)

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseList(self):
    with self.assertRaisesRegexp(AssertionError, r"not close dif"):
      self.assertAllClose([0], [1])

  @test_util.run_in_graph_and_eager_modes
  def testAllCloseDictToNonDict(self):
    with self.assertRaisesRegexp(ValueError, r"Can't compare dict to non-dict"):
      self.assertAllClose(1, {"a": 1})
    with self.assertRaisesRegexp(ValueError, r"Can't compare dict to non-dict"):
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
    with self.assertRaisesRegexp(AssertionError,
                                 r"\[y\]\[1\]\[0\]\[nested\]\[n\]"):
      self.assertAllClose(a, b)

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
    self.assertAllEqual([120] * 3, k)
    self.assertAllEqual([20] * 3, j)

    with self.assertRaisesRegexp(AssertionError, r"not equal lhs"):
      self.assertAllEqual([0] * 3, k)

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

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testRandomSeed(self):
    # Call setUp again for WithCApi case (since it makes a new defeault graph
    # after setup).
    # TODO(skyewm): remove this when C API is permanently enabled.
    self.setUp()
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
    f = test_util.run_in_graph_and_eager_modes()(inc)
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
    with self.assertRaisesRegexp(ValueError, msg):
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

  @test_util.run_deprecated_v1
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

    self.assertEqual(modes[0:2], ["setup_graph", "run_graph"])
    self.assertEqual(modes[2:], ["setup_eager", "run_eager"])

  @parameterized.named_parameters(dict(testcase_name="argument",
                                       arg=True))
  @test_util.run_in_graph_and_eager_modes
  def test_run_in_graph_and_eager_works_with_parameterized_keyword(self, arg):
    self.assertEqual(arg, True)

  def test_build_as_function_and_v1_graph(self):

    class GraphModeAndFuncionTest(parameterized.TestCase):

      def __init__(inner_self):  # pylint: disable=no-self-argument
        super(GraphModeAndFuncionTest, inner_self).__init__()
        inner_self.graph_mode_tested = False
        inner_self.inside_function_tested = False

      def runTest(self):
        del self

      @test_util.build_as_function_and_v1_graph
      def test_modes(inner_self):  # pylint: disable=no-self-argument
        is_building_function = ops.get_default_graph().building_function
        if is_building_function:
          self.assertFalse(inner_self.inside_function_tested)
          inner_self.inside_function_tested = True
        else:
          self.assertFalse(inner_self.graph_mode_tested)
          inner_self.graph_mode_tested = True

    test_object = GraphModeAndFuncionTest()
    test_object.test_modes_v1_graph()
    test_object.test_modes_function()
    self.assertTrue(test_object.graph_mode_tested)
    self.assertTrue(test_object.inside_function_tested)


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

    with self.assertRaisesRegexp(AssertionError, "Tensors not deallocated"):
      LeakedTensorTest().test_has_leak()

    LeakedTensorTest().test_has_no_leak()

  def test_no_new_objects_decorator(self):

    class LeakedObjectTest(object):

      def __init__(inner_self):  # pylint: disable=no-self-argument
        inner_self.assertEqual = self.assertEqual  # pylint: disable=invalid-name
        inner_self.accumulation = []

      @test_util.assert_no_new_pyobjects_executing_eagerly
      def test_has_leak(self):
        self.accumulation.append([1.])

      @test_util.assert_no_new_pyobjects_executing_eagerly
      def test_has_no_leak(self):
        self.not_accumulating = [1.]

    with self.assertRaises(AssertionError):
      LeakedObjectTest().test_has_leak()

    LeakedObjectTest().test_has_no_leak()


if __name__ == "__main__":
  googletest.main()
