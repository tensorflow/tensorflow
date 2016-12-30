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
from six.moves import xrange  # pylint: disable=redefined-builtin

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
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
      node_def = ops._NodeDef("op_type", "name")
      node_def_orig = ops._NodeDef("op_type_orig", "orig")
      op_orig = ops.Operation(node_def_orig, ops.get_default_graph())
      op = ops.Operation(node_def, ops.get_default_graph(), original_op=op_orig)
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
      self.assertAllClose(7, 8)

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
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Cannot assign a device to node"):
      with self.test_session(force_gpu=True):
        # this relies on us not having a GPU implementation for assert, which
        # seems sensible
        x = constant_op.constant(True)
        y = [15]
        control_flow_ops.Assert(x, y).run()

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


if __name__ == "__main__":
  googletest.main()
