"""Tests for tensorflow.ops.test_util."""
import threading

import tensorflow.python.platform
import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import types
from tensorflow.python.platform import googletest
from tensorflow.python.ops import logging_ops

class TestUtilTest(test_util.TensorFlowTestCase):

  def testIsGoogleCudaEnabled(self):
    # The test doesn't assert anything. It ensures the py wrapper
    # function is generated correctly.
    if test_util.IsGoogleCudaEnabled():
      print "GoogleCuda is enabled"
    else:
      print "GoogleCuda is disabled"

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
      return 1 / 0

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("integer division or modulo by zero"
                    in fe.exception.message)

  def testCheckedThreadWithWrongAssertionFails(self):
    x = 37

    def err_func():
      self.assertTrue(x < 10)

    t = self.checkedThread(target=err_func)
    t.start()
    with self.assertRaises(self.failureException) as fe:
      t.join()
    self.assertTrue("False is not true" in fe.exception.message)

  def testMultipleThreadsWithOneFailure(self):
    def err_func(i):
      self.assertTrue(i != 7)

    threads = [self.checkedThread(target=err_func, args=(i,))
               for i in range(10)]
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

  def testForceGPU(self):
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Cannot assign a device to node"):
      with self.test_session(force_gpu=True):
        # this relies on us not having a GPU implementation for assert, which
        # seems sensible
        x = [True]
        y = [15]
        logging_ops.Assert(x, y).run()

if __name__ == "__main__":
  googletest.main()
