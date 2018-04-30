# -*- coding: utf-8 -*-
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
"""Tests for py_func op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
from six.moves import queue
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


def np_func(x, y):
  return np.sinh(x) + np.cosh(y)


def matmul(x, y):
  return math_ops.matmul(x, y)


class PyFuncTest(test.TestCase):
  """Encapsulates tests for py_func and eager_py_func."""

  # ----- Tests for py_func -----
  def testRealDataTypes(self):
    def sum_func(x, y):
      return x + y
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64,
                  dtypes.uint8, dtypes.int8, dtypes.uint16, dtypes.int16,
                  dtypes.int32, dtypes.int64]:
      with self.test_session():
        x = constant_op.constant(1, dtype=dtype)
        y = constant_op.constant(2, dtype=dtype)
        z = self.evaluate(script_ops.py_func(sum_func, [x, y], dtype))
        self.assertEqual(z, 3)

  def testComplexDataTypes(self):
    def sub_func(x, y):
      return x - y
    for dtype in [dtypes.complex64, dtypes.complex128]:
      with self.test_session():
        x = constant_op.constant(1 + 1j, dtype=dtype)
        y = constant_op.constant(2 - 2j, dtype=dtype)
        z = self.evaluate(script_ops.py_func(sub_func, [x, y], dtype))
        self.assertEqual(z, -1 + 3j)

  def testBoolDataTypes(self):
    def and_func(x, y):
      return x and y
    dtype = dtypes.bool
    with self.test_session():
      x = constant_op.constant(True, dtype=dtype)
      y = constant_op.constant(False, dtype=dtype)
      z = self.evaluate(script_ops.py_func(and_func, [x, y], dtype))
      self.assertEqual(z, False)

  def testSingleType(self):
    with self.test_session():
      x = constant_op.constant(1.0, dtypes.float32)
      y = constant_op.constant(2.0, dtypes.float32)
      z = self.evaluate(script_ops.py_func(np_func, [x, y], dtypes.float32))
      self.assertEqual(z, np_func(1.0, 2.0).astype(np.float32))

  def testScalar(self):
    with self.test_session():
      x = constant_op.constant(1.0, dtypes.float32)
      y = constant_op.constant(2.0, dtypes.float32)
      z = self.evaluate(
          script_ops.eager_py_func(np_func, [x, y], [dtypes.float32]))
      self.assertEqual(z[0], np_func(1.0, 2.0).astype(np.float32))

  def testArray(self):
    with self.test_session():
      x = constant_op.constant([1.0, 2.0], dtypes.float64)
      y = constant_op.constant([2.0, 3.0], dtypes.float64)
      z = self.evaluate(script_ops.py_func(np_func, [x, y], [dtypes.float64]))
      self.assertAllEqual(z[0],
                          np_func([1.0, 2.0], [2.0, 3.0]).astype(np.float64))

  def testComplexType(self):
    with self.test_session():
      x = constant_op.constant(1 + 2j, dtypes.complex64)
      y = constant_op.constant(3 + 4j, dtypes.complex64)
      z = self.evaluate(script_ops.py_func(np_func, [x, y], dtypes.complex64))
      self.assertAllClose(z, np_func(1 + 2j, 3 + 4j))

  def testRFFT(self):
    with self.test_session():
      x = constant_op.constant([1., 2., 3., 4.], dtypes.float32)

      def rfft(x):
        return np.fft.rfft(x).astype(np.complex64)

      y = self.evaluate(script_ops.py_func(rfft, [x], dtypes.complex64))
      self.assertAllClose(y, np.fft.rfft([1., 2., 3., 4.]))

  def testPythonLiteral(self):
    with self.test_session():

      def literal(x):
        return 1.0 if float(x) == 0.0 else 0.0

      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(script_ops.py_func(literal, [x], dtypes.float64))
      self.assertAllClose(y, 1.0)

  def testList(self):
    with self.test_session():

      def list_func(x):
        return [x, x + 1]

      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(
          script_ops.py_func(list_func, [x], [dtypes.float64] * 2))
      self.assertAllClose(y, [0.0, 1.0])

  def testTuple(self):
    # returns a tuple
    with self.test_session():

      def tuple_func(x):
        return x, x + 1

      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(
          script_ops.py_func(tuple_func, [x], [dtypes.float64] * 2))
      self.assertAllClose(y, [0.0, 1.0])

    # returns a tuple, Tout and inp a tuple
    with self.test_session():
      x = constant_op.constant(0.0, dtypes.float64)
      y = self.evaluate(
          script_ops.py_func(tuple_func, (x,),
                             (dtypes.float64, dtypes.float64)))
      self.assertAllClose(y, [0.0, 1.0])

  def testStrings(self):

    def read_fixed_length_numpy_strings():
      return np.array([b" there"])

    def read_and_return_strings(x, y):
      return x + y

    with self.test_session():
      x = constant_op.constant([b"hello", b"hi"], dtypes.string)
      y = self.evaluate(
          script_ops.py_func(read_fixed_length_numpy_strings, [],
                             dtypes.string))
      z = self.evaluate(
          script_ops.py_func(read_and_return_strings, [x, y], dtypes.string))
      self.assertAllEqual(z, [b"hello there", b"hi there"])

  def testStringsAreConvertedToBytes(self):

    def read_fixed_length_numpy_strings():
      return np.array([" there"])

    def read_and_return_strings(x, y):
      return x + y

    with self.test_session():
      x = constant_op.constant(["hello", "hi"], dtypes.string)
      y = self.evaluate(
          script_ops.py_func(read_fixed_length_numpy_strings, [],
                             dtypes.string))
      z = self.evaluate(
          script_ops.py_func(read_and_return_strings, [x, y], dtypes.string))
      self.assertAllEqual(z, [b"hello there", b"hi there"])

  def testObjectArraysAreConvertedToBytes(self):

    def read_object_array():
      return np.array([b" there", u" ya"], dtype=np.object)

    def read_and_return_strings(x, y):
      return x + y

    with self.test_session():
      x = constant_op.constant(["hello", "hi"], dtypes.string)
      y, = script_ops.py_func(read_object_array, [],
                              [dtypes.string])
      z, = script_ops.py_func(read_and_return_strings, [x, y], [dtypes.string])
      self.assertListEqual(list(z.eval()), [b"hello there", b"hi ya"])

  def testStringPadding(self):
    correct = [b"this", b"is", b"a", b"test"]
    with self.test_session():
      s, = script_ops.py_func(lambda: [correct], [], [dtypes.string])
      self.assertAllEqual(s.eval(), correct)

  def testStringPaddingAreConvertedToBytes(self):
    inp = ["this", "is", "a", "test"]
    correct = [b"this", b"is", b"a", b"test"]
    with self.test_session():
      s, = script_ops.py_func(lambda: [inp], [], [dtypes.string])
      self.assertAllEqual(s.eval(), correct)

  def testLarge(self):
    with self.test_session() as sess:
      x = array_ops.zeros([1000000], dtype=np.float32)
      y = script_ops.py_func(lambda x: x + 1, [x], [dtypes.float32])
      z = script_ops.py_func(lambda x: x * 2, [x], [dtypes.float32])
      for _ in xrange(100):
        sess.run([y[0].op, z[0].op])

  def testNoInput(self):
    with self.test_session():
      x = self.evaluate(script_ops.py_func(lambda: 42.0, [], dtypes.float64))
      self.assertAllClose(x, 42.0)

  def testAlias(self):
    with self.test_session():
      np_array = np.array([1.0, 2.0], dtype=np.float32)
      tf_array = script_ops.py_func(lambda: np_array, [], [dtypes.float32])
      value = tf_array + constant_op.constant([2.0, 3.0], dtype=dtypes.float32)
      value.op.run()
      self.assertAllEqual(np_array, [1.0, 2.0])

  def testReturnUnicodeString(self):
    with self.test_session():
      correct = u"你好 世界"

      def unicode_string():
        return correct

      z, = script_ops.py_func(unicode_string, [], [dtypes.string])
      self.assertEqual(z.eval(), correct.encode("utf8"))

  def testBadNumpyReturnType(self):
    with self.test_session():

      def bad():
        # Structured numpy arrays aren't supported.
        return np.array([], dtype=[("foo", np.float32)])

      y, = script_ops.py_func(bad, [], [dtypes.float32])

      with self.assertRaisesRegexp(errors.UnimplementedError,
                                   "Unsupported numpy type"):
        y.eval()

  def testBadReturnType(self):
    with self.test_session():

      def bad():
        # Non-string python objects aren't supported.
        return {"foo": dtypes.float32}

      z, = script_ops.py_func(bad, [], [dtypes.int64])

      with self.assertRaisesRegexp(errors.UnimplementedError,
                                   "Unsupported object type"):
        z.eval()

  def testReturnInput(self):
    with self.test_session():

      def ident(x):
        return x[0]

      p = array_ops.placeholder(dtypes.float32)

      # Create a numpy array aliasing a tensor and a tensor aliasing this array
      z, = script_ops.py_func(ident, [p], [dtypes.float32])
      z += 0.0  # Makes sure we release the tensor aliasing the numpy array x[0]
      # above instead of using its memory as the return value of
      # session.run
      self.assertEqual(0.0, z.eval(feed_dict={p: [0.0]}))

  def testStateful(self):
    # Not using self.test_session(), which disables optimization.
    with session_lib.Session() as sess:
      producer = iter(range(3))
      x, = script_ops.py_func(lambda: next(producer), [], [dtypes.int64])
      self.assertEqual(sess.run(x), 0)
      self.assertEqual(sess.run(x), 1)
      self.assertEqual(sess.run(x), 2)

  def testStateless(self):
    # Not using self.test_session(), which disables optimization.
    with session_lib.Session() as sess:
      producer = iter(range(3))
      x, = script_ops.py_func(
          lambda: next(producer), [], [dtypes.int64], stateful=False)
      self.assertEqual(sess.run(x), 0)
      self.assertEqual(sess.run(x), 0)
      self.assertEqual(sess.run(x), 0)

  def testGradientFunction(self):
    # Input to tf.py_func is necessary, otherwise get_gradient_function()
    # returns None per default.
    a = constant_op.constant(0)
    x, = script_ops.py_func(lambda a: 0, [a], [dtypes.int64])
    y, = script_ops.py_func(lambda a: 0, [a], [dtypes.int64], stateful=False)
    self.assertEqual(None, ops.get_gradient_function(x.op))
    self.assertEqual(None, ops.get_gradient_function(y.op))

  def testCOrder(self):
    with self.test_session():
      val = [[1, 2], [3, 4]]
      x, = script_ops.py_func(lambda: np.array(val, order="F"), [],
                              [dtypes.int64])
      self.assertAllEqual(val, x.eval())

  def testParallel(self):
    # Tests that tf.py_func's can run in parallel if they release the GIL.
    with self.test_session() as session:
      q = queue.Queue(1)

      def blocking_put():
        q.put(42)
        q.join()  # Wait for task_done().
        return 42

      def blocking_get():
        v = q.get(block=True)  # Wait for put().
        q.task_done()
        return v

      x, = script_ops.py_func(blocking_put, [], [dtypes.int64])
      y, = script_ops.py_func(blocking_get, [], [dtypes.int64])

      # This will result in a deadlock if the py_func's don't run in parallel.
      session.run([x, y])

  def testNoReturnValueStateful(self):

    class State(object):

      def __init__(self):
        self._value = np.array([1], np.int64)

      def _increment(self, diff):
        self._value += diff

      def increment(self, diff):
        return script_ops.py_func(self._increment, [diff], [], stateful=True)

      @property
      def value(self):
        return self._value

    with self.test_session():
      s = State()
      op = s.increment(constant_op.constant(2, dtypes.int64))
      ret = self.evaluate(op)
      self.assertIsNone(ret)
      self.assertAllEqual([3], s.value)

  def testNoReturnValueStateless(self):

    def do_nothing(unused_x):
      pass

    f = script_ops.py_func(
        do_nothing, [constant_op.constant(3, dtypes.int64)], [], stateful=False)
    with self.test_session() as sess:
      self.assertEqual(sess.run(f), [])

  def _testExceptionHandling(self, py_exp, tf_exp, eager=False):

    def inner_exception():
      raise py_exp("blah")  # pylint: disable=not-callable

    def raise_exception():
      inner_exception()

    expected_regexp = r": blah.*"               # Error at the top
    expected_regexp += r"in raise_exception.*"  # Stacktrace outer
    expected_regexp += r"in inner_exception.*"  # Stacktrace inner
    expected_regexp += r": blah"                # Stacktrace of raise
    def expected_error_check(exception):
      return re.search(expected_regexp, str(exception), re.DOTALL)

    if eager:
      if context.executing_eagerly():
        with self.assertRaisesWithPredicateMatch(tf_exp, expected_error_check):
          f = script_ops.eager_py_func(raise_exception, [], [])
        return
      else:
        f = script_ops.eager_py_func(raise_exception, [], [])
    else:
      f = script_ops.py_func(raise_exception, [], [])

    with self.test_session():
      with self.assertRaisesWithPredicateMatch(tf_exp, expected_error_check):
        self.evaluate(f)

  def testExceptionHandling(self):
    self._testExceptionHandling(ValueError, errors.InvalidArgumentError)
    self._testExceptionHandling(TypeError, errors.InvalidArgumentError)
    self._testExceptionHandling(StopIteration, errors.OutOfRangeError)
    self._testExceptionHandling(MemoryError, errors.ResourceExhaustedError)
    self._testExceptionHandling(NotImplementedError, errors.UnimplementedError)

    class WeirdError(Exception):
      pass

    self._testExceptionHandling(WeirdError, errors.UnknownError)

  # ----- Tests shared by py_func and eager_py_func -----
  def testCleanup(self):
    for _ in xrange(1000):
      g = ops.Graph()
      with g.as_default():
        c = constant_op.constant([1.], dtypes.float32)
        _ = script_ops.py_func(lambda x: x + 1, [c], [dtypes.float32])
        _ = script_ops.eager_py_func(lambda x: x + 1, [c], [dtypes.float32])
    self.assertTrue(script_ops._py_funcs.size() < 100)

  # ----- Tests for eager_py_func -----
  @test_util.run_in_graph_and_eager_modes()
  def testEagerSingleOutputInt32(self):
    a = array_ops.ones((3, 3), dtype=dtypes.int32)
    x = array_ops.ones((3, 1), dtype=dtypes.int32)
    output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.int32)
    with self.test_session():
      ret = self.evaluate(output)
      self.assertAllEqual(ret, [[3], [3], [3]])

  @test_util.run_in_graph_and_eager_modes()
  def testEagerSingleOutputFloat32(self):
    with test_util.device(use_gpu=True):
      a = array_ops.ones((3, 3), dtype=dtypes.float32)
      x = array_ops.ones((3, 1), dtype=dtypes.float32)
      output = script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)
      ret = self.evaluate(output)
      self.assertAllClose(ret, [[3.0], [3.0], [3.0]])

  @test_util.run_in_graph_and_eager_modes()
  def testEagerArrayOutput(self):
    with test_util.device(use_gpu=True):
      a = array_ops.ones((3, 3), dtype=dtypes.float32)
      x = array_ops.ones((3, 1), dtype=dtypes.float32)
      output = script_ops.eager_py_func(
          lambda a, x: [matmul(a, x)], inp=[a, x], Tout=[dtypes.float32])
      ret = self.evaluate(output)
      self.assertAllEqual(ret, [[[3.0], [3.0], [3.0]]])

  @test_util.run_in_graph_and_eager_modes()
  def testEagerReturnNone(self):
    with test_util.device(use_gpu=True):
      def no_return_value():
        return

      output = script_ops.eager_py_func(no_return_value, inp=[], Tout=[])
      ret = self.evaluate(output)
      if context.executing_eagerly():
        self.assertEquals(len(ret), 0)
      else:
        self.assertIsNone(ret)

  @test_util.run_in_graph_and_eager_modes()
  def testEagerPyFuncInDefun(self):
    with test_util.device(use_gpu=True):
      def wrapper():
        a = array_ops.ones((3, 3), dtype=dtypes.float32)
        x = array_ops.ones((3, 1), dtype=dtypes.float32)
        return script_ops.eager_py_func(matmul, inp=[a, x], Tout=dtypes.float32)

      wrapped = function.defun(wrapper)
      ret = self.evaluate(wrapped())
      self.assertAllEqual(ret, [[3.0], [3.0], [3.0]])

  @test_util.run_in_graph_and_eager_modes()
  def testEagerExceptionHandling(self):
    with test_util.device(use_gpu=True):
      self._testExceptionHandling(
          ValueError, errors.InvalidArgumentError, eager=True)
      self._testExceptionHandling(
          TypeError, errors.InvalidArgumentError, eager=True)
      self._testExceptionHandling(
          StopIteration, errors.OutOfRangeError, eager=True)
      self._testExceptionHandling(
          MemoryError, errors.ResourceExhaustedError, eager=True)
      self._testExceptionHandling(
          NotImplementedError, errors.UnimplementedError, eager=True)

      class WeirdError(Exception):
        pass

      self._testExceptionHandling(WeirdError, errors.UnknownError, eager=True)

  @test_util.run_in_graph_and_eager_modes()
  def testEagerReturningVariableRaisesError(self):
    def return_variable():
      variable = resource_variable_ops.ResourceVariable(0.0)
      return variable

    with self.assertRaisesRegexp(errors.UnknownError,
                                 "Attempting to return a variable"):
      output = script_ops.eager_py_func(
          return_variable, inp=[], Tout=dtypes.float32)
      self.evaluate(output)


if __name__ == "__main__":
  test.main()
