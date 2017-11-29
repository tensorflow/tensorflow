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

import numpy as np
from six.moves import queue
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class PyOpTest(test.TestCase):

  def testBasic(self):

    def my_func(x, y):
      return np.sinh(x) + np.cosh(y)

    # single type
    with self.test_session():
      x = constant_op.constant(1.0, dtypes.float32)
      y = constant_op.constant(2.0, dtypes.float32)
      z = script_ops.py_func(my_func, [x, y], dtypes.float32)
      self.assertEqual(z.eval(), my_func(1.0, 2.0).astype(np.float32))

    # scalar
    with self.test_session():
      x = constant_op.constant(1.0, dtypes.float32)
      y = constant_op.constant(2.0, dtypes.float32)
      z = script_ops.py_func(my_func, [x, y], [dtypes.float32])
      self.assertEqual(z[0].eval(), my_func(1.0, 2.0).astype(np.float32))

    # array
    with self.test_session():
      x = constant_op.constant([1.0, 2.0], dtypes.float64)
      y = constant_op.constant([2.0, 3.0], dtypes.float64)
      z = script_ops.py_func(my_func, [x, y], [dtypes.float64])
      self.assertAllEqual(z[0].eval(),
                          my_func([1.0, 2.0], [2.0, 3.0]).astype(np.float64))

    # a bit exotic type (complex64)
    with self.test_session():
      x = constant_op.constant(1 + 2j, dtypes.complex64)
      y = constant_op.constant(3 + 4j, dtypes.complex64)
      z, = script_ops.py_func(my_func, [x, y], [dtypes.complex64])
      self.assertAllClose(z.eval(), my_func(1 + 2j, 3 + 4j))

    # a bit excotic function (rfft)
    with self.test_session():
      x = constant_op.constant([1., 2., 3., 4.], dtypes.float32)

      def rfft(x):
        return np.fft.rfft(x).astype(np.complex64)

      y, = script_ops.py_func(rfft, [x], [dtypes.complex64])
      self.assertAllClose(y.eval(), np.fft.rfft([1., 2., 3., 4.]))

    # returns a python literal.
    with self.test_session():

      def literal(x):
        return 1.0 if x == 0.0 else 0.0

      x = constant_op.constant(0.0, dtypes.float64)
      y, = script_ops.py_func(literal, [x], [dtypes.float64])
      self.assertAllClose(y.eval(), 1.0)

    # returns a list
    with self.test_session():

      def list_func(x):
        return [x, x + 1]

      x = constant_op.constant(0.0, dtypes.float64)
      y, z = script_ops.py_func(list_func, [x], [dtypes.float64] * 2)
      self.assertAllClose(y.eval(), 0.0)
      self.assertAllClose(z.eval(), 1.0)

    # returns a tuple
    with self.test_session():

      def tuple_func(x):
        return x, x + 1

      x = constant_op.constant(0.0, dtypes.float64)
      y, z = script_ops.py_func(tuple_func, [x], [dtypes.float64] * 2)
      self.assertAllClose(y.eval(), 0.0)
      self.assertAllClose(z.eval(), 1.0)

    # returns a tuple, Tout and inp a tuple
    with self.test_session():
      x = constant_op.constant(0.0, dtypes.float64)
      y, z = script_ops.py_func(tuple_func, (x,), (dtypes.float64,
                                                   dtypes.float64))
      self.assertAllClose(y.eval(), 0.0)
      self.assertAllClose(z.eval(), 1.0)

  def testStrings(self):

    def read_fixed_length_numpy_strings():
      return np.array([b" there"])

    def read_and_return_strings(x, y):
      return x + y

    with self.test_session():
      x = constant_op.constant([b"hello", b"hi"], dtypes.string)
      y, = script_ops.py_func(read_fixed_length_numpy_strings, [],
                              [dtypes.string])
      z, = script_ops.py_func(read_and_return_strings, [x, y], [dtypes.string])
      self.assertListEqual(list(z.eval()), [b"hello there", b"hi there"])

  def testStringPadding(self):
    correct = [b"this", b"is", b"a", b"test"]
    with self.test_session():
      s, = script_ops.py_func(lambda: [correct], [], [dtypes.string])
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
      x, = script_ops.py_func(lambda: 42.0, [], [dtypes.float64])
      self.assertAllClose(x.eval(), 42.0)

  def testCleanup(self):
    for _ in xrange(1000):
      g = ops.Graph()
      with g.as_default():
        c = constant_op.constant([1.], dtypes.float32)
        _ = script_ops.py_func(lambda x: x + 1, [c], [dtypes.float32])
    self.assertTrue(script_ops._py_funcs.size() < 100)

  def testAlias(self):
    with self.test_session():
      np_array = np.array([1.0, 2.0], dtype=np.float32)
      tf_array = script_ops.py_func(lambda: np_array, [], [dtypes.float32])
      value = tf_array + constant_op.constant([2.0, 3.0], dtype=dtypes.float32)
      value.op.run()
      self.assertAllEqual(np_array, [1.0, 2.0])

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

    with self.test_session() as sess:
      s = State()
      op = s.increment(constant_op.constant(2, dtypes.int64))
      ret = sess.run(op)
      self.assertIsNone(ret)
      self.assertAllEqual([3], s.value)

  def testNoReturnValueStateless(self):

    def do_nothing(unused_x):
      pass

    f = script_ops.py_func(
        do_nothing, [constant_op.constant(3, dtypes.int64)], [], stateful=False)
    with self.test_session() as sess:
      self.assertEqual(sess.run(f), [])

  def _testExceptionHandling(self, py_exp, tf_exp):

    def raise_exception():
      raise py_exp("blah")  # pylint: disable=not-callable

    f = script_ops.py_func(raise_exception, [], [])
    with self.test_session() as sess:
      with self.assertRaisesRegexp(tf_exp, "blah"):
        sess.run(f)

  def testExceptionHandling(self):
    self._testExceptionHandling(ValueError, errors.InvalidArgumentError)
    self._testExceptionHandling(TypeError, errors.InvalidArgumentError)
    self._testExceptionHandling(StopIteration, errors.OutOfRangeError)
    self._testExceptionHandling(MemoryError, errors.ResourceExhaustedError)
    self._testExceptionHandling(NotImplementedError, errors.UnimplementedError)

    class WeirdError(Exception):
      pass

    self._testExceptionHandling(WeirdError, errors.UnknownError)


if __name__ == "__main__":
  test.main()
