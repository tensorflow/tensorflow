# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for Tensorflow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class FunctionTest(xla_test.XLATestCase):

  def testFunction(self):
    """Executes a simple TensorFlow function."""

    def APlus2B(a, b):
      return a + b * 2

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([5, 6, 7, 8]).reshape([2, 2]).astype(np.float32)
    expected = APlus2B(aval, bval)

    with self.cached_session():

      @function.Defun(dtypes.float32, dtypes.float32)
      def Foo(a, b):
        return APlus2B(a, b)

      a = constant_op.constant(aval, name="a")
      b = constant_op.constant(bval, name="b")
      with self.test_scope():
        call_f = Foo(a, b)
      result = self.evaluate(call_f)
    self.assertAllClose(result, expected, rtol=1e-3)

  def testNestedFunctions(self):
    """Executes two nested TensorFlow functions."""

    def TimesTwo(x):
      return x * 2

    def APlus2B(a, b):
      return a + TimesTwo(b)

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    expected = APlus2B(aval, bval)

    with self.cached_session():

      @function.Defun(dtypes.float32, dtypes.float32)
      def Foo(a, b):
        return APlus2B(a, b)

      a = constant_op.constant(aval, name="a")
      b = constant_op.constant(bval, name="b")
      with self.test_scope():
        call_g = Foo(a, b)
      result = self.evaluate(call_g)
    self.assertAllClose(result, expected, rtol=1e-3)

  def testFunctionMultipleRetvals(self):
    """Executes a function with multiple return values."""

    # This function will run on the XLA device
    def Func(a, b):
      return a + b, a - b

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([5, 6, 7, 8]).reshape([2, 2]).astype(np.float32)
    expected = Func(aval, bval)

    with self.cached_session():

      @function.Defun(dtypes.float32, dtypes.float32)
      def Foo(a, b):
        return Func(a, b)

      a = constant_op.constant(aval, name="a")
      b = constant_op.constant(bval, name="b")
      with self.test_scope():
        call_f = Foo(a, b)
      result = self.evaluate(call_f)
    self.assertAllClose(result, expected, rtol=1e-3)

  def testCompileTimeConstantsInDefun(self):
    """Tests that XLA handles compile-time constants in defuns."""
    with self.cached_session() as sess:

      @function.Defun(dtypes.float32, dtypes.int32, dtypes.int32)
      def Foo(a, c, d):
        # c and d must be known at compile time
        x = array_ops.slice(a, c, d)
        return x

      a = array_ops.placeholder(dtypes.float32)
      c = array_ops.placeholder(dtypes.int32, shape=[4])
      d = array_ops.placeholder(dtypes.int32, shape=[4])
      with self.test_scope():
        call_f = Foo(a, c, d)
      result = sess.run(call_f, feed_dict={
          a: np.ones([1, 4, 4, 1]),
          c: [0, 0, 0, 0],
          d: [1, 2, 2, 1]})

    self.assertAllEqual(np.ones([1, 2, 2, 1]), result)

  # TODO(b/36139787): Re-enable this test when noinline works again.
  def DISABLED_testFunctionsNoInline(self):

    @function.Defun(dtypes.float32, noinline=True)
    def TimesTwo(x):
      return x * 2

    @function.Defun(dtypes.float32, dtypes.float32)
    def APlus2B(a, b):
      return a + TimesTwo(b)

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    expected = aval + bval * 2

    with self.cached_session() as sess:
      with self.test_scope():
        a = array_ops.placeholder(dtypes.float32, name="a")
        b = array_ops.placeholder(dtypes.float32, name="b")
        call = APlus2B(a, b)
      result = sess.run(call, {a: aval, b: bval})
    self.assertAllClose(result, expected, rtol=1e-3)


if __name__ == "__main__":
  googletest.main()
