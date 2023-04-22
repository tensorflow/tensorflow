# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ndarray."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
# Required for operator overloads
from tensorflow.python.ops.numpy_ops import np_math_ops  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class ArrayTest(test.TestCase):

  def testDtype(self):
    a = array_ops.zeros(shape=[1, 2], dtype=dtypes.int64)
    self.assertIs(a.dtype.as_numpy_dtype, np.int64)
    np_dt = a.dtype.as_numpy_dtype
    self.assertAllEqual(0, np_dt(0))

  def testAstype(self):
    a = ops.convert_to_tensor(value=1.1, dtype=dtypes.float32).astype(np.int32)
    self.assertIs(a.dtype.as_numpy_dtype, np.int32)
    self.assertAllEqual(1, a)
    a = ops.convert_to_tensor(value=[0.0, 1.1], dtype=dtypes.float32).astype(
        np.bool_)
    self.assertIs(a.dtype.as_numpy_dtype, np.bool_)
    self.assertAllEqual([False, True], a)

  def testNeg(self):
    a = ops.convert_to_tensor(value=[1.0, 2.0])
    self.assertAllEqual([-1.0, -2.0], -a)  # pylint: disable=invalid-unary-operand-type

  def _testBinOp(self, a, b, out, f, types=None):
    a = ops.convert_to_tensor(value=a, dtype=np.int32)
    b = ops.convert_to_tensor(value=b, dtype=np.int32)
    if not isinstance(out, np_arrays.ndarray):
      out = ops.convert_to_tensor(value=out, dtype=np.int32)
    if types is None:
      types = [[np.int32, np.int32, np.int32], [np.int64, np.int32, np.int64],
               [np.int32, np.int64, np.int64],
               [np.float32, np.int32, np.float64],
               [np.int32, np.float32, np.float64],
               [np.float32, np.float32, np.float32],
               [np.float64, np.float32, np.float64],
               [np.float32, np.float64, np.float64]]
    for a_type, b_type, out_type in types:
      o = f(a.astype(a_type), b.astype(b_type))
      self.assertIs(o.dtype.as_numpy_dtype, out_type)
      out = out.astype(out_type)
      if np.issubdtype(out_type, np.inexact):
        self.assertAllClose(out, o)
      else:
        self.assertAllEqual(out, o)

  def testAdd(self):
    self._testBinOp([1, 2], [3, 4], [4, 6], lambda a, b: a.__add__(b))

  def testRadd(self):
    self._testBinOp([1, 2], [3, 4], [4, 6], lambda a, b: b.__radd__(a))

  def testSub(self):
    self._testBinOp([1, 2], [3, 5], [-2, -3], lambda a, b: a.__sub__(b))

  def testRsub(self):
    self._testBinOp([1, 2], [3, 5], [-2, -3], lambda a, b: b.__rsub__(a))

  def testMul(self):
    self._testBinOp([1, 2], [3, 4], [3, 8], lambda a, b: a.__mul__(b))

  def testRmul(self):
    self._testBinOp([1, 2], [3, 4], [3, 8], lambda a, b: b.__rmul__(a))

  def testPow(self):
    self._testBinOp([4, 5], [3, 2], [64, 25], lambda a, b: a.__pow__(b))

  def testRpow(self):
    self._testBinOp([4, 5], [3, 2], [64, 25], lambda a, b: b.__rpow__(a))

  _truediv_types = [[np.int32, np.int32, np.float64],
                    [np.int64, np.int32, np.float64],
                    [np.int32, np.int64, np.float64],
                    [np.float32, np.int32, np.float64],
                    [np.int32, np.float32, np.float64],
                    [np.float32, np.float32, np.float32],
                    [np.float64, np.float32, np.float64],
                    [np.float32, np.float64, np.float64]]

  def testTruediv(self):
    self._testBinOp([3, 5], [2, 4],
                    ops.convert_to_tensor(value=[1.5, 1.25]),
                    lambda a, b: a.__truediv__(b),
                    types=self._truediv_types)

  def testRtruediv(self):
    self._testBinOp([3, 5], [2, 4],
                    ops.convert_to_tensor(value=[1.5, 1.25]),
                    lambda a, b: b.__rtruediv__(a),
                    types=self._truediv_types)

  def _testCmp(self, a, b, out, f):
    a = ops.convert_to_tensor(value=a, dtype=np.int32)
    b = ops.convert_to_tensor(value=b, dtype=np.int32)

    types = [[np.int32, np.int32], [np.int64, np.int32], [np.int32, np.int64],
             [np.float32, np.int32], [np.int32, np.float32],
             [np.float32, np.float32], [np.float64, np.float32],
             [np.float32, np.float64]]
    for a_type, b_type in types:
      o = f(a.astype(a_type), b.astype(b_type))
      self.assertAllEqual(out, o)

  def testLt(self):
    self._testCmp([1, 2, 3], [3, 2, 1], [True, False, False],
                  lambda a, b: a.__lt__(b))

  def testLe(self):
    self._testCmp([1, 2, 3], [3, 2, 1], [True, True, False],
                  lambda a, b: a.__le__(b))

  def testGt(self):
    self._testCmp([1, 2, 3], [3, 2, 1], [False, False, True],
                  lambda a, b: a.__gt__(b))

  def testGe(self):
    self._testCmp([1, 2, 3], [3, 2, 1], [False, True, True],
                  lambda a, b: a.__ge__(b))

  def testEq(self):
    self._testCmp([1, 2, 3], [3, 2, 1], [False, True, False],
                  lambda a, b: a.__eq__(b))

  def testNe(self):
    self._testCmp([1, 2, 3], [3, 2, 1], [True, False, True],
                  lambda a, b: a.__ne__(b))

  def testInt(self):
    v = 10
    u = int(ops.convert_to_tensor(value=v))
    self.assertIsInstance(u, int)
    self.assertAllEqual(v, u)

  def testFloat(self):
    v = 21.32
    u = float(ops.convert_to_tensor(value=v))
    self.assertIsInstance(u, float)
    self.assertAllClose(v, u)

  def testBool(self):
    b = bool(ops.convert_to_tensor(value=10))
    self.assertIsInstance(b, bool)
    self.assertTrue(b)
    self.assertFalse(bool(ops.convert_to_tensor(value=0)))
    self.assertTrue(bool(ops.convert_to_tensor(value=0.1)))
    self.assertFalse(bool(ops.convert_to_tensor(value=0.0)))

  def testHash(self):
    a = ops.convert_to_tensor(value=10)
    def eager():
      hash(a)
    def graph():
      @def_function.function
      def f(x):
        hash(x)
      f(a)
    for f in [eager, graph]:
      with self.assertRaisesRegexp(
          TypeError,
          r'Tensor is unhashable. Instead, use tensor.ref\(\) as the key.'):
        f()

  def testFromToCompositeTensor(self):
    tensors = [ops.convert_to_tensor(0.1), ops.convert_to_tensor(0.2)]

    flattened = nest.flatten(tensors, expand_composites=True)
    # Each ndarray contains only one tensor, so the flattened output should be
    # just 2 tensors in a list.
    self.assertLen(flattened, 2)
    self.assertIsInstance(flattened[0], ops.Tensor)
    self.assertIsInstance(flattened[1], ops.Tensor)

    repacked = nest.pack_sequence_as(tensors, flattened, expand_composites=True)
    self.assertLen(repacked, 2)
    self.assertIsInstance(repacked[0], np_arrays.ndarray)
    self.assertIsInstance(repacked[1], np_arrays.ndarray)

    self.assertAllClose(tensors, repacked)


if __name__ == '__main__':
  # TODO(wangpeng): Test in graph mode as well. Also test in V2 (the requirement
  # for setting _USE_EQUALITY points to V2 behavior not being on).
  ops.enable_eager_execution()
  ops.Tensor._USE_EQUALITY = True
  ops.enable_numpy_style_type_promotion()
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
