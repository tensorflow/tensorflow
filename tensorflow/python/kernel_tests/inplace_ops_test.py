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
"""Tests for inplace_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class InplaceOpsTest(test.TestCase):

  def testBasicUpdate(self):
    for dtype in [dtypes.float32, dtypes.int32]:
      with self.test_session(use_gpu=True):
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_update(x, [3], array_ops.ones([1, 3],
                                                                  dtype))
        y[3, :] = 1
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_update(x, [-1],
                                           array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] = 2
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_update(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] = 7
        self.assertAllClose(x.eval(), y)

  def testBasicAdd(self):
    for dtype in [dtypes.float32, dtypes.int32]:
      with self.test_session(use_gpu=True):
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_add(x, [3], array_ops.ones([1, 3], dtype))
        y[3, :] += 1
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_add(x, [-1],
                                        array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] += 2
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_add(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] += 7
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_add(x, None,
                                        array_ops.ones([7, 3], dtype) * 99)
        y[:, :] += 99
        self.assertAllClose(x.eval(), y)

  def testBasicSub(self):
    for dtype in [dtypes.float32, dtypes.int32]:
      with self.test_session(use_gpu=True):
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_subtract(x, [3],
                                             array_ops.ones([1, 3], dtype))
        y[3, :] -= 1
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_subtract(x, [-1],
                                             array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] -= 2
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_subtract(x, 5,
                                             array_ops.ones([3], dtype) * 7)
        y[5, :] -= 7
        self.assertAllClose(x.eval(), y)
        x = array_ops.alias_inplace_subtract(x, None,
                                             array_ops.ones([7, 3], dtype) * 99)
        y[:, :] -= 99
        self.assertAllClose(x.eval(), y)

  def testRandom(self):
    with self.test_session(use_gpu=True):
      d0, d1, d2 = 100, 3, 5
      x = array_ops.zeros([d0, d1, d2])
      y = np.zeros([d0, d1, d2])
      for _ in range(20):
        idx = np.random.choice(d0, d0 / 10, replace=False)
        val = np.random.randint(10, size=(d0 / 10, d1, d2))
        op = np.random.randint(3)
        if op == 0:
          x = array_ops.alias_inplace_update(x, idx, val)
          y[idx, :] = val
        elif op == 1:
          x = array_ops.alias_inplace_add(x, idx, val)
          y[idx, :] += val
        elif op == 2:
          x = array_ops.alias_inplace_subtract(x, idx, val)
          y[idx, :] -= val
        self.assertAllClose(x.eval(), y)

  def testRandom1D(self):
    with self.test_session(use_gpu=True):
      d0 = 100
      x = array_ops.zeros([d0])
      y = np.zeros([d0])
      for _ in range(20):
        idx = np.random.choice(d0, d0 / 10, replace=False)
        val = np.random.randint(10, size=(d0 / 10))
        op = np.random.randint(3)
        if op == 0:
          x = array_ops.alias_inplace_update(x, idx, val)
          y[idx] = val
        elif op == 1:
          x = array_ops.alias_inplace_add(x, idx, val)
          y[idx] += val
        elif op == 2:
          x = array_ops.alias_inplace_subtract(x, idx, val)
          y[idx] -= val
        self.assertAllClose(x.eval(), y)

  def testError(self):
    with self.test_session():
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "must be a vector"):
        _ = array_ops.alias_inplace_update([[1.]], [[0]], [[10]]).eval()
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "value and update shape doesn't match"):
        _ = array_ops.alias_inplace_update([[1.]], [0], [10]).eval()
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "loc and update shape doesn't match"):
        _ = array_ops.alias_inplace_update([[1.]], [0, 1], [[10]]).eval()

  def testEmpty(self):
    # Not much to test except the output a empty should have the shape
    # and dtype we specify.
    for dtype in [dtypes.float32, dtypes.float64, dtypes.int32]:
      with self.test_session(use_gpu=True):
        test_shapes = [(), (1,), (2, 3), (0, 2), (2, 3, 5), (2, 0, 5)]
        for shape in test_shapes:
          val = array_ops.empty(shape, dtype).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          val = array_ops.empty(shape, dtype, init=True).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))
          val = array_ops.empty_like(array_ops.zeros(shape, dtype)).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          val = array_ops.empty_like(
              array_ops.zeros(shape, dtype), init=True).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))

        val = array_ops.empty((1, 2), dtypes.string, init=True).eval()
        self.assertEqual(val.tolist(), [[b"", b""]])

        val = array_ops.empty((1, 2), dtypes.string, init=False).eval()
        self.assertEqual(val.tolist(), [[b"", b""]])

  def testEmptyStateful(self):
    with session.Session("") as sess:
      v1 = array_ops.placeholder(dtypes.float32, shape=[])
      v2 = array_ops.placeholder(dtypes.float32, shape=[])

      a = array_ops.empty((1,), dtypes.float32, init=False)
      b = array_ops.empty((1,), dtypes.float32, init=False)

      a = array_ops.alias_inplace_update(a, 0, v1)
      b = array_ops.alias_inplace_update(b, 0, v2)

      res1, res2 = sess.run([a, b], feed_dict={v1: 1, v2: 2})
      self.assertEqual(res1, 1)
      self.assertEqual(res2, 2)


if __name__ == "__main__":
  test.main()
