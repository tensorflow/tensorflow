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
"""Tests for inplace_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.platform import test as test_lib


class InplaceOpsTest(test_util.TensorFlowTestCase):

  def testBasicUpdate(self):
    for dtype in [dtypes.float32, dtypes.int32, dtypes.int64]:
      with self.test_session(use_gpu=True):
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_update(x, [3], array_ops.ones([1, 3], dtype))
        y[3, :] = 1
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_update(x, [-1],
                                       array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] = 2
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_update(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] = 7
        self.assertAllClose(x.eval(), y)

  def testBasicUpdateBool(self):
    with self.test_session(use_gpu=True):
      x = array_ops.ones([7, 3], dtypes.bool)
      y = np.ones([7, 3], dtypes.bool.as_numpy_dtype)
      self.assertAllClose(x.eval(), y)
      x = inplace_ops.inplace_update(x, [3], array_ops.ones([1, 3],
                                                            dtypes.bool))
      y[3, :] = True
      self.assertAllClose(x.eval(), y)
      x = inplace_ops.inplace_update(x, [-1],
                                     array_ops.zeros([1, 3], dtypes.bool))
      y[-1, :] = False
      self.assertAllClose(x.eval(), y)
      x = inplace_ops.inplace_update(x, 5, array_ops.zeros([3], dtypes.bool))
      y[5, :] = False
      self.assertAllClose(x.eval(), y)

  def testBasicAdd(self):
    for dtype in [dtypes.float32, dtypes.int32, dtypes.int64]:
      with self.test_session(use_gpu=True):
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x.eval(), y)
        x = array_ops.inplace_add(x, [3], array_ops.ones([1, 3], dtype))
        y[3, :] += 1
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_add(x, [-1], array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] += 2
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_add(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] += 7
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_add(x, None, array_ops.ones([7, 3], dtype) * 99)
        y[:, :] += 99
        self.assertAllClose(x.eval(), y)

  def testBasicSub(self):
    for dtype in [dtypes.float32, dtypes.int32, dtypes.int64]:
      with self.test_session(use_gpu=True):
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_sub(x, [3], array_ops.ones([1, 3], dtype))
        y[3, :] -= 1
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_sub(x, [-1], array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] -= 2
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_sub(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] -= 7
        self.assertAllClose(x.eval(), y)
        x = inplace_ops.inplace_sub(x, None, array_ops.ones([7, 3], dtype) * 99)
        y[:, :] -= 99
        self.assertAllClose(x.eval(), y)

  def testRandom(self):
    with self.test_session(use_gpu=True):
      d0, d1, d2 = 100, 3, 5
      x = array_ops.zeros([d0, d1, d2])
      y = np.zeros([d0, d1, d2])
      for _ in xrange(20):
        idx = np.random.choice(d0, d0 // 10, replace=False)
        val = np.random.randint(10, size=(d0 // 10, d1, d2))
        op = np.random.randint(3)
        if op == 0:
          x = inplace_ops.inplace_update(x, idx, val)
          y[idx, :] = val
        elif op == 1:
          x = inplace_ops.inplace_add(x, idx, val)
          y[idx, :] += val
        elif op == 2:
          x = inplace_ops.inplace_sub(x, idx, val)
          y[idx, :] -= val
        self.assertAllClose(x.eval(), y)

  def testRandom1D(self):
    with self.test_session(use_gpu=True):
      d0 = 100
      x = array_ops.zeros([d0])
      y = np.zeros([d0])
      for _ in xrange(20):
        idx = np.random.choice(d0, d0 // 10, replace=False)
        val = np.random.randint(10, size=(d0 // 10))
        op = np.random.randint(3)
        if op == 0:
          x = inplace_ops.inplace_update(x, idx, val)
          y[idx] = val
        elif op == 1:
          x = inplace_ops.inplace_add(x, idx, val)
          y[idx] += val
        elif op == 2:
          x = inplace_ops.inplace_sub(x, idx, val)
          y[idx] -= val
        self.assertAllClose(x.eval(), y)

  def testAlias(self):
    with self.test_session(use_gpu=True) as sess:
      x = array_ops.ones([2, 3])
      y = inplace_ops.alias_inplace_add(x, [0], [[1, 2, 3]])
      with ops.control_dependencies([y]):
        z = array_ops.identity(x)
        _, vy, vz = sess.run([x, y, z])
      self.assertAllClose(vy, vz)

  def testError(self):
    with self.test_session():
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "must be a vector"):
        _ = inplace_ops.inplace_update([[1.]], [[0]], [[10]]).eval()
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "x and v shape doesn't match"):
        _ = inplace_ops.inplace_update([[1.]], [0], [10]).eval()
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "i and x shape doesn't match"):
        _ = inplace_ops.inplace_update([[1.]], [0, 1], [[10]]).eval()

  def testEmpty(self):
    for dtype in [
        dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64, dtypes.bool
    ]:
      with self.test_session(use_gpu=True):
        test_shapes = [(), (1,), (2, 3), (0, 2), (2, 3, 5), (2, 0, 5)]
        for shape in test_shapes:
          val = inplace_ops.empty(shape, dtype).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          val = inplace_ops.empty(shape, dtype, init=True).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))
          val = inplace_ops.empty_like(array_ops.zeros(shape, dtype)).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          val = inplace_ops.empty_like(
              array_ops.zeros(shape, dtype), init=True).eval()
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))

        val = inplace_ops.empty((1, 2), dtypes.string, init=True).eval()
        self.assertEqual(val.tolist(), [[b"", b""]])

        val = inplace_ops.empty((1, 2), dtypes.string, init=False).eval()
        self.assertEqual(val.tolist(), [[b"", b""]])


if __name__ == "__main__":
  test_lib.main()
