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
"""Tests for tensorflow.kernels.unique_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


class UniqueTest(test.TestCase):

  def testInt32(self):
    x = np.random.randint(2, high=10, size=7000)
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testInt32OutIdxInt64(self):
    x = np.random.randint(2, high=10, size=7000)
    y, idx = array_ops.unique(x, out_idx=dtypes.int64)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testString(self):
    indx = np.random.randint(65, high=122, size=7000)
    x = [chr(i) for i in indx]
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))

  def testInt32Axis(self):
    for dtype in [np.int32, np.int64]:
      with self.subTest(dtype=dtype):
        x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
        y0, idx0 = gen_array_ops.unique_v2(x, axis=np.array([0], dtype))
        self.assertEqual(y0.shape.rank, 2)
        tf_y0, tf_idx0 = self.evaluate([y0, idx0])
        y1, idx1 = gen_array_ops.unique_v2(x, axis=np.array([1], dtype))
        self.assertEqual(y1.shape.rank, 2)
        tf_y1, tf_idx1 = self.evaluate([y1, idx1])
        self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
        self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
        self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
        self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))

  def testInt32V2(self):
    # This test is only temporary, once V2 is used
    # by default, the axis will be wrapped to allow `axis=None`.
    x = np.random.randint(2, high=10, size=7000)
    y, idx = gen_array_ops.unique_v2(x, axis=np.array([], np.int32))
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testBool(self):
    x = np.random.choice([True, False], size=7000)
    y, idx = array_ops.unique(x)
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])

  def testBoolV2(self):
    x = np.random.choice([True, False], size=7000)
    y, idx = gen_array_ops.unique_v2(x, axis=np.array([], np.int32))
    tf_y, tf_idx = self.evaluate([y, idx])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])


class UniqueWithCountsTest(test.TestCase):

  def testInt32(self):
    x = np.random.randint(2, high=10, size=7000)
    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testInt32OutIdxInt64(self):
    x = np.random.randint(2, high=10, size=7000)
    y, idx, count = array_ops.unique_with_counts(x, out_idx=dtypes.int64)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testString(self):
    indx = np.random.randint(65, high=122, size=7000)
    x = [chr(i) for i in indx]

    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]].decode('ascii'))
    for value, count in zip(tf_y, tf_count):
      with self.subTest(value=value, count=count):
        v = [1 if x[i] == value.decode('ascii') else 0 for i in range(7000)]
        self.assertEqual(count, sum(v))

  def testInt32Axis(self):
    for dtype in [np.int32, np.int64]:
      with self.subTest(dtype=dtype):
        x = np.array([[1, 0, 0], [1, 0, 0], [2, 0, 0]])
        y0, idx0, count0 = gen_array_ops.unique_with_counts_v2(
            x, axis=np.array([0], dtype))
        self.assertEqual(y0.shape.rank, 2)
        tf_y0, tf_idx0, tf_count0 = self.evaluate([y0, idx0, count0])
        y1, idx1, count1 = gen_array_ops.unique_with_counts_v2(
            x, axis=np.array([1], dtype))
        self.assertEqual(y1.shape.rank, 2)
        tf_y1, tf_idx1, tf_count1 = self.evaluate([y1, idx1, count1])
        self.assertAllEqual(tf_y0, np.array([[1, 0, 0], [2, 0, 0]]))
        self.assertAllEqual(tf_idx0, np.array([0, 0, 1]))
        self.assertAllEqual(tf_count0, np.array([2, 1]))
        self.assertAllEqual(tf_y1, np.array([[1, 0], [1, 0], [2, 0]]))
        self.assertAllEqual(tf_idx1, np.array([0, 1, 1]))
        self.assertAllEqual(tf_count1, np.array([1, 2]))

  def testInt32V2(self):
    # This test is only temporary, once V2 is used
    # by default, the axis will be wrapped to allow `axis=None`.
    x = np.random.randint(2, high=10, size=7000)
    y, idx, count = gen_array_ops.unique_with_counts_v2(
        x, axis=np.array([], np.int32))
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testBool(self):
    x = np.random.choice([True, False], size=7000)
    y, idx, count = array_ops.unique_with_counts(x)
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testBoolV2(self):
    x = np.random.choice([True, False], size=7000)
    y, idx, count = gen_array_ops.unique_with_counts_v2(
        x, axis=np.array([], np.int32))
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      self.assertEqual(count, np.sum(x == value))

  def testFloat(self):
    # NOTE(mrry): The behavior when a key is NaN is inherited from
    # `std::unordered_map<float, ...>`: each NaN becomes a unique key in the
    # map.
    x = [0.0, 1.0, np.nan, np.nan]
    y, idx, count = gen_array_ops.unique_with_counts_v2(
        x, axis=np.array([], np.int32))
    tf_y, tf_idx, tf_count = self.evaluate([y, idx, count])

    self.assertEqual(len(x), len(tf_idx))
    self.assertEqual(len(tf_y), len(np.unique(x)))
    for i in range(len(x)):
      if np.isnan(x[i]):
        self.assertTrue(np.isnan(tf_y[tf_idx[i]]))
      else:
        self.assertEqual(x[i], tf_y[tf_idx[i]])
    for value, count in zip(tf_y, tf_count):
      if np.isnan(value):
        self.assertEqual(count, 1)
      else:
        self.assertEqual(count, np.sum(x == value))


if __name__ == '__main__':
  test.main()
