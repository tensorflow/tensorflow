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
"""Tests for the sort wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.framework.python.ops import sort_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class SortTest(test.TestCase):

  def testRandom_lowDimensionality(self):
    self._testRandom_lowDimensionality(negative_axis=False)

  def testRandom_lowDimensionality_negative(self):
    self._testRandom_lowDimensionality(negative_axis=True)

  def _testRandom_lowDimensionality(self, negative_axis):
    np.random.seed(42)
    for _ in range(20):
      rank = np.random.randint(1, 3)
      shape = [np.random.randint(0, 20) for _ in range(rank)]
      arr = np.random.random(shape)
      sort_axis = np.random.choice(rank)
      if negative_axis:
        sort_axis = -1 - sort_axis
      with self.test_session():
        self.assertAllEqual(
            np.sort(arr, axis=sort_axis),
            sort_ops.sort(constant_op.constant(arr), axis=sort_axis).eval())

  def testRandom_highDimensionality(self):
    np.random.seed(100)
    for _ in range(20):
      rank = np.random.randint(5, 15)
      shape = [np.random.randint(1, 4) for _ in range(rank)]
      arr = np.random.random(shape)
      sort_axis = np.random.choice(rank)
      with self.test_session():
        self.assertAllEqual(
            np.sort(arr, axis=sort_axis),
            sort_ops.sort(constant_op.constant(arr), axis=sort_axis).eval())

  def testScalar(self):
    # Create an empty scalar where the static shape is unknown.
    zeros_length_1 = array_ops.zeros(
        random_ops.random_uniform([1], minval=0, maxval=1, dtype=dtypes.int32),
        dtype=dtypes.int32)
    scalar = array_ops.zeros(zeros_length_1)

    sort = sort_ops.sort(scalar)
    with self.test_session():
      with self.assertRaises(errors.InvalidArgumentError):
        sort.eval()

  def testNegativeOutOfBounds_staticShape(self):
    arr = constant_op.constant([3, 4, 5])
    with self.assertRaises(ValueError):
      sort_ops.sort(arr, axis=-4)

  def testDescending(self):
    arr = np.random.random((10, 5, 5))
    with self.test_session():
      self.assertAllEqual(
          np.sort(arr, axis=0)[::-1],
          sort_ops.sort(
              constant_op.constant(arr),
              axis=0,
              direction='DESCENDING').eval())


if __name__ == '__main__':
  test.main()
