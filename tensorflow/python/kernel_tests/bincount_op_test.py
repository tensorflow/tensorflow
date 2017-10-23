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
"""Tests for math_ops.bincount."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

class BincountTest(test_util.TensorFlowTestCase):

  def test_empty(self):
    with self.test_session(use_gpu=True):
      self.assertAllEqual(
          math_ops.bincount([], minlength=5).eval(), [0, 0, 0, 0, 0])
      self.assertAllEqual(math_ops.bincount([], minlength=1).eval(), [0])
      self.assertAllEqual(math_ops.bincount([], minlength=0).eval(), [])
      self.assertEqual(
          math_ops.bincount([], minlength=0, dtype=np.float32).eval().dtype,
          np.float32)
      self.assertEqual(
          math_ops.bincount([], minlength=3, dtype=np.float64).eval().dtype,
          np.float64)

  def test_values(self):
    with self.test_session(use_gpu=True):
      self.assertAllEqual(
          math_ops.bincount([1, 1, 1, 2, 2, 3]).eval(), [0, 3, 2, 1])
      arr = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5]
      self.assertAllEqual(math_ops.bincount(arr).eval(), [0, 5, 4, 3, 2, 1])
      arr += [0, 0, 0, 0, 0, 0]
      self.assertAllEqual(math_ops.bincount(arr).eval(), [6, 5, 4, 3, 2, 1])

      self.assertAllEqual(math_ops.bincount([]).eval(), [])
      self.assertAllEqual(math_ops.bincount([0, 0, 0]).eval(), [3])
      self.assertAllEqual(math_ops.bincount([5]).eval(), [0, 0, 0, 0, 0, 1])
      self.assertAllEqual(
          math_ops.bincount(np.arange(10000)).eval(), np.ones(10000))

  def test_maxlength(self):
    with self.test_session(use_gpu=True):
      self.assertAllEqual(math_ops.bincount([5], maxlength=3).eval(), [0, 0, 0])
      self.assertAllEqual(math_ops.bincount([1], maxlength=3).eval(), [0, 1])
      self.assertAllEqual(math_ops.bincount([], maxlength=3).eval(), [])

  def test_random_with_weights(self):
    num_samples = 10000
    with self.test_session(use_gpu=True):
      np.random.seed(42)
      for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
        arr = np.random.randint(0, 1000, num_samples)
        if dtype == dtypes.int32 or dtype == dtypes.int64:
          weights = np.random.randint(-100, 100, num_samples)
        else:
          weights = np.random.random(num_samples)
        self.assertAllClose(
            math_ops.bincount(arr, weights).eval(),
            np.bincount(arr, weights))

  def test_random_without_weights(self):
    num_samples = 10000
    with self.test_session(use_gpu=True):
      np.random.seed(42)
      for dtype in [np.int32, np.float32]:
        arr = np.random.randint(0, 1000, num_samples)
        weights = np.ones(num_samples).astype(dtype)
        self.assertAllClose(
            math_ops.bincount(arr, None).eval(),
            np.bincount(arr, weights))

  def test_zero_weights(self):
    with self.test_session(use_gpu=True):
      self.assertAllEqual(
          math_ops.bincount(np.arange(1000), np.zeros(1000)).eval(),
          np.zeros(1000))

  def test_negative(self):
    # unsorted_segment_sum will only report InvalidArgumentError on CPU
    with self.test_session():
      with self.assertRaises(errors.InvalidArgumentError):
        math_ops.bincount([1, 2, 3, -1, 6, 8]).eval()


if __name__ == "__main__":
  googletest.main()
