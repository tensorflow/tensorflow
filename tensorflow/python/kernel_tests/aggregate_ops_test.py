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
"""Tests for aggregate_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class AddNTest(test.TestCase):
  # AddN special-cases adding the first M inputs to make (N - M) divisible by 8,
  # after which it adds the remaining (N - M) tensors 8 at a time in a loop.
  # Test N in [1, 10] so we check each special-case from 1 to 9 and one
  # iteration of the loop.
  _MAX_N = 10

  def _supported_types(self):
    if test.is_gpu_available():
      return [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
              dtypes.complex128]
    return [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
            dtypes.float16, dtypes.float32, dtypes.float64, dtypes.complex64,
            dtypes.complex128]

  def _buildData(self, shape, dtype):
    data = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testAddN(self):
    np.random.seed(12345)
    with self.test_session(use_gpu=True) as sess:
      for dtype in self._supported_types():
        for count in range(1, self._MAX_N + 1):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = sess.run(math_ops.add_n(data))
          expected = np.sum(np.vstack(
              [np.expand_dims(d, 0) for d in data]), axis=0)
          tol = 5e-3 if dtype == dtypes.float16 else 5e-7
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)

  def testUnknownShapes(self):
    np.random.seed(12345)
    with self.test_session(use_gpu=True) as sess:
      for dtype in self._supported_types():
        data = self._buildData((2, 2), dtype)
        for count in range(1, self._MAX_N + 1):
          data_ph = array_ops.placeholder(dtype=dtype)
          actual = sess.run(math_ops.add_n([data_ph] * count), {data_ph: data})
          expected = np.sum(np.vstack([np.expand_dims(data, 0)] * count),
                            axis=0)
          tol = 5e-3 if dtype == dtypes.float16 else 5e-7
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)


if __name__ == "__main__":
  test.main()
