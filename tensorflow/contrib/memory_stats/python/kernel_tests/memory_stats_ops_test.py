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
"""Tests for memory statistics ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.memory_stats.python.ops import memory_stats_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class MemoryStatsOpsTest(test_util.TensorFlowTestCase):

  def testBytesLimit(self):
    # AllocatorStats.bytes_limit is set to zero for CPU allocators, so we skip
    # the check.
    if not test.is_gpu_available():
      return

    with self.test_session(use_gpu=True) as sess:
      bytes_limit = sess.run(memory_stats_ops.BytesLimit())
      self.assertLess(0, bytes_limit)

  # Tests the peak memory usage of the following computation.
  #   a   b
  #   | / |
  #   c   |
  #    \  |
  #     \ |
  #       d
  # The memory for matrix "a" can be reused for matrix "d". Therefore, this
  # computation needs space for only three matrix plus some small overhead.
  def testChainOfMatmul(self):
    # MaxBytesInUse is registerd on GPU only. See kernels/memory_stats_ops.cc.
    if not test.is_gpu_available():
      return

    with self.test_session(use_gpu=True) as sess:
      matrix_size = 64
      matrix_shape = tensor_shape.TensorShape([matrix_size, matrix_size])
      dtype = dtypes.float32
      matrix_size_in_bytes = matrix_shape.num_elements() * dtype.size
      a = random_ops.random_uniform(matrix_shape, dtype=dtype)
      b = random_ops.random_uniform(matrix_shape, dtype=dtype)
      c = math_ops.matmul(a, b)
      d = math_ops.matmul(c, b)
      sess.run(d)

      max_bytes_in_use = sess.run(memory_stats_ops.MaxBytesInUse())
      self.assertGreaterEqual(max_bytes_in_use, matrix_size_in_bytes * 3)
      self.assertLess(max_bytes_in_use, matrix_size_in_bytes * 4)


if __name__ == '__main__':
  test.main()
