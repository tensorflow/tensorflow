# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for UnravelIndex, previously unsupported under XLA (b/no XLA_CPU_JIT kernel)."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class UnravelIndexOpTest(xla_test.XLATestCase):

  def testVectorIndices(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.session():
        with self.test_scope():
          indices = array_ops.placeholder(dtype, shape=[4])
          dims = array_ops.constant([2, 2], dtype=dtype)
          o = array_ops.unravel_index(indices, dims)
        result = o.eval(feed_dict={indices: [0, 1, 2, 3]})
        self.assertAllEqual(np.unravel_index([0, 1, 2, 3], (2, 2)), result)

  def testScalarIndex(self):
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.session():
        with self.test_scope():
          index = array_ops.placeholder(dtype, shape=[])
          dims = array_ops.constant([3, 4, 5], dtype=dtype)
          o = array_ops.unravel_index(index, dims)
        result = o.eval(feed_dict={index: 37})
        self.assertAllEqual(np.unravel_index(37, (3, 4, 5)), result)

  def testHigherRank(self):
    with self.session():
      with self.test_scope():
        indices = array_ops.constant([22, 41, 37])
        dims = array_ops.constant([7, 6])
        o = array_ops.unravel_index(indices, dims)
      result = o.eval()
      self.assertAllEqual(np.unravel_index([22, 41, 37], (7, 6)), result)


if __name__ == '__main__':
  googletest.main()
