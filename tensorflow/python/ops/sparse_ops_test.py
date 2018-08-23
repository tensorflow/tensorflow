# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sparse ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class SparseOpsTest(test_util.TensorFlowTestCase):

  def testSparseEye(self):
    def test_one(n, m, as_tensors):
      expected = np.eye(n, m)
      if as_tensors:
        m = constant_op.constant(m)
        n = constant_op.constant(n)
      s = sparse_ops.sparse_eye(n, m)
      d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
      self.assertAllEqual(self.evaluate(d), expected)

    for n in range(2, 10, 2):
      for m in range(2, 10, 2):
        # Test with n and m as both constants and tensors.
        test_one(n, m, True)
        test_one(n, m, False)

if __name__ == '__main__':
  googletest.main()
