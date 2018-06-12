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


"""
This Module is a test file
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.ops import sparse_tile_like_op
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest
from tensorflow.python.framework.sparse_tensor import SparseTensor


import numpy as np


class SparseOpsTest(TensorFlowTestCase):

  """Test
  """

  def test_sparse_tile(self):
    with self.test_session():
      a_indices = np.array([[0, 1], [0, 3], [1, 2], [1, 3], [0, 2]],
                           dtype=np.int64)
      a_values = np.array([6, 7, 8, 9, 7], dtype=np.float32)
      a_shape = np.array([3, 5], dtype=np.int64)
      b_indices = np.array([[0, 1, 2], [0, 1, 3], [0, 3, 0], [1, 2, 3],
                            [1, 3, 2], [0, 2, 1]], dtype=np.int64)
      b_values = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
      b_shape = np.array([3, 5, 4], dtype=np.int64)
      sp_a = SparseTensor(a_indices, a_values, a_shape)
      sp_b = SparseTensor(b_indices, b_values, b_shape)
      result = sparse_tile_like_op.sparse_tile_like(sp_a, sp_b, axes=2)
      r_indices, r_values, r_shape = result.eval()
      self.assertAllEqual(r_indices, np.array([[0, 1, 2], [0, 1, 3],
                                               [0, 2, 1], [0, 3, 0],
                                               [1, 2, 3], [1, 3, 2]],
                                              dtype=np.int64))
      self.assertAllEqual(r_values, np.array([6, 6, 7, 8, 9, 7],
                                             dtype=np.float32))
      self.assertAllEqual(r_shape, b_shape)


if __name__ == "__main__":
  googletest.main()
