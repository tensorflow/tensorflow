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
"""Tests for feature_column utils."""

from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


class UtilsTest(test.TestCase):

  def test_sequence_length_from_sparse_tensor_unsorted(self):
    # This tensor has unsorted row_ids. It should not cause memory allocation
    # errors or SegmentMax errors. It should be handled gracefully (e.g., via
    # reorder or unsorted_segment_max).
    sp_tensor = sparse_tensor.SparseTensor(
        indices=[[2, 0], [0, 0], [1, 0]],
        values=[1, 1, 1],
        dense_shape=[3, 1])

    with self.cached_session():
      res = fc_utils.sequence_length_from_sparse_tensor(sp_tensor)
      # Before the fix, this evaluates to an InvalidArgumentError on CPU,
      # and causes a heap buffer overflow on GPU.
      # After the fix, it should return the correct sequence lengths
      # [1, 1, 1].
      seq_lengths = self.evaluate(res)
      self.assertAllEqual([1, 1, 1], seq_lengths)

if __name__ == "__main__":
  test.main()
