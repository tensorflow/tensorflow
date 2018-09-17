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
"""Tests for the PaddedBatchDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class PaddedBatchDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def testPaddedBatch(self):

    def build_dataset(seq_lens):
      return dataset_ops.Dataset.from_tensor_slices(seq_lens).map(
          lambda x: array_ops.fill([x], x)).padded_batch(
              4, padded_shapes=[-1])

    seq_lens1 = np.random.randint(1, 20, size=(32,)).astype(np.int32)
    seq_lens2 = np.random.randint(21, 40, size=(32,)).astype(np.int32)
    self.run_core_tests(lambda: build_dataset(seq_lens1),
                        lambda: build_dataset(seq_lens2), 8)

  def testPaddedBatchNonDefaultPadding(self):

    def build_dataset(seq_lens):

      def fill_tuple(x):
        filled = array_ops.fill([x], x)
        return (filled, string_ops.as_string(filled))

      padded_shape = [-1]
      return dataset_ops.Dataset.from_tensor_slices(seq_lens).map(
          fill_tuple).padded_batch(
              4,
              padded_shapes=(padded_shape, padded_shape),
              padding_values=(-1, "<end>"))

    seq_lens1 = np.random.randint(1, 20, size=(32,)).astype(np.int32)
    seq_lens2 = np.random.randint(21, 40, size=(32,)).astype(np.int32)
    self.run_core_tests(lambda: build_dataset(seq_lens1),
                        lambda: build_dataset(seq_lens2), 8)


if __name__ == "__main__":
  test.main()
