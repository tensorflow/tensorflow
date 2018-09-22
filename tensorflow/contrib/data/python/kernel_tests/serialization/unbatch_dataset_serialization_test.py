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
"""Tests for the UnbatchDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class UnbatchDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def build_dataset(self, multiplier=15.0, tensor_slice_len=2, batch_size=2):
    components = (
        np.arange(tensor_slice_len),
        np.array([[1, 2, 3]]) * np.arange(tensor_slice_len)[:, np.newaxis],
        np.array(multiplier) * np.arange(tensor_slice_len))

    return dataset_ops.Dataset.from_tensor_slices(components).batch(
        batch_size).apply(batching.unbatch())

  def testCore(self):
    tensor_slice_len = 8
    batch_size = 2
    num_outputs = tensor_slice_len
    self.run_core_tests(
        lambda: self.build_dataset(15.0, tensor_slice_len, batch_size),
        lambda: self.build_dataset(20.0, tensor_slice_len, batch_size),
        num_outputs)


if __name__ == "__main__":
  test.main()
