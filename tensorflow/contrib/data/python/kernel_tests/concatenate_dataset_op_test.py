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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class ConcatenateDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_concatenate_dataset(self, var_array):
    input_components = (np.tile(np.array([[1], [2], [3], [4]]), 20),
                        np.tile(np.array([[12], [13], [14], [15]]), 4))
    to_concatenate_components = (np.tile(
        np.array([[5], [6], [7], [8], [9]]), 20), var_array)

    return dataset_ops.Dataset.from_tensor_slices(input_components).concatenate(
        dataset_ops.Dataset.from_tensor_slices(to_concatenate_components))

  def testConcatenateCore(self):
    num_outputs = 9
    array = np.tile(np.array([[16], [17], [18], [19], [20]]), 15)
    diff_array = np.array([[1], [2], [3], [4], [5]])
    self.run_core_tests(lambda: self._build_concatenate_dataset(array),
                        lambda: self._build_concatenate_dataset(diff_array),
                        num_outputs)


if __name__ == "__main__":
  test.main()
