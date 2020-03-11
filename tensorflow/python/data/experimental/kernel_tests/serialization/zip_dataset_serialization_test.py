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
"""Tests for the ZipDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class ZipDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_dataset(self, arr):
    components = [
        np.tile(np.array([[1], [2], [3], [4]]), 20),
        np.tile(np.array([[12], [13], [14], [15]]), 22),
        np.array(arr)
    ]
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(component)
        for component in components
    ]
    return dataset_ops.Dataset.zip((datasets[0], (datasets[1], datasets[2])))

  @combinations.generate(test_base.default_test_combinations())
  def testCore(self):
    # Equal length components
    arr = [37.0, 38.0, 39.0, 40.0]
    num_outputs = len(arr)
    self.run_core_tests(lambda: self._build_dataset(arr), num_outputs)
    # Variable length components
    diff_size_arr = [1.0, 2.0]
    self.run_core_tests(lambda: self._build_dataset(diff_size_arr), 2)


if __name__ == "__main__":
  test.main()
