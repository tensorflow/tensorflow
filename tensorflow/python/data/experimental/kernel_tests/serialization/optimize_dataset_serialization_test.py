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
"""Tests for the OptimizeDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class OptimizeDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def testCore(self):

    def build_dataset(num_elements, batch_size):
      return dataset_ops.Dataset.range(num_elements).map(lambda x: x * x).batch(
          batch_size).apply(optimization.optimize(["map_and_batch_fusion"]))

    self.run_core_tests(lambda: build_dataset(200, 10), None, 20)

  def testWithNewFunction(self):
    """Tests that optimized datasets with new functions work."""

    def build_dataset():
      dataset = dataset_ops.Dataset.range(100)
      dataset = dataset.map(lambda x: x)
      dataset = dataset.batch(5)
      # map_vectorization adds a new vectorized function to the function
      # library.
      dataset = dataset.apply(optimization.optimize(["map_vectorization"]))
      return dataset

    self.run_core_tests(build_dataset, None, 20)


if __name__ == "__main__":
  test.main()
