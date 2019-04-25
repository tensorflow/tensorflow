# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the ShardDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class ShardDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_dataset(self, num_elements, num_shards, index):
    return dataset_ops.Dataset.range(num_elements).shard(num_shards, index)

  @parameterized.parameters((10, 5, 2, 3), (10, 10, 0, 9), (100, 2, 0, 1))
  def testCore(self, elems, num_shards, index1, index2):
    self.run_core_tests(lambda: self._build_dataset(elems, num_shards, index1),
                        lambda: self._build_dataset(elems, num_shards, index2),
                        elems // num_shards)


if __name__ == "__main__":
  test.main()
