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
"""Tests for the PrefetchDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class PrefetchDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def build_dataset(self, seed):
    return dataset_ops.Dataset.range(100).prefetch(10).shuffle(
        buffer_size=10, seed=seed, reshuffle_each_iteration=False)

  def testCore(self):
    num_outputs = 100
    self.run_core_tests(lambda: self.build_dataset(10),
                        lambda: self.build_dataset(20), num_outputs)


if __name__ == "__main__":
  test.main()
