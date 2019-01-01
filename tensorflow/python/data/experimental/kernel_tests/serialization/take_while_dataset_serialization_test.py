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
"""Tests for the TakeWhileDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class TakeWhileDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_dataset(self, num_elements, upper_bound):
    return dataset_ops.Dataset.range(num_elements).apply(
              take_while_ops.take_while(lambda x: x < upper_bound))

  def testCore(self):
    def run_test(num_elem1, num_elem2, upper_bound):
        self.run_core_tests(lambda: self._build_dataset(num_elem1, upper_bound),
                            lambda: self._build_dataset(num_elem2, upper_bound), 
                            upper_bound)

    run_test(23, 10, 7)
    run_test(10, 50, 0)
    run_test(25, 30, 25)


if __name__ == "__main__":
  test.main()
