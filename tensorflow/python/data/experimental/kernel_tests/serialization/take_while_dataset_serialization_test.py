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

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class TakeWhileDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_dataset(self, num_elements, upper_bound):
    return dataset_ops.Dataset.range(num_elements).apply(
        take_while_ops.take_while(lambda x: x < upper_bound))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(num_elements=[10, 23], upper_bound=[10, 23])))
  def testCore(self, num_elements, upper_bound):
    self.run_core_tests(lambda: self._build_dataset(num_elements, upper_bound),
                        min(num_elements, upper_bound))


if __name__ == "__main__":
  test.main()
