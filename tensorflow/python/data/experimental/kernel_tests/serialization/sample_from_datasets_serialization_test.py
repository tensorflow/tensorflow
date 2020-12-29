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
"""Tests for the SampleFromDatasets serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class SampleFromDatasetsSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase,
    parameterized.TestCase):

  def _build_dataset(self, probs, num_samples):
    dataset = interleave_ops.sample_from_datasets(
        [
            dataset_ops.Dataset.from_tensors(i).repeat(None)
            for i in range(len(probs))
        ],
        probs,
        seed=1813)
    return dataset.take(num_samples)

  @combinations.generate(test_base.default_test_combinations())
  def testSerializationCore(self):
    self.run_core_tests(lambda: self._build_dataset([0.5, 0.5], 100), 100)


if __name__ == "__main__":
  test.main()
