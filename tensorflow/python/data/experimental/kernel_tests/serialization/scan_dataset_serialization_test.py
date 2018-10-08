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
"""Tests for the ScanDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test


class ScanDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_dataset(self, num_elements):
    return dataset_ops.Dataset.from_tensors(1).repeat(num_elements).apply(
        scan_ops.scan([0, 1], lambda a, _: ([a[1], a[0] + a[1]], a[1])))

  def testScanCore(self):
    num_output = 5
    self.run_core_tests(lambda: self._build_dataset(num_output),
                        lambda: self._build_dataset(2), num_output)


if __name__ == "__main__":
  test.main()
