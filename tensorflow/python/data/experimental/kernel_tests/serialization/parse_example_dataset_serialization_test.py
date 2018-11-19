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
"""Tests for the ParseExampleDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.platform import test


class ParseExampleDatasetSerializationTest(
    reader_dataset_ops_test_base.MakeBatchedFeaturesDatasetTestBase,
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def ParseExampleDataset(self, num_repeat, batch_size):
    return self.make_batch_feature(
        filenames=self.test_filenames,
        num_epochs=num_repeat,
        batch_size=batch_size,
        reader_num_threads=5,
        parser_num_threads=10)

  def testSerializationCore(self):
    num_repeat = 5
    batch_size = 2
    num_outputs = self._num_records * self._num_files * num_repeat // batch_size
    # pylint: disable=g-long-lambda
    self.run_core_tests(
        lambda: self.ParseExampleDataset(
            num_repeat=num_repeat, batch_size=batch_size),
        lambda: self.ParseExampleDataset(num_repeat=10, batch_size=4),
        num_outputs)


if __name__ == "__main__":
  test.main()
