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
"""Tests for the TextLineDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.platform import test


class TextLineDatasetSerializationTest(
    reader_dataset_ops_test_base.TextLineDatasetTestBase,
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_iterator_graph(self, test_filenames, compression_type=None):
    return core_readers.TextLineDataset(
        test_filenames, compression_type=compression_type, buffer_size=10)

  def testTextLineCore(self):
    compression_types = [None, "GZIP", "ZLIB"]
    num_files = 5
    lines_per_file = 5
    num_outputs = num_files * lines_per_file
    for compression_type in compression_types:
      test_filenames = self._createFiles(
          num_files,
          lines_per_file,
          crlf=True,
          compression_type=compression_type)
      # pylint: disable=cell-var-from-loop
      self.run_core_tests(
          lambda: self._build_iterator_graph(test_filenames, compression_type),
          lambda: self._build_iterator_graph(test_filenames), num_outputs)
      # pylint: enable=cell-var-from-loop


if __name__ == "__main__":
  test.main()
