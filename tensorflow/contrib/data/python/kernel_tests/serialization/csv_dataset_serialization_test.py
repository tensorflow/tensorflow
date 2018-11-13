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
"""Tests for the CsvDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

from tensorflow.contrib.data.python.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import readers
from tensorflow.python.platform import test


class CsvDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def setUp(self):
    self._num_cols = 7
    self._num_rows = 10
    self._num_epochs = 14
    self._num_outputs = self._num_rows * self._num_epochs

    inputs = [
        ",".join(str(self._num_cols * j + i)
                 for i in range(self._num_cols))
        for j in range(self._num_rows)
    ]
    contents = "\n".join(inputs).encode("utf-8")

    self._filename = os.path.join(self.get_temp_dir(), "file.csv")
    self._compressed = os.path.join(self.get_temp_dir(),
                                    "comp.csv")  # GZip compressed

    with open(self._filename, "wb") as f:
      f.write(contents)
    with gzip.GzipFile(self._compressed, "wb") as f:
      f.write(contents)

  def ds_func(self, **kwargs):
    compression_type = kwargs.get("compression_type", None)
    if compression_type == "GZIP":
      filename = self._compressed
    elif compression_type is None:
      filename = self._filename
    else:
      raise ValueError("Invalid compression type:", compression_type)

    return readers.CsvDataset(filename, **kwargs).repeat(self._num_epochs)

  def testSerializationCore(self):
    defs = [[0]] * self._num_cols
    self.run_core_tests(
        lambda: self.ds_func(record_defaults=defs, buffer_size=2),
        lambda: self.ds_func(record_defaults=defs, buffer_size=12),
        self._num_outputs)


if __name__ == "__main__":
  test.main()
