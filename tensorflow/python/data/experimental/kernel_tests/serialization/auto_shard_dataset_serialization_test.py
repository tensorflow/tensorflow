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
"""Tests for the _AutoShard dataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class AutoShardDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _record(self, f, r):
    return compat.as_bytes("Record %d of file %d" % (r, f))

  def _createFiles(self):
    filenames = []
    for i in range(10):
      fn = os.path.join(self.get_temp_dir(), "tf_record.%d.txt" % i)
      filenames.append(fn)
      writer = python_io.TFRecordWriter(fn)
      for j in range(10):
        writer.write(self._record(i, j))
      writer.close()
    return filenames

  def setUp(self):
    self._filenames = self._createFiles()

  def testCore(self):

    def build_dataset():
      dataset = dataset_ops.Dataset.list_files(self._filenames, shuffle=False)
      dataset = dataset.apply(
          interleave_ops.parallel_interleave(readers.TFRecordDataset, 10))
      dataset = distribute._AutoShardDataset(dataset, 5, 3)
      return dataset

    self.run_core_tests(build_dataset, None, 20)


if __name__ == "__main__":
  test.main()
