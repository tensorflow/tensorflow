# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for ParquetDataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.parquet.python.ops import parquet_dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test

class ParquetDatasetTest(test.TestCase):

  def test_parquet_dataset(self):
    """Test case for ParquetDataset.

    Note: The sample file is generated from:
    `parquet-cpp/examples/low-level-api/reader_writer`
    This test extracts columns of [0, 1, 2, 4, 5]
    with column data types of [bool, int32, int64, float, double].
    Please check `parquet-cpp/examples/low-level-api/reader-writer.cc`
    to find details of how records are generated:
    Column 0 (bool): True for even rows and False otherwise.
    Column 1 (int32): Equal to row_index.
    Column 2 (int64): Equal to row_index * 1000 * 1000 * 1000 * 1000.
    Column 4 (float): Equal to row_index * 1.1.
    Column 5 (double): Equal to row_index * 1.1111111.
    """
    filename = os.path.join(
        resource_loader.get_data_files_path(),
        'testdata/parquet_cpp_example.parquet')

    filenames = constant_op.constant([filename], dtypes.string)
    columns = [0, 1, 2, 4, 5]
    output_types = (
        dtypes.bool, dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64)
    num_repeats = 2

    dataset = parquet_dataset_ops.ParquetDataset(
        filenames, columns, output_types).repeat(num_repeats)
    iterator = dataset.make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.test_session() as sess:
      sess.run(init_op)
      for _ in range(num_repeats):  # Dataset is repeated.
        for i in range(500): # 500 rows.
          v0 = True if ((i % 2) == 0) else False
          v1 = i
          v2 = i * 1000 * 1000 * 1000 * 1000
          v4 = 1.1 * i
          v5 = 1.1111111 * i
          self.assertAllClose((v0, v1, v2, v4, v5), sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

if __name__ == "__main__":
  test.main()
