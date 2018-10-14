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
"""Tests for the SqlDataset serialization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.data.experimental.kernel_tests import sql_dataset_test_base
from tensorflow.python.data.experimental.kernel_tests.serialization import dataset_serialization_test_base
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class SqlDatasetSerializationTest(
    sql_dataset_test_base.SqlDatasetTestBase,
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_dataset(self, num_repeats):
    data_source_name = os.path.join(test.get_temp_dir(), "tftest.sqlite")
    driver_name = array_ops.placeholder_with_default(
        array_ops.constant("sqlite", dtypes.string), shape=[])
    query = ("SELECT first_name, last_name, motto FROM students ORDER BY "
             "first_name DESC")
    output_types = (dtypes.string, dtypes.string, dtypes.string)
    return readers.SqlDataset(driver_name, data_source_name, query,
                              output_types).repeat(num_repeats)

  def testSQLSaveable(self):
    num_repeats = 4
    num_outputs = num_repeats * 2
    self.run_core_tests(lambda: self._build_dataset(num_repeats),
                        lambda: self._build_dataset(num_repeats // 2),
                        num_outputs)


if __name__ == "__main__":
  test.main()
