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
"""Tests for tensorflow.python.data.experimental.ops.lookup_ops."""

import os

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import lookup_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as reader_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops as core_lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class DatasetInitializerTest(test.TestCase):

  def getHashTable(self):
    if tf2.enabled():
      return core_lookup_ops.StaticHashTable
    else:
      return core_lookup_ops.StaticHashTableV1

  def initialize_table(self, table):
    if not tf2.enabled():
      self.evaluate(table.initializer)

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_basic(self):
    keys = dataset_ops.Dataset.range(100)
    values = dataset_ops.Dataset.range(100).map(
        lambda x: string_ops.as_string(x * 2))
    ds = dataset_ops.Dataset.zip((keys, values))
    init = lookup_ops.DatasetInitializer(ds)
    table = self.getHashTable()(init, default_value="")
    self.initialize_table(table)

    output = table.lookup(constant_op.constant([0, 2, 5], dtypes.int64))
    result = self.evaluate(output)
    self.assertAllEqual(["0", "4", "10"], result)

  def test_basic_bad_shape(self):
    keys = dataset_ops.Dataset.range(100)
    values = dataset_ops.Dataset.range(100).map(
        lambda x: string_ops.as_string(x * 2))
    values = values.batch(4)
    ds = dataset_ops.Dataset.zip((keys, values))
    with self.assertRaises(ValueError):
      lookup_ops.DatasetInitializer(ds)

  def test_from_file(self):
    vocabulary_file = self._createVocabFile("test.txt", ("one", "two", "three"))
    ds = reader_ops.TextLineDataset(vocabulary_file)
    ds = ds.enumerate(start=1)
    init = lookup_ops.DatasetInitializer(ds)
    table = self.getHashTable()(init, default_value="")
    self.initialize_table(table)

    output = table.lookup(constant_op.constant([2, 3, 4], dtypes.int64))
    result = self.evaluate(output)
    self.assertAllEqual(["two", "three", ""], result)

  def test_from_multiple_files(self):
    vocabulary_file1 = self._createVocabFile("test1.txt",
                                             ("one", "two", "three"))
    vocabulary_file2 = self._createVocabFile("test2.txt",
                                             ("four", "five", "six"))
    ds = reader_ops.TextLineDataset([vocabulary_file1, vocabulary_file2])
    ds = ds.enumerate(start=1)
    init = lookup_ops.DatasetInitializer(ds)
    table = self.getHashTable()(init, default_value="")
    self.initialize_table(table)

    output = table.lookup(constant_op.constant([2, 3, 4], dtypes.int64))
    result = self.evaluate(output)
    self.assertAllEqual(["two", "three", "four"], result)

  def test_map_variable(self):
    ds = dataset_ops.Dataset.range(100)
    captured_var = variables.Variable(0)

    def func(_):
      return captured_var.assign_add(1)

    ds = ds.map(func)
    ds = ds.enumerate(start=1)
    init = lookup_ops.DatasetInitializer(ds)
    table = self.getHashTable()(init, default_value=-1)
    self.evaluate(captured_var.initializer)
    self.initialize_table(table)

    output = table.lookup(constant_op.constant([1, 2, 101], dtypes.int64))
    result = self.evaluate(output)
    self.assertAllEqual([1, 2, -1], result)

  def test_compatibility(self):
    with ops.Graph().as_default():
      keys = dataset_ops.Dataset.range(100)
      values = dataset_ops.Dataset.range(100).map(string_ops.as_string)
      ds = dataset_ops.Dataset.zip((keys, values))
      init = lookup_ops.DatasetInitializer(ds)
      table = self.getHashTable()(init, default_value="")
      output = table.lookup(constant_op.constant([0, 2, 5], dtypes.int64))
      self.evaluate(core_lookup_ops.tables_initializer())
      result = self.evaluate(output)
    self.assertAllEqual(["0", "2", "5"], result)

  def test_table_from_dataset(self):
    keys = dataset_ops.Dataset.from_tensor_slices([2, 3, 4])
    values = dataset_ops.Dataset.from_tensor_slices(["two", "three", "four"])
    ds = dataset_ops.Dataset.zip((keys, values))
    table = lookup_ops.table_from_dataset(
        ds, default_value="n/a", key_dtype=dtypes.int64)
    output = table.lookup(constant_op.constant([2, 3, 4], dtypes.int32))
    self.evaluate(core_lookup_ops.tables_initializer())
    result = self.evaluate(output)
    self.assertAllEqual(["two", "three", "four"], result)

  def test_index_table_from_dataset(self):
    ds = dataset_ops.Dataset.range(100).map(
        lambda x: string_ops.as_string(x * 2))
    table = lookup_ops.index_table_from_dataset(ds, key_dtype=dtypes.int64)
    output = table.lookup(
        constant_op.constant(["0", "2", "4"], dtype=dtypes.string))
    self.evaluate(core_lookup_ops.tables_initializer())
    result = self.evaluate(output)
    self.assertAllEqual([0, 1, 2], result)


if __name__ == "__main__":
  test.main()
