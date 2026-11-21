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
from tensorflow.python.framework import errors
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

  def test_index_table_from_dataset_vocab_size_exceeds_cardinality(self):
    data = ["apple", "banana", "orange", "grape", "kiwi", "mango"]
    ds = dataset_ops.Dataset.from_tensor_slices(data)
    for vocab_size in (
        10,
        2**31 - 1,
        2**40,
        2**50,
        dtypes.int64.max,
        constant_op.constant(2**40, dtype=dtypes.int64),
    ):
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "Input dataset was expected to contain .* but contained only "
          "6 elements.",
      ):
        lookup_ops.index_table_from_dataset(
            dataset=ds,
            vocab_size=vocab_size,
            num_oov_buckets=1,
            default_value=-1,
        )

  def test_table_from_dataset_vocab_size_exceeds_cardinality(self):
    keys = dataset_ops.Dataset.from_tensor_slices(["a", "b"])
    values = dataset_ops.Dataset.from_tensor_slices([0, 1])
    ds = dataset_ops.Dataset.zip((keys, values))
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "Input dataset was expected to contain .* but contained only "
        "2 elements.",
    ):
      lookup_ops.table_from_dataset(ds, vocab_size=2**40)

    with self.assertRaisesRegex(ValueError, "must not exceed"):
      lookup_ops.table_from_dataset(ds, vocab_size=dtypes.int64.max + 1)

  def test_index_table_from_dataset_empty_with_vocab_size(self):
    ds = dataset_ops.Dataset.from_tensor_slices(
        constant_op.constant([], dtype=dtypes.string)
    )
    for vocab_size in (1, 2**40):
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          "contained only 0 elements.",
      ):
        lookup_ops.index_table_from_dataset(
            dataset=ds, vocab_size=vocab_size, default_value=-1
        )

  def test_index_table_from_dataset_valid_vocab_size(self):
    data = ["apple", "banana", "orange"]
    ds = dataset_ops.Dataset.from_tensor_slices(data)

    # Exact cardinality match.
    table_exact = lookup_ops.index_table_from_dataset(
        ds, vocab_size=3, default_value=-1
    )
    self.evaluate(core_lookup_ops.tables_initializer())
    self.assertAllEqual(
        [0, 1, 2, -1],
        self.evaluate(
            table_exact.lookup(
                constant_op.constant(["apple", "banana", "orange", "kiwi"])
            )
        ),
    )

    # Truncated vocabulary (vocab_size < cardinality).
    table_trunc = lookup_ops.index_table_from_dataset(
        ds, vocab_size=2, default_value=-1
    )
    self.evaluate(core_lookup_ops.tables_initializer())
    self.assertAllEqual(
        [0, 1, -1],
        self.evaluate(
            table_trunc.lookup(
                constant_op.constant(["apple", "banana", "orange"])
            )
        ),
    )

    # Infinite dataset with finite vocab_size.
    ds_inf = dataset_ops.Dataset.range(10).map(string_ops.as_string).repeat()
    table_inf = lookup_ops.index_table_from_dataset(
        ds_inf, vocab_size=3, default_value=-1
    )
    self.evaluate(core_lookup_ops.tables_initializer())
    self.assertAllEqual(
        [0, 1, 2, -1],
        self.evaluate(
            table_inf.lookup(constant_op.constant(["0", "1", "2", "3"]))
        ),
    )

    # Unknown cardinality dataset with matching and mismatched vocab_size.
    ds_unknown = ds.filter(lambda _: True)
    table_unknown = lookup_ops.index_table_from_dataset(
        ds_unknown, vocab_size=3, default_value=-1
    )
    self.evaluate(core_lookup_ops.tables_initializer())
    self.assertAllEqual(
        [0, 1, 2],
        self.evaluate(
            table_unknown.lookup(
                constant_op.constant(["apple", "banana", "orange"])
            )
        ),
    )
    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        "Input dataset was expected to contain 4 elements but contained only "
        "3 elements.",
    ):
      lookup_ops.index_table_from_dataset(
          ds_unknown, vocab_size=4, default_value=-1
      )

  def test_index_table_from_dataset_oversized_vocab_size_graph_mode(self):
    for use_tensor in (False, True):
      with ops.Graph().as_default() as g, self.session(graph=g):
        ds = dataset_ops.Dataset.from_tensor_slices(["apple", "banana"])
        vocab_size = (
            constant_op.constant(2**40, dtype=dtypes.int64)
            if use_tensor
            else 2**40
        )
        lookup_ops.index_table_from_dataset(
            dataset=ds,
            vocab_size=vocab_size,
            num_oov_buckets=1,
            default_value=-1,
        )
        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            "Input dataset was expected to contain",
        ):
          self.evaluate(core_lookup_ops.tables_initializer())


if __name__ == "__main__":
  test.main()
