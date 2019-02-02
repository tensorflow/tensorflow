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
"""Tests for lookup ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import numpy as np
import six

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training.checkpointable import util as checkpointable


class HashTableTest(test.TestCase):

  def testHashTable(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

      exported_keys_tensor, exported_values_tensor = table.export()

      self.assertItemsEqual([b"brain", b"salad", b"surgery"],
                            self.evaluate(exported_keys_tensor))
      self.assertItemsEqual([0, 1, 2], self.evaluate(exported_values_tensor))

  def testHashTableFindHighRank(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([["brain", "salad"],
                                           ["tank", "tarkus"]])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testHashTableInitWithPythonArrays(self):
    with self.cached_session():
      default_val = -1
      keys = ["brain", "salad", "surgery"]
      values = [0, 1, 2]
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(
              keys, values, value_dtype=dtypes.int64), default_val)
      self.evaluate(table.initializer)

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableInitWithNumPyArrays(self):
    with self.cached_session():
      default_val = -1
      keys = np.array(["brain", "salad", "surgery"], dtype=np.str)
      values = np.array([0, 1, 2], dtype=np.int64)
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testMultipleHashTables(self):
    with self.cached_session():

      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)

      table1 = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      table2 = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      table3 = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)

      self.evaluate(table1.initializer)
      self.evaluate(table2.initializer)
      self.evaluate(table3.initializer)
      self.assertAllEqual(3, self.evaluate(table1.size()))
      self.assertAllEqual(3, self.evaluate(table2.size()))
      self.assertAllEqual(3, self.evaluate(table3.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testHashTableWithTensorDefault(self):
    with self.cached_session():
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableWithSparseTensorInput(self):
    with self.cached_session():
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      sp_indices = [[0, 0], [0, 1], [1, 0]]
      sp_shape = [2, 2]
      input_tensor = sparse_tensor.SparseTensor(
          constant_op.constant(sp_indices, dtypes.int64),
          constant_op.constant(["brain", "salad", "tank"]),
          constant_op.constant(sp_shape, dtypes.int64))
      output = table.lookup(input_tensor)

      out_indices, out_values, out_shape = self.evaluate(output)

      self.assertAllEqual([0, 1, -1], out_values)
      self.assertAllEqual(sp_indices, out_indices)
      self.assertAllEqual(sp_shape, out_shape)

  def testSignatureMismatch(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      # Ref types do not produce a lookup signature mismatch.
      input_string_ref = variables.Variable("brain")
      self.evaluate(input_string_ref.initializer)
      self.assertEqual(0, self.evaluate(table.lookup(input_string_ref)))

      input_string = constant_op.constant([1, 2, 3], dtypes.int64)
      with self.assertRaises(TypeError):
        table.lookup(input_string)

      with self.assertRaises(TypeError):
        lookup_ops.HashTable(
            lookup_ops.KeyValueTensorInitializer(keys, values), "UNK")

  def testDTypes(self):
    with self.cached_session():
      default_val = -1
      with self.assertRaises(TypeError):
        lookup_ops.HashTable(
            lookup_ops.KeyValueTensorInitializer(["a"], [1], [dtypes.string],
                                                 dtypes.int64), default_val)

  @test_util.run_deprecated_v1
  def testNotInitialized(self):
    with self.cached_session():
      default_val = -1
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(["a"], [1],
                                               value_dtype=dtypes.int64),
          default_val)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      with self.assertRaisesOpError("Table not initialized"):
        self.evaluate(output)

  @test_util.run_deprecated_v1
  def testInitializeTwice(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      with self.assertRaisesOpError("Table already initialized"):
        self.evaluate(table.initializer)

  @test_util.run_deprecated_v1
  def testInitializationWithInvalidDimensions(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)

      with self.assertRaises(ValueError):
        lookup_ops.HashTable(
            lookup_ops.KeyValueTensorInitializer(keys, values), default_val)

  @test_util.run_deprecated_v1
  def testMultipleSessions(self):
    # Start a server
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.HashTable(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        name="t1")

    # Init the table in the first session.
    with session1:
      self.evaluate(table.initializer)
      self.assertAllEqual(3, self.evaluate(table.size()))

    # Init the table in the second session and verify that we do not get a
    # "Table already initialized" error.
    with session2:
      table.initializer.run()
      self.assertAllEqual(3, self.evaluate(table.size()))

  def testHashTableInt32String(self):
    with self.cached_session():
      default_val = "n/a"
      keys = constant_op.constant([0, 1, 2], dtypes.int32)
      values = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup_ops.HashTable(
          lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
      self.evaluate(table.initializer)

      input_tensor = constant_op.constant([0, 1, -1])
      output = table.lookup(input_tensor)

      result = self.evaluate(output)
      self.assertAllEqual([b"brain", b"salad", b"n/a"], result)


class IndexTableFromFile(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_string_index_table_from_file(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_multicolumn_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain\t300", "salad\t20", "surgery\t1"))
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file,
          num_oov_buckets=1,
          key_column_index=0,
          value_column_index=lookup_ops.TextFileIndex.LINE_NUMBER)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_multicolumn_file_custom_delimiter(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain 300", "salad 20", "surgery 1"))
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file,
          num_oov_buckets=1,
          key_column_index=0,
          value_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
          delimiter=" ")
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_file_tensor_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_file = constant_op.constant(vocabulary_file)
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))
      if not context.executing_eagerly():
        self.assertEqual(1,
                         len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  @test_util.run_v1_only("placeholder usage")
  def test_string_index_table_from_file_placeholder_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_placeholder = array_ops.placeholder(dtypes.string, [])
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_placeholder, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)

      feed_dict = {vocabulary_placeholder.name: vocabulary_file}
      lookup_ops.tables_initializer().run(feed_dict=feed_dict)
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))
      self.assertEqual(0,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  def test_int32_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab2.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file,
          num_oov_buckets=1,
          key_dtype=dtypes.int32)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int64_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file,
          num_oov_buckets=1,
          key_dtype=dtypes.int64)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_index_table_from_file_with_default_value(self):
    default_value = -42
    vocabulary_file = self._createVocabFile("f2i_vocab4.txt")
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, default_value), self.evaluate(ids))

  def test_index_table_from_file_with_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab5.txt")
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1000)
      ids = table.lookup(
          constant_op.constant(["salad", "surgery", "tarkus", "toccata"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual(
          (
              1,  # From vocabulary file.
              2,  # From vocabulary file.
              867,  # 3 + fingerprint("tarkus") mod 300.
              860),  # 3 + fingerprint("toccata") mod 300.
          self.evaluate(ids))

  def test_index_table_from_file_fails_with_empty_vocabulary_file_name(self):
    self.assertRaises(
        ValueError, lookup_ops.index_table_from_file, vocabulary_file="")

  def test_index_table_from_file_fails_with_empty_vocabulary(self):
    self.assertRaises(
        ValueError, lookup_ops.index_table_from_file, vocabulary_file=None)

  def test_index_table_from_file_str_fails_with_zero_size_vocabulary(self):
    vocabulary_file = self._createVocabFile("zero_vocab_str.txt")
    self.assertRaisesRegexp(
        ValueError,
        "vocab_size must be greater than 0, got 0. "
        "vocabulary_file: .*zero_vocab_str.txt",
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

  def test_index_table_from_file_tensor_fails_with_zero_size_vocabulary(self):
    vocabulary_file = constant_op.constant(
        self._createVocabFile("zero_vocab_tensor.txt"))
    self.assertRaisesRegexp(
        ValueError,
        "vocab_size must be greater than 0, got 0. "
        "vocabulary_file: .*zero_vocab_tensor.txt",
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

  def test_index_table_from_file_with_vocab_size_too_small(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=2)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, -1, -1), self.evaluate(ids))
      self.assertEqual(2, self.evaluate(table.size()))

  def test_index_table_from_file_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.cached_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Invalid vocab_size"):
        table = lookup_ops.index_table_from_file(
            vocabulary_file=vocabulary_file, vocab_size=4)
        self.evaluate(table.initializer)

  def test_index_table_from_file_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab8.txt")

    self.assertRaises(
        ValueError,
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, -1), self.evaluate(ids))
      self.assertEqual(3, self.evaluate(table.size()))

  def test_index_table_from_file_with_invalid_hashers(self):
    vocabulary_file = self._createVocabFile("invalid_hasher.txt")
    with self.cached_session():
      with self.assertRaises(TypeError):
        lookup_ops.index_table_from_file(
            vocabulary_file=vocabulary_file,
            vocab_size=3,
            num_oov_buckets=1,
            hasher_spec=1)

      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=3,
          num_oov_buckets=1,
          hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        constant_op.constant(["salad", "surgery", "tarkus"]))

  def test_index_table_from_file_table_ref_with_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab9.txt")
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      self.assertIsNotNone(table.resource_handle)

  def test_index_table_from_file_table_ref_without_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab10.txt")
    with self.cached_session():
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=0)
      self.assertIsNotNone(table.resource_handle)


class KeyValueTensorInitializerTest(test.TestCase):

  def test_string(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      self.assertEqual("", init._shared_name)
      table = lookup_ops.HashTable(init, default_value=-1)
      table.initializer.run()

  def test_multiple_tables(self):
    with ops.Graph().as_default(), self.cached_session():
      with ops.name_scope("table_scope"):
        init1 = lookup_ops.KeyValueTensorInitializer(
            ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string,
            dtypes.int64)
        self.assertEqual("", init1._shared_name)
        table1 = lookup_ops.HashTable(init1, default_value=-1)
        self.assertEquals("hash_table", table1.name)
        self.assertEquals("table_scope/hash_table",
                          table1.resource_handle.op.name)
        init2 = lookup_ops.KeyValueTensorInitializer(
            ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string,
            dtypes.int64)
        self.assertEqual("", init2._shared_name)
        table2 = lookup_ops.HashTable(init2, default_value=-1)
        self.assertEquals("hash_table_1", table2.name)
        self.assertEquals("table_scope/hash_table_1",
                          table2.resource_handle.op.name)

  def test_int64(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                  dtypes.int64, dtypes.int64)
      self.assertEqual("", init._shared_name)
      table = lookup_ops.HashTable(init, default_value=-1)
      table.initializer.run()

  def test_int32(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                  dtypes.int32, dtypes.int64)
      self.assertEqual("", init._shared_name)
      table = lookup_ops.HashTable(init, default_value=-1)
      with self.assertRaisesRegexp(errors_impl.OpError,
                                   "No OpKernel was registered"):
        table.initializer.run()


class IndexTableFromTensor(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_index_table_from_tensor_with_tensor_init(self):
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=("brain", "salad", "surgery"), num_oov_buckets=1)

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(
            table.lookup(constant_op.constant(("salad", "surgery", "tarkus"))))
    else:
      # Reinitializing a table in eager should work.
      table = lookup_ops.index_table_from_tensor(
          vocabulary_list=("brain", "salad", "surgery"), num_oov_buckets=1)
    self.evaluate(lookup_ops.tables_initializer())
    ids = table.lookup(constant_op.constant(("salad", "surgery", "tarkus")))
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int32_index_table_from_tensor_with_tensor_init(self):
    with self.cached_session():
      table = lookup_ops.index_table_from_tensor(
          vocabulary_list=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int32)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.FailedPreconditionError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int64_index_table_from_tensor_with_tensor_init(self):
    with self.cached_session():
      table = lookup_ops.index_table_from_tensor(
          vocabulary_list=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int64)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.FailedPreconditionError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_index_table_from_tensor_with_default_value(self):
    default_value = -42
    with self.cached_session():
      table = lookup_ops.index_table_from_tensor(
          vocabulary_list=["brain", "salad", "surgery"],
          default_value=default_value)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.FailedPreconditionError):
          self.evaluate(ids)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((1, 2, default_value), self.evaluate(ids))

  def test_index_table_from_tensor_missing_vocabulary_list(self):
    with self.cached_session():
      with self.assertRaisesRegexp(ValueError,
                                   "vocabulary_list must be specified"):
        lookup_ops.index_table_from_tensor(
            vocabulary_list=None, num_oov_buckets=1)

  def test_index_table_from_tensor_empty_vocabulary_list(self):
    with self.cached_session():
      with self.assertRaisesRegexp(
          errors_impl.OpError, "keys and values cannot be empty"):
        _ = lookup_ops.index_table_from_tensor(
            vocabulary_list=np.array([], dtype=np.str_), num_oov_buckets=1)
        self.evaluate(lookup_ops.tables_initializer())

  def test_index_table_from_tensor_with_invalid_hashers(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        lookup_ops.index_table_from_tensor(
            vocabulary_list=["brain", "salad", "surgery"],
            num_oov_buckets=1,
            hasher_spec=1)

      table = lookup_ops.index_table_from_tensor(
          vocabulary_list=["brain", "salad", "surgery"],
          num_oov_buckets=1,
          hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        constant_op.constant(["salad", "surgery", "tarkus"]))


class IndexToStringTableFromFileTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_index_to_string_table(self):
    vocabulary_path = self._createVocabFile("i2f_vocab1.txt")
    # vocabulary_file supports string and tensor
    type_funcs = [str, constant_op.constant]
    for type_func in type_funcs:
      vocabulary_file = type_func(vocabulary_path)
      with self.cached_session():
        table = lookup_ops.index_to_string_table_from_file(
            vocabulary_file=vocabulary_file)
        features = table.lookup(
            constant_op.constant([0, 1, 2, 3], dtypes.int64))
        if not context.executing_eagerly():
          with self.assertRaises(errors_impl.OpError):
            self.evaluate(features)
        self.evaluate(lookup_ops.tables_initializer())
        self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                            self.evaluate(features))

  def test_index_to_string_table_from_multicolumn_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain\t300", "salad\t20", "surgery\t1"))
    with self.cached_session():
      table = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file,
          key_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
          value_column_index=0)
      features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          self.evaluate(features))

  def test_index_to_string_table_from_multicolumn_file_custom_delimiter(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain 300", "salad 20", "surgery 1"))
    with self.cached_session():
      table = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file,
          key_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
          value_column_index=0,
          delimiter=" ")
      features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          self.evaluate(features))

  def test_index_to_string_table_with_default_value(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.cached_session():
      table = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          self.evaluate(features))

  def test_index_to_string_table_with_vocab_size_too_small(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.cached_session():
      table = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=2,
          default_value=default_value)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"salad", default_value, default_value),
                          self.evaluate(features))

  def test_index_to_string_table_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.cached_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Invalid vocab_size"):
        _ = lookup_ops.index_to_string_table_from_file(
            vocabulary_file=vocabulary_file, vocab_size=4)
        self.evaluate(lookup_ops.tables_initializer())

  def test_index_to_string_table_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.cached_session():
      table = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))

      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"salad", b"surgery", b"UNK"),
                          self.evaluate(features))


class IndexToStringTableFromTensorTest(test.TestCase):

  def test_index_to_string_table_from_tensor(self):
    with self.cached_session():
      vocabulary_list = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup_ops.index_to_string_table_from_tensor(
          vocabulary_list=vocabulary_list)

      indices = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      features = table.lookup(indices)
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          self.evaluate(features))

  def test_duplicate_entries(self):
    with self.cached_session():
      vocabulary_list = constant_op.constant(["hello", "hello"])
      table = lookup_ops.index_to_string_table_from_tensor(
          vocabulary_list=vocabulary_list)
      indices = constant_op.constant([0, 1, 4], dtypes.int64)
      features = table.lookup(indices)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"hello", b"hello", b"UNK"), self.evaluate(features))

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    with self.cached_session():
      vocabulary_list = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup_ops.index_to_string_table_from_tensor(
          vocabulary_list=vocabulary_list, default_value=default_value)
      indices = constant_op.constant([1, 2, 4], dtypes.int64)
      features = table.lookup(indices)
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          self.evaluate(features))


class InitializeTableFromFileOpTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  @test_util.run_in_graph_and_eager_modes
  def testInitializeStringTable(self):
    vocabulary_file = self._createVocabFile("one_column_1.txt")
    default_value = -1
    init = lookup_ops.TextFileInitializer(
        vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
        dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
    self.assertTrue("one_column_1.txt_-2_-1" in init._shared_name)
    table = lookup_ops.HashTable(init, default_value)
    self.evaluate(table.initializer)

    output = table.lookup(constant_op.constant(["brain", "salad", "tank"]))

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testInitializeInt64Table(self):
    vocabulary_file = self._createVocabFile(
        "one_column_int64.txt", values=("42", "1", "-1000"))

    with self.cached_session():
      default_value = -1
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.int64, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertTrue("one_column_int64.txt_-2_-1" in init._shared_name)
      table = lookup_ops.HashTable(init, default_value)
      self.evaluate(table.initializer)

      output = table.lookup(
          constant_op.constant((42, 1, 11), dtype=dtypes.int64))

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testInitializeIndexTable(self):
    vocabulary_file = self._createVocabFile("one_column_2.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup_ops.TextFileIndex.LINE_NUMBER
      value_index = lookup_ops.TextFileIndex.WHOLE_LINE
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.int64, key_index, dtypes.string, value_index)
      self.assertTrue("one_column_2.txt_-1_-2" in init._shared_name)
      table = lookup_ops.HashTable(init, default_value)
      self.evaluate(table.initializer)

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      output = table.lookup(input_values)

      result = self.evaluate(output)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], result)

  def testMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 1
      value_index = 2

      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertTrue("three_columns.txt_1_2" in init._shared_name)
      table = lookup_ops.HashTable(init, default_value)
      self.evaluate(table.initializer)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([1, 5, 6], result)

  def testInvalidDataTypeInMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 2
      value_index = 1
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertTrue("three_columns.txt_2_1" in init._shared_name)
      with self.assertRaisesOpError("is not a valid"):
        table = lookup_ops.HashTable(init, default_value)
        self.evaluate(table.initializer)

  def testInvalidDataType(self):
    vocabulary_file = self._createVocabFile("one_column_3.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup_ops.TextFileIndex.WHOLE_LINE
      value_index = lookup_ops.TextFileIndex.LINE_NUMBER

      with self.assertRaises(ValueError):
        init = lookup_ops.TextFileInitializer(vocabulary_file, dtypes.int64,
                                              key_index, dtypes.string,
                                              value_index)
        self.assertTrue("one_column_3.txt_-2_-1" in init._shared_name)
        lookup_ops.HashTable(init, default_value)

  def testInvalidIndex(self):
    vocabulary_file = self._createVocabFile("one_column_4.txt")
    with self.cached_session():
      default_value = -1
      key_index = 1  # second column of the line
      value_index = lookup_ops.TextFileIndex.LINE_NUMBER
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertTrue("one_column_4.txt_1_-1" in init._shared_name)

      with self.assertRaisesOpError("Invalid number of columns"):
        table = lookup_ops.HashTable(init, default_value)
        self.evaluate(table.initializer)

  def testInitializeSameTableWithMultipleNodes(self):
    vocabulary_file = self._createVocabFile("one_column_5.txt")

    with self.cached_session():
      default_value = -1
      init1 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertTrue("one_column_5.txt_-2_-1" in init1._shared_name)
      table1 = lookup_ops.HashTable(init1, default_value)
      init2 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertTrue("one_column_5.txt_-2_-1" in init2._shared_name)
      table2 = lookup_ops.HashTable(init2, default_value)
      init3 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertTrue("one_column_5.txt_-2_-1" in init3._shared_name)
      table3 = lookup_ops.HashTable(init3, default_value)

      self.evaluate(lookup_ops.tables_initializer())

      input_string = constant_op.constant(["brain", "salad", "tank"])

      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testInitializeTableWithNoFilename(self):
    with self.cached_session():
      default_value = -1
      with self.assertRaises(ValueError):
        lookup_ops.HashTable(
            lookup_ops.TextFileInitializer(
                "", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
            default_value)

  def testInitializeWithVocabSize(self):
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      vocabulary_file1 = self._createVocabFile("one_column6.txt")
      init1 = lookup_ops.TextFileInitializer(
          vocabulary_file1,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertTrue("one_column6.txt_3_-2_-1" in init1._shared_name)
      table1 = lookup_ops.HashTable(init1, default_value)

      # Initialize from file.
      self.evaluate(table1.initializer)
      self.assertEquals(vocab_size, self.evaluate(table1.size()))

      vocabulary_file2 = self._createVocabFile("one_column7.txt")
      vocab_size = 5
      init2 = lookup_ops.TextFileInitializer(
          vocabulary_file2,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertTrue("one_column7.txt_5_-2_-1" in init2._shared_name)
      with self.assertRaisesOpError("Invalid vocab_size"):
        table2 = lookup_ops.HashTable(init2, default_value)
        self.evaluate(table2.initializer)

      vocab_size = 1
      vocabulary_file3 = self._createVocabFile("one_column3.txt")
      init3 = lookup_ops.TextFileInitializer(
          vocabulary_file3,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertTrue("one_column3.txt_1_-2_-1" in init3._shared_name)
      table3 = lookup_ops.HashTable(init3, default_value)

      # Smaller vocab size reads only vocab_size records.
      self.evaluate(table3.initializer)
      self.assertEquals(vocab_size, self.evaluate(table3.size()))

  @test_util.run_v1_only("placeholder usage")
  def testFeedVocabularyName(self):
    vocabulary_file = self._createVocabFile("feed_vocabulary.txt")

    with self.cached_session():
      default_value = -1
      init = lookup_ops.TextFileInitializer(
          "old_file.txt", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertTrue("old_file.txt_-2_-1" in init._shared_name)
      table = lookup_ops.HashTable(init, default_value)

      # Initialize with non existing file (old_file.txt) should fail.
      # TODO(yleon): Update message, which might change per FileSystem.
      with self.assertRaisesOpError("old_file.txt"):
        table.initializer.run()

      # Initialize the model feeding the vocabulary file.
      filenames = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
      table.initializer.run(feed_dict={filenames[0]: vocabulary_file})

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testInvalidFilenames(self):
    vocabulary_file = self._createVocabFile("filename_shape.txt")

    with self.cached_session():
      default_value = -1

      # Invalid data type
      other_type = constant_op.constant(1)
      with self.assertRaises(ValueError):
        lookup_ops.HashTable(
            lookup_ops.TextFileInitializer(
                other_type, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
            default_value)

      # Non-scalar filename
      filenames = constant_op.constant([vocabulary_file, vocabulary_file])
      if not context.executing_eagerly():
        with self.assertRaises(ValueError):
          lookup_ops.HashTable(
              lookup_ops.TextFileInitializer(
                  filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                  dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
              default_value)
      else:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          lookup_ops.HashTable(
              lookup_ops.TextFileInitializer(
                  filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                  dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
              default_value)

  def testIdToStringTable(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = "UNK"
      vocab_size = 3
      init = lookup_ops.TextFileStringTableInitializer(
          vocab_file, vocab_size=vocab_size)
      self.assertTrue("feat_to_id_1.txt_3_-1_-2", init._shared_name)
      table = lookup_ops.HashTable(init, default_value)

      self.evaluate(table.initializer)

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)

      out = table.lookup(input_values)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"],
                          self.evaluate(out))
      self.assertEquals(vocab_size, self.evaluate(table.size()))

  def testStringToIdTable(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      init = lookup_ops.TextFileIdTableInitializer(
          vocab_file, vocab_size=vocab_size)
      self.assertTrue("feat_to_id_2.txt_3_-1_-2", init._shared_name)
      table = lookup_ops.HashTable(init, default_value)
      self.evaluate(table.initializer)

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, -1], self.evaluate(out))
      self.assertEquals(vocab_size, self.evaluate(table.size()))

  def testInt64ToIdTable(self):
    vocab_file = self._createVocabFile(
        "feat_to_id_3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      init = lookup_ops.TextFileIdTableInitializer(
          vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64)
      self.assertTrue("feat_to_id_3.txt_3_-1_-2", init._shared_name)
      table = lookup_ops.HashTable(init, default_value)
      self.evaluate(table.initializer)

      out = table.lookup(
          constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64))
      self.assertAllEqual((0, 1, 2, -1), self.evaluate(out))
      self.assertEquals(vocab_size, self.evaluate(table.size()))


class IdTableWithHashBucketsTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  @test_util.run_deprecated_v1
  def testStringIdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size), default_value),
          oov_buckets)

      table.initializer.run()

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  @test_util.run_deprecated_v1
  def testInt32IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt", ("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
              default_value),
          oov_buckets,
          key_dtype=dtypes.int32)

      table.initializer.run()

      values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int32)

      out = table.lookup(values)
      self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  @test_util.run_deprecated_v1
  def testInt64IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
              default_value), oov_buckets)

      table.initializer.run()

      values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64)

      out = table.lookup(values)
      self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  @test_util.run_deprecated_v1
  def testStringIdTableWithOnlyHashBucket(self):
    with self.cached_session():
      oov_buckets = 5

      # Set a table that only uses hash buckets, for each input value returns
      # an id calculated by fingerprint("input") mod oov_buckets.
      table = lookup_ops.IdTableWithHashBuckets(None, oov_buckets)
      table.initializer.run()

      values = constant_op.constant(("brain", "salad", "surgery"))

      out = table.lookup(values)
      self.assertAllEqual(
          [
              3,  # fingerprint("brain") mod 5.
              1,  # fingerprint("salad") mod 5.
              4  # fingerprint("surgery") mod 5
          ],
          self.evaluate(out))
      self.assertEquals(oov_buckets, table.size().eval())

  @test_util.run_deprecated_v1
  def testInt32IdTableWithOnlyHashBucket(self):
    with self.cached_session():
      oov_buckets = 5

      # Set a table that only uses hash buckets, for each input value returns
      # an id calculated by fingerprint("input") mod oov_buckets.
      table = lookup_ops.IdTableWithHashBuckets(
          None, oov_buckets, key_dtype=dtypes.int32)
      table.initializer.run()

      input_string = constant_op.constant([42, 1, -1000], dtype=dtypes.int32)

      out = table.lookup(input_string)
      self.assertAllEqual(
          [
              1,  # fingerprint("42") mod 5.
              4,  # fingerprint("1") mod 5.
              2  # fingerprint("-1000") mod 5
          ],
          self.evaluate(out))
      self.assertEquals(oov_buckets, table.size().eval())

  def testFloat64IdTableWithOnlyHashBucket(self):
    with self.cached_session():
      with self.assertRaisesRegexp(TypeError, "Invalid key_dtype"):
        lookup_ops.IdTableWithHashBuckets(
            None, num_oov_buckets=5, key_dtype=dtypes.float64)

  def testBoolIdTableWithOnlyHashBucket(self):
    with self.cached_session():
      with self.assertRaisesRegexp(TypeError, "Invalid key_dtype"):
        lookup_ops.IdTableWithHashBuckets(
            None, num_oov_buckets=5, key_dtype=dtypes.bool)

  @test_util.run_deprecated_v1
  def testIdTableWithHashBucketsWithMultipleInitializers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    with self.cached_session() as sess:
      default_value = -1
      vocab_size = 3
      oov_buckets = 3

      vocab_table = lookup_ops.HashTable(
          lookup_ops.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size), default_value)
      table1 = lookup_ops.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=lookup_ops.FastHashSpec,
          name="table1")

      table2 = lookup_ops.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=lookup_ops.StrongHashSpec((1, 2)),
          name="table2")

      lookup_ops.tables_initializer().run()

      input_string = constant_op.constant(
          ["fruit", "brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string)
      out2 = table2.lookup(input_string)

      out1, out2 = self.evaluate([out1, out2])
      self.assertAllEqual([5, 0, 1, 2, 5], out1)
      self.assertAllEqual([5, 0, 1, 2, 3], out2)
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())
      test_util.assert_ops_in_graph({
          "table1_Lookup/hash_bucket": "StringToHashBucketFast",
          "table2_Lookup/hash_bucket": "StringToHashBucketStrong",
      }, sess.graph)

  @test_util.run_deprecated_v1
  def testIdTableWithHashBucketsInitializationAcrossSessions(self):
    vocab_file = self._createVocabFile("feat_to_id_5.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table1 = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size), default_value),
          oov_buckets)

      table1.initializer.run()

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string_1)

      self.assertAllEqual([0, 1, 2, 3], self.evaluate(out1))
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())

    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1

      # Underlying lookup table already initialized in previous session.
      # No need to call table2.initializer.run()
      table2 = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size), default_value),
          oov_buckets)

      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out2 = table2.lookup(input_string_2)

      self.assertAllEqual([3, 1, 3], self.evaluate(out2))
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())

  @test_util.run_deprecated_v1
  def testIdTableWithHashBucketsWithMultipleInitializersDifferentDefault(self):
    vocab_file = self._createVocabFile("feat_to_id_6.txt")
    with self.cached_session() as sess:
      default_value1 = -1
      vocab_size = 3
      oov_buckets = 0
      table1 = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size), default_value1),
          oov_buckets)

      default_value2 = -2
      table2 = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size), default_value2),
          oov_buckets)

      lookup_ops.tables_initializer().run()

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])
      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out1 = table1.lookup(input_string_1)
      out2 = table2.lookup(input_string_2)

      out1, out2 = self.evaluate([out1, out2])
      self.assertAllEqual([0, 1, 2, -1], out1)
      self.assertAllEqual([-2, 1, -2], out2)
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())

  @test_util.run_deprecated_v1
  def testSparseTensor(self):
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                               dtypes.string),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
              -1), 1)
      table.initializer.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  @test_util.run_deprecated_v1
  def testInt32SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.KeyValueTensorInitializer(
                  (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
          1,
          key_dtype=dtypes.int32)
      table.initializer.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  @test_util.run_deprecated_v1
  def testInt64SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup_ops.IdTableWithHashBuckets(
          lookup_ops.HashTable(
              lookup_ops.KeyValueTensorInitializer(
                  (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
          1,
          key_dtype=dtypes.int64)
      table.initializer.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  def testIdTableWithHashBucketsWithInvalidHashers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      lookup_table = lookup_ops.HashTable(
          lookup_ops.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size), default_value)

      with self.assertRaises(TypeError):
        lookup_ops.IdTableWithHashBuckets(
            lookup_table, oov_buckets, hasher_spec=1)

      table = lookup_ops.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      with self.assertRaises(ValueError):
        table.lookup(input_string)

      with self.assertRaises(ValueError):
        table = lookup_ops.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup_ops.StrongHashSpec([]))

      with self.assertRaises(ValueError):
        table = lookup_ops.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup_ops.StrongHashSpec([1, 2, 3]))

      with self.assertRaises(TypeError):
        table = lookup_ops.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup_ops.StrongHashSpec([None, 2]))

  def testIdTableWithHashBucketsNoInnerTable(self):
    with self.cached_session():
      table = lookup_ops.IdTableWithHashBuckets(None, num_oov_buckets=1)
      self.assertIsNone(table.resource_handle)


class MutableHashTableOpTest(test.TestCase):

  def testMutableHashTable(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant(["tarkus", "tank"])
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

      exported_keys, exported_values = table.export()

      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(self.evaluate(exported_keys))
      sorted_values = np.sort(self.evaluate(exported_values))
      self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
      self.assertAllEqual([0, 1, 2], sorted_values)

  @test_util.run_v1_only("SaverV1")
  def testSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant(["b", "c", "d"], dtypes.string)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(
          dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)

      save = saver.Saver()
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = -1
      table = lookup_ops.MutableHashTable(
          dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)
      self.evaluate(
          table.insert(
              constant_op.constant(["a", "c"], dtypes.string),
              constant_op.constant([12, 24], dtypes.int64)))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                          dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(20.0, name="v1")

    default_val = -1
    keys = constant_op.constant(["b", "c", "d"], dtypes.string)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)

    checkpoint = checkpointable.Checkpoint(table=table, v0=v0, v1=v1)
    self.evaluate([v0.initializer, v1.initializer])

    # Check that the parameter nodes have been initialized.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(0, self.evaluate(table.size()))
    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    save_path = checkpoint.save(save_prefix)
    del table, checkpoint, v0, v1

    v0 = variables.Variable(-1.0, name="v0")
    v1 = variables.Variable(-1.0, name="v1")
    default_val = -1
    table = lookup_ops.MutableHashTable(
        dtypes.string, dtypes.int64, default_val, name="t1", checkpoint=True)
    self.evaluate(
        table.insert(
            constant_op.constant(["a", "c"], dtypes.string),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(table.size()))

    checkpoint = checkpointable.Checkpoint(table=table, v0=v0, v1=v1)

    # Restore the saved values in the parameter nodes.
    checkpoint.restore(save_path).run_restore_ops()
    # Check that the parameter nodes have been restored.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                        dtypes.string)
    output = table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v1_only("Multiple sessions")
  def testSharing(self):
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    table = lookup_ops.MutableHashTable(
        dtypes.int64, dtypes.string, "-", name="t1")

    # Populate the table in the first session
    with session1:
      self.assertAllEqual(0, table.size().eval())

      keys = constant_op.constant([11, 12], dtypes.int64)
      values = constant_op.constant(["a", "b"])
      table.insert(keys, values).run()
      self.assertAllEqual(2, table.size().eval())

      output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
      self.assertAllEqual([b"a", b"b", b"-"], output.eval())

    # Verify that we can access the shared data from the second session
    with session2:
      self.assertAllEqual(2, table.size().eval())

      output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
      self.assertAllEqual([b"-", b"a", b"b"], output.eval())

  def testMutableHashTableOfTensors(self):
    with self.cached_session():
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
      values = constant_op.constant([[0, 1], [2, 3], [4, 5], [6, 7]],
                                    dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant(["tarkus", "tank"])
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3, 2], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -1]], result)

      exported_keys, exported_values = table.export()
      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(self.evaluate(exported_keys))
      sorted_values = np.sort(self.evaluate(exported_values))
      self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
      sorted_expected_values = np.sort([[4, 5], [2, 3], [0, 1]])
      self.assertAllEqual(sorted_expected_values, sorted_values)

  def testMutableHashTableExportInsert(self):
    with self.cached_session():
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
      table1 = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                           default_val)
      self.assertAllEqual(0, self.evaluate(table1.size()))
      self.evaluate(table1.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table1.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      expected_output = [[0, 1], [2, 3], [-1, -1]]
      output1 = table1.lookup(input_string)
      self.assertAllEqual(expected_output, self.evaluate(output1))

      exported_keys, exported_values = table1.export()
      self.assertAllEqual(3, self.evaluate(exported_keys).size)
      self.assertAllEqual(6, self.evaluate(exported_values).size)

      # Populate a second table from the exported data
      table2 = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                           default_val)
      self.assertAllEqual(0, self.evaluate(table2.size()))
      self.evaluate(table2.insert(exported_keys, exported_values))
      self.assertAllEqual(3, self.evaluate(table2.size()))

      # Verify lookup result is still the same
      output2 = table2.lookup(input_string)
      self.assertAllEqual(expected_output, self.evaluate(output2))

  def testMutableHashTableOfTensorsInvalidShape(self):
    with self.cached_session():
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      # Shape [6] instead of [3, 2]
      values = constant_op.constant([0, 1, 2, 3, 4, 5], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.insert(keys, values))

      # Shape [2,3] instead of [3, 2]
      values = constant_op.constant([[0, 1, 2], [3, 4, 5]], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.insert(keys, values))

      # Shape [2, 2] instead of [3, 2]
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.insert(keys, values))

      # Shape [3, 1] instead of [3, 2]
      values = constant_op.constant([[0], [2], [4]], dtypes.int64)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.insert(keys, values))

      # Valid Insert
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

  def testMutableHashTableInvalidDefaultValue(self):
    with self.cached_session():
      default_val = constant_op.constant([[-1, -1]], dtypes.int64)
      with self.assertRaisesOpError("Default value must be a vector"):
        table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                            default_val)
        self.assertAllEqual(0, self.evaluate(table.size()))

  def testMutableHashTableDuplicateInsert(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery", "brain"])
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([3, 1, -1], result)

  def testMutableHashTableFindHighRank(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([["brain", "salad"],
                                           ["tank", "tarkus"]])
      output = table.lookup(input_string)
      self.assertAllEqual([2, 2], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testMutableHashTableInsertHighRank(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant([["brain", "salad"], ["surgery", "tank"]])
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank", "tarkus"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, 3, -1], result)

  def testMutableHashTableRemoveHighRank(self):
    with self.test_session():
      default_val = -1
      keys = constant_op.constant([["brain", "salad"], ["surgery", "tank"]])
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant(["salad", "tarkus"])
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank", "tarkus"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, -1, 3, -1], result)

  def testMutableHashTableOfTensorsFindHighRank(self):
    with self.cached_session():
      default_val = constant_op.constant([-1, -1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([["brain", "salad"],
                                           ["tank", "tarkus"]])
      output = table.lookup(input_string)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
          [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def testMutableHashTableOfTensorsRemoveHighRank(self):
    with self.test_session():
      default_val = constant_op.constant([-1, -1, -1], dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_string = constant_op.constant([["brain", "tank"]])
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(2, self.evaluate(table.size()))

      input_string = constant_op.constant([["brain", "salad"],
                                           ["surgery", "tank"]])
      output = table.lookup(input_string)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
          [[[-1, -1, -1], [2, 3, 4]], [[4, 5, 6], [-1, -1, -1]]], result)

  def testMultipleMutableHashTables(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)

      table1 = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                           default_val)
      table2 = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                           default_val)
      table3 = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                           default_val)
      self.evaluate(table1.insert(keys, values))
      self.evaluate(table2.insert(keys, values))
      self.evaluate(table3.insert(keys, values))

      self.assertAllEqual(3, self.evaluate(table1.size()))
      self.assertAllEqual(3, self.evaluate(table2.size()))
      self.assertAllEqual(3, self.evaluate(table3.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testMutableHashTableWithTensorDefault(self):
    with self.cached_session():
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testSignatureMismatch(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.int64,
                                          default_val)

      # insert with keys of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(table.insert(constant_op.constant([4, 5, 6]), values))

      # insert with values of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(table.insert(keys, constant_op.constant(["a", "b", "c"])))

      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string_ref = variables.Variable("brain")
      input_int64_ref = variables.Variable(-1, dtype=dtypes.int64)
      self.evaluate(variables.global_variables_initializer())

      # Ref types do not produce an insert signature mismatch.
      self.evaluate(table.insert(input_string_ref, input_int64_ref))
      self.assertAllEqual(3, self.evaluate(table.size()))

      # Ref types do not produce a lookup signature mismatch.
      self.assertEqual(-1, self.evaluate(table.lookup(input_string_ref)))

      # lookup with keys of the wrong type
      input_string = constant_op.constant([1, 2, 3], dtypes.int64)
      with self.assertRaises(ValueError):
        self.evaluate(table.lookup(input_string))

      # default value of the wrong type
      with self.assertRaises(TypeError):
        lookup_ops.MutableHashTable(dtypes.string, dtypes.int64, "UNK")

  def testMutableHashTableStringFloat(self):
    with self.cached_session():
      default_val = -1.5
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1.1, 2.2], dtypes.float32)
      table = lookup_ops.MutableHashTable(dtypes.string, dtypes.float32,
                                          default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllClose([0, 1.1, default_val], result)

  def testMutableHashTableIntFloat(self):
    with self.cached_session():
      default_val = -1.0
      keys = constant_op.constant([3, 7, 0], dtypes.int64)
      values = constant_op.constant([7.5, -1.2, 9.9], dtypes.float32)
      table = lookup_ops.MutableHashTable(dtypes.int64, dtypes.float32,
                                          default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([7, 0, 11], dtypes.int64)
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllClose([-1.2, 9.9, default_val], result)

  def testMutableHashTableInt64String(self):
    with self.cached_session():
      default_val = "n/a"
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup_ops.MutableHashTable(dtypes.int64, dtypes.string,
                                          default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual((b"brain", b"salad", b"n/a"), result)


class MutableDenseHashTableOpTest(test.TestCase):

  def testBasic(self):
    with self.cached_session():

      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          deleted_key=-1)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant([12, 15], dtypes.int64)
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([11, 12, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([0, -1, -1], result)

  def testBasicBool(self):
    with self.cached_session():

      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([True, True, True, True], dtypes.bool)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.bool,
          default_value=False,
          empty_key=0,
          deleted_key=-1)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant([11, 15], dtypes.int64)
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([11, 12, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([False, True, False], result)

  def testSameEmptyAndDeletedKey(self):
    with self.cached_session():
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Empty and deleted keys"):
        table = lookup_ops.MutableDenseHashTable(
            dtypes.int64,
            dtypes.int64,
            default_value=-1,
            empty_key=42,
            deleted_key=42)
        self.assertAllEqual(0, self.evaluate(table.size()))

  @test_util.run_v1_only("uses placeholders")
  def testLookupUnknownShape(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          deleted_key=-1)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      placeholder_keys = array_ops.placeholder(dtypes.int64)
      output = table.lookup(placeholder_keys)
      self.assertAllEqual(None, output.get_shape())
      result = output.eval({placeholder_keys: [11, 12, 15]})
      self.assertAllEqual([0, 1, -1], result)

  def testMapStringToFloat(self):
    with self.cached_session():

      keys = constant_op.constant(["a", "b", "c", "d"], dtypes.string)
      values = constant_op.constant([0.0, 1.1, 2.2, 3.3], dtypes.float32)
      default_value = constant_op.constant(-1.5, dtypes.float32)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.string,
          dtypes.float32,
          default_value=default_value,
          empty_key="",
          deleted_key="$")
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant(["b", "e"])
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["a", "b", "d", "e"], dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([4], output.get_shape())

      result = self.evaluate(output)
      self.assertAllClose([0, -1.5, 3.3, -1.5], result)

  def testMapInt64ToFloat(self):
    for float_dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():

        keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
        values = constant_op.constant([0.0, 1.1, 2.2, 3.3], float_dtype)
        default_value = constant_op.constant(-1.5, float_dtype)
        table = lookup_ops.MutableDenseHashTable(
            dtypes.int64,
            float_dtype,
            default_value=default_value,
            empty_key=0,
            deleted_key=-1)
        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.insert(keys, values))
        self.assertAllEqual(4, self.evaluate(table.size()))

        remove_string = constant_op.constant([12, 15], dtypes.int64)
        self.evaluate(table.remove(remove_string))
        self.assertAllEqual(3, self.evaluate(table.size()))

        input_string = constant_op.constant([11, 12, 14, 15], dtypes.int64)
        output = table.lookup(input_string)
        self.assertAllEqual([4], output.get_shape())

        result = self.evaluate(output)
        self.assertAllClose([0, -1.5, 3.3, -1.5], result)

  def testVectorValues(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]],
                                    dtypes.int64)
      default_value = constant_op.constant([-1, -2, -3, -4], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=0,
          deleted_key=-1,
          initial_num_buckets=4)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))
      self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

      self.evaluate(
          table.insert(
              constant_op.constant([14], dtypes.int64),
              constant_op.constant([[2, 3, 4, 5]], dtypes.int64)))
      self.assertAllEqual(4, self.evaluate(table.size()))
      self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

      remove_string = constant_op.constant([12, 16], dtypes.int64)
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))
      self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

      input_string = constant_op.constant([11, 12, 14, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([4, 4],
                          output.shape,
                          msg="Saw shape: %s" % output.shape)

      result = self.evaluate(output)
      self.assertAllEqual(
          [[0, 1, 2, 3], [-1, -2, -3, -4], [2, 3, 4, 5], [-1, -2, -3, -4]],
          result)

  def testVectorKeys(self):
    with self.cached_session():
      keys = constant_op.constant([[0, 1], [1, 2], [1, 3]], dtypes.int64)
      values = constant_op.constant([10, 11, 12], dtypes.int64)
      empty_key = constant_op.constant([0, 3], dtypes.int64)
      deleted_key = constant_op.constant([-1, -1], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          initial_num_buckets=8)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      self.evaluate(
          table.insert(
              constant_op.constant([[0, 0]], dtypes.int64),
              constant_op.constant([13], dtypes.int64)))
      self.assertAllEqual(4, self.evaluate(table.size()))
      self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

      remove_string = constant_op.constant([[1, 2], [7, 8]], dtypes.int64)
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))
      self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

      input_string = constant_op.constant([[0, 1], [1, 2], [1, 3], [0, 2]],
                                          dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([4], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([10, -1, 12, -1], result)

  def testResize(self):
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          deleted_key=-1,
          initial_num_buckets=4)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))
      self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

      keys2 = constant_op.constant([12, 99], dtypes.int64)
      self.evaluate(table.remove(keys2))
      self.assertAllEqual(2, self.evaluate(table.size()))
      self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

      keys3 = constant_op.constant([13, 14, 15, 16, 17], dtypes.int64)
      values3 = constant_op.constant([3, 4, 5, 6, 7], dtypes.int64)

      self.evaluate(table.insert(keys3, values3))
      self.assertAllEqual(6, self.evaluate(table.size()))
      self.assertAllEqual(16, len(self.evaluate(table.export()[0])))

      keys4 = constant_op.constant([10, 11, 12, 13, 14, 15, 16, 17, 18],
                                   dtypes.int64)
      output = table.lookup(keys4)
      self.assertAllEqual([-1, 0, -1, 3, 4, 5, 6, 7, -1], self.evaluate(output))

  def testExport(self):
    with self.cached_session():

      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([1, 2, 3, 4], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=100,
          deleted_key=200,
          initial_num_buckets=8)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      keys2 = constant_op.constant([12, 15], dtypes.int64)
      self.evaluate(table.remove(keys2))
      self.assertAllEqual(3, self.evaluate(table.size()))

      exported_keys, exported_values = table.export()

      np_keys = self.evaluate(exported_keys)
      np_values = self.evaluate(exported_values)

      self.assertAllEqual(8, len(np_keys))
      self.assertAllEqual(8, len(np_values))

      # pair up keys and values, drop extra added dimension
      pairs = np.dstack((np_keys.flatten(), np_values.flatten()))[0]
      # sort by key
      pairs = pairs[pairs[:, 0].argsort()]
      self.assertAllEqual([[11, 1], [13, 3], [14, 4], [100, 0], [100, 0],
                           [100, 0], [100, 0], [200, 2]], pairs)

  @test_util.run_v1_only("Saver V1 only")
  def testSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      default_value = -1
      empty_key = 0
      deleted_key = -1
      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32)

      save = saver.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([12, 15], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          constant_op.constant([11, 14], dtypes.int64),
          constant_op.constant([12, 24], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, -1, 2, 3], output.eval())

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    default_value = -1
    empty_key = 0
    deleted_key = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    save_table = lookup_ops.MutableDenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=32)

    save_checkpoint = checkpointable.Checkpoint(table=save_table)

    self.assertAllEqual(0, self.evaluate(save_table.size()))
    self.evaluate(save_table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(save_table.size()))
    self.assertAllEqual(32, len(self.evaluate(save_table.export()[0])))

    save_path = save_checkpoint.save(save_prefix)
    del save_table, save_checkpoint

    load_table = lookup_ops.MutableDenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=64)
    self.evaluate(
        load_table.insert(
            constant_op.constant([11, 14], dtypes.int64),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(load_table.size()))
    self.assertAllEqual(64, len(self.evaluate(load_table.export()[0])))

    restore_checkpoint = checkpointable.Checkpoint(table=load_table)

    # Restore the saved values in the parameter nodes.
    restore_checkpoint.restore(save_path).run_restore_ops()

    self.assertAllEqual(3, self.evaluate(load_table.size()))
    self.assertAllEqual(32, len(self.evaluate(load_table.export()[0])))

    input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
    output = load_table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v1_only("Saver V1 only")
  def testVectorSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "vector_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-2, -3], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [12, 13], [13, 14]],
                                  dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [2, 4], [4, 5]],
                                    dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32)

      save = saver.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([[12, 13], [16, 17]], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-2, -3], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([[21, 22], [23, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -2], [4, 5], [-1, -2]],
                          output.eval())

  @test_util.run_v1_only("Saver V1 only")
  def testVectorScalarSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "vector_scalar_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-1, -1], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [12, 13], [13, 14]],
                                  dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=32)

      save = saver.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([[12, 13], [15, 16]], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-1, -1], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([3, 4], dtypes.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([0, 1, -1, 3, -1], output.eval())

  def testReprobe(self):
    with self.cached_session():
      # Insert 6 keys into a table with 8 buckets.
      # The values are chosen to make sure collisions occur when using GCC STL
      keys = constant_op.constant([11, 12, 13, 19, 20, 21], dtypes.int64)
      values = constant_op.constant([51, 52, 53, 54, 55, 56], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          deleted_key=-1,
          initial_num_buckets=8)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(6, self.evaluate(table.size()))

      input_string = constant_op.constant([10, 11, 12, 13, 14, 19, 20, 21, 22],
                                          dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([9], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([-1, 51, 52, 53, -1, 54, 55, 56, -1], result)

  def testCustomEmptyKey(self):
    with self.cached_session():
      keys = constant_op.constant([11, 0, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=12,
          deleted_key=-1)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([11, 0, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testErrors(self):
    with self.cached_session():
      table = lookup_ops.MutableDenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          deleted_key=-1)

      # Inserting the empty key returns an error
      keys1 = constant_op.constant([11, 0], dtypes.int64)
      values1 = constant_op.constant([0, 1], dtypes.int64)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "empty_key"):
        self.evaluate(table.insert(keys1, values1))

      # Looking up the empty key returns an error
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "empty_key"):
        self.evaluate(table.lookup(keys1))

      # Inserting the deleted key returns an error
      keys2 = constant_op.constant([11, -1], dtypes.int64)
      values2 = constant_op.constant([0, 1], dtypes.int64)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "deleted_key"):
        self.evaluate(table.insert(keys2, values2))

      # Looking up the empty key returns an error
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "deleted_key"):
        self.evaluate(table.lookup(keys2))

      # Arbitrary tensors of keys are not supported
      keys = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
      values = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Expected key shape"):
        self.evaluate(table.lookup(keys))
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Expected key shape"):
        self.evaluate(table.insert(keys, values))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Number of buckets must be"):
        table2 = lookup_ops.MutableDenseHashTable(
            dtypes.int64,
            dtypes.int64,
            default_value=-1,
            empty_key=17,
            deleted_key=-1,
            initial_num_buckets=12)
        self.assertAllEqual(0, self.evaluate(table2.size()))

      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          "Empty and deleted keys must have same shape"):
        table3 = lookup_ops.MutableDenseHashTable(
            dtypes.int64,
            dtypes.int64,
            default_value=-1,
            empty_key=42,
            deleted_key=[1, 2])
        self.assertAllEqual(0, self.evaluate(table3.size()))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Empty and deleted keys cannot be equal"):
        table4 = lookup_ops.MutableDenseHashTable(
            dtypes.int64,
            dtypes.int64,
            default_value=-1,
            empty_key=42,
            deleted_key=42)
        self.assertAllEqual(0, self.evaluate(table4.size()))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Empty and deleted keys cannot be equal"):
        table5 = lookup_ops.MutableDenseHashTable(
            dtypes.int64,
            dtypes.int64,
            default_value=-1,
            empty_key=[1, 2, 3],
            deleted_key=[1, 2, 3])
        self.assertAllEqual(0, self.evaluate(table5.size()))


class MutableHashTableBenchmark(test.Benchmark):

  def _create_table(self):
    return lookup_ops.MutableHashTable(dtypes.int64, dtypes.float32, 0.0)

  def benchmark_single_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable(1.0)
    insert = table.insert(0, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) == 1

  def benchmark_many_repeated_scalar_insert_scalar(self):
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable(1.0)
    insert = table.insert(c, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) >= 10000

  def benchmark_single_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) == 32

  def benchmark_many_repeated_batch_32_insert_scalar(self):
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(32 * c + list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) >= 1000 * 32


class MutableDenseHashTableBenchmark(MutableHashTableBenchmark):

  def _create_table(self):
    return lookup_ops.MutableDenseHashTable(
        dtypes.int64,
        dtypes.float32,
        default_value=0.0,
        empty_key=-1,
        deleted_key=-2)


if __name__ == "__main__":
  test.main()
