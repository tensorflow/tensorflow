# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.lookup.lookup."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.contrib import lookup
from tensorflow.python.client import session
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
from tensorflow.python.training import server_lib


class HashTableOpTest(test.TestCase):

  def testHashTable(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

      exported_keys_tensor, exported_values_tensor = table.export()

      self.assertItemsEqual([b"brain", b"salad", b"surgery"],
                            exported_keys_tensor.eval())
      self.assertItemsEqual([0, 1, 2], exported_values_tensor.eval())

  def testHashTableFindHighRank(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(
          [["brain", "salad"], ["tank", "tarkus"]])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testHashTableInitWithPythonArrays(self):
    with self.cached_session():
      default_val = -1
      keys = ["brain", "salad", "surgery"]
      values = [0, 1, 2]
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(
              keys, values, value_dtype=dtypes.int64),
          default_val)
      table.initializer.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableInitWithNumPyArrays(self):
    with self.cached_session():
      default_val = -1
      keys = np.array(["brain", "salad", "surgery"], dtype=np.str)
      values = np.array([0, 1, 2], dtype=np.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testMultipleHashTables(self):
    with self.cached_session() as sess:
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)

      table1 = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table2 = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table3 = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual(3, table1.size().eval())
      self.assertAllEqual(3, table2.size().eval())
      self.assertAllEqual(3, table3.size().eval())

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testHashTableWithTensorDefault(self):
    with self.cached_session():
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableWithSparseTensorInput(self):
    with self.cached_session() as sess:
      default_val = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      sp_indices = [[0, 0], [0, 1], [1, 0]]
      sp_shape = [2, 2]
      input_tensor = sparse_tensor.SparseTensor(
          constant_op.constant(sp_indices, dtypes.int64),
          constant_op.constant(["brain", "salad", "tank"]),
          constant_op.constant(sp_shape, dtypes.int64))
      output = table.lookup(input_tensor)

      out_indices, out_values, out_shape = sess.run(output)

      self.assertAllEqual([0, 1, -1], out_values)
      self.assertAllEqual(sp_indices, out_indices)
      self.assertAllEqual(sp_shape, out_shape)

  def testSignatureMismatch(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      # Ref types do not produce a lookup signature mismatch.
      input_string_ref = variables.Variable("brain")
      variables.global_variables_initializer().run()
      self.assertEqual(0, table.lookup(input_string_ref).eval())

      input_string = constant_op.constant([1, 2, 3], dtypes.int64)
      with self.assertRaises(TypeError):
        table.lookup(input_string)

      with self.assertRaises(TypeError):
        lookup.HashTable(
            lookup.KeyValueTensorInitializer(keys, values), "UNK")

  def testDTypes(self):
    with self.cached_session():
      default_val = -1
      with self.assertRaises(TypeError):
        lookup.HashTable(
            lookup.KeyValueTensorInitializer(["a"], [1], [dtypes.string],
                                             dtypes.int64), default_val)

  def testNotInitialized(self):
    with self.cached_session():
      default_val = -1
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(
              ["a"], [1], value_dtype=dtypes.int64),
          default_val)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      with self.assertRaisesOpError("Table not initialized"):
        output.eval()

  def testInitializeTwice(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      with self.assertRaisesOpError("Table already initialized"):
        table.initializer.run()

  def testInitializationWithInvalidDimensions(self):
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)

      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.KeyValueTensorInitializer(keys, values), default_val)

  def testMultipleSessions(self):
    # Start a server
    server = server_lib.Server(
        {
            "local0": ["localhost:0"]
        }, protocol="grpc", start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup.HashTable(
        lookup.KeyValueTensorInitializer(keys, values),
        default_val,
        name="t1")

    # Init the table in the first session.
    with session1:
      table.initializer.run()
      self.assertAllEqual(3, table.size().eval())

    # Init the table in the second session and verify that we do not get a
    # "Table already initialized" error.
    with session2:
      table.initializer.run()
      self.assertAllEqual(3, table.size().eval())

  def testHashTableInt32String(self):
    with self.cached_session():
      default_val = "n/a"
      keys = constant_op.constant([0, 1, 2], dtypes.int32)
      values = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.HashTable(
          lookup.KeyValueTensorInitializer(keys, values), default_val)
      table.initializer.run()

      input_tensor = constant_op.constant([0, 1, -1])
      output = table.lookup(input_tensor)

      result = output.eval()
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
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_string_index_table_from_file_tensor_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_file = constant_op.constant(vocabulary_file)
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())
      self.assertEqual(1,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  def test_string_index_table_from_file_placeholder_filename(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_placeholder = array_ops.placeholder(dtypes.string, [])
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_placeholder, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)

      feed_dict = {vocabulary_placeholder.name: vocabulary_file}
      lookup_ops.tables_initializer().run(feed_dict=feed_dict)
      self.assertAllEqual((1, 2, 3), ids.eval())
      self.assertEqual(0,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  def test_int32_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab2.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1,
          key_dtype=dtypes.int32)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_int64_index_table_from_file(self):
    vocabulary_file = self._createVocabFile(
        "f2i_vocab3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1,
          key_dtype=dtypes.int64)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_index_table_from_file_with_default_value(self):
    default_value = -42
    vocabulary_file = self._createVocabFile("f2i_vocab4.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, default_value), ids.eval())

  def test_index_table_from_file_with_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab5.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1000)
      ids = table.lookup(
          constant_op.constant(["salad", "surgery", "tarkus", "toccata"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual(
          (
              1,  # From vocabulary file.
              2,  # From vocabulary file.
              867,  # 3 + fingerprint("tarkus") mod 300.
              860),  # 3 + fingerprint("toccata") mod 300.
          ids.eval())

  def test_index_table_from_file_fails_with_empty_vocabulary_file_name(self):
    self.assertRaises(
        ValueError,
        lookup.index_table_from_file,
        vocabulary_file="")

  def test_index_table_from_file_fails_with_empty_vocabulary(self):
    self.assertRaises(
        ValueError,
        lookup.index_table_from_file,
        vocabulary_file=None)

  def test_index_table_from_file_with_vocab_size_too_small(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=2)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, -1, -1), ids.eval())
      self.assertEqual(2, table.size().eval())

  def test_index_table_from_file_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                              "Invalid vocab_size", table.initializer.run)

  def test_index_table_from_file_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab8.txt")

    self.assertRaises(
        ValueError,
        lookup.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

    with self.cached_session():
      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, -1), ids.eval())
      self.assertEqual(3, table.size().eval())

  def test_index_table_from_file_with_invalid_hashers(self):
    vocabulary_file = self._createVocabFile("invalid_hasher.txt")
    with self.cached_session():
      with self.assertRaises(TypeError):
        lookup.index_table_from_file(
            vocabulary_file=vocabulary_file,
            vocab_size=3,
            num_oov_buckets=1,
            hasher_spec=1)

      table = lookup.index_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=3,
          num_oov_buckets=1,
          hasher_spec=lookup.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        constant_op.constant(["salad", "surgery", "tarkus"]))


class KeyValueTensorInitializerTest(test.TestCase):

  def test_string(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table = lookup.HashTable(init, default_value=-1)
      table.initializer.run()

  def test_int64(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup.KeyValueTensorInitializer(
          (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64)
      table = lookup.HashTable(init, default_value=-1)
      table.initializer.run()

  def test_int32(self):
    with ops.Graph().as_default(), self.cached_session():
      init = lookup.KeyValueTensorInitializer(
          (42, 1, -1000), (0, 1, 2), dtypes.int32, dtypes.int64)
      table = lookup.HashTable(init, default_value=-1)
      with self.assertRaisesRegexp(
          errors_impl.OpError, "No OpKernel was registered"):
        table.initializer.run()


class IndexTableFromTensor(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_index_table_from_tensor_with_tensor_init(self):
    table = lookup.index_table_from_tensor(
        mapping=("brain", "salad", "surgery"), num_oov_buckets=1)

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(table.lookup(
            constant_op.constant(("salad", "surgery", "tarkus"))))
    else:
      # Reinitializing a table in eager should work.
      table = lookup.index_table_from_tensor(
          mapping=("brain", "salad", "surgery"), num_oov_buckets=1)
    self.evaluate(lookup_ops.tables_initializer())
    ids = table.lookup(constant_op.constant(("salad", "surgery", "tarkus")))
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int32_index_table_from_tensor_with_tensor_init(self):
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int32)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_int64_index_table_from_tensor_with_tensor_init(self):
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int64)
      ids = table.lookup(
          constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_index_table_from_tensor_with_default_value(self):
    default_value = -42
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=["brain", "salad", "surgery"], default_value=default_value)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(errors_impl.OpError, ids.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, default_value), ids.eval())

  def test_index_table_from_tensor_missing_mapping(self):
    with self.cached_session():
      with self.assertRaisesRegexp(ValueError, "mapping must be specified"):
        lookup.index_table_from_tensor(mapping=None, num_oov_buckets=1)

  def test_index_table_from_tensor_empty_mapping(self):
    with self.cached_session():
      table = lookup.index_table_from_tensor(
          mapping=np.array([], dtype=np.str_), num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "brain"]))
      self.assertRaises(errors_impl.OpError, ids.eval)
      with self.assertRaisesRegexp(
          errors_impl.OpError, "keys and values cannot be empty"):
        lookup_ops.tables_initializer().run()

  def test_index_table_from_tensor_with_invalid_hashers(self):
    with self.cached_session():
      with self.assertRaises(TypeError):
        lookup.index_table_from_tensor(
            mapping=["brain", "salad", "surgery"],
            num_oov_buckets=1,
            hasher_spec=1)

      table = lookup.index_table_from_tensor(
          mapping=["brain", "salad", "surgery"],
          num_oov_buckets=1,
          hasher_spec=lookup.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        constant_op.constant(["salad", "surgery", "tarkus"]))


class StringToIndexTest(test.TestCase):

  def test_string_to_index(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      feats = constant_op.constant(["salad", "surgery", "tarkus"])
      indices = lookup.string_to_index(feats, mapping=mapping_strings)

      self.assertRaises(errors_impl.OpError, indices.eval)
      lookup_ops.tables_initializer().run()

      self.assertAllEqual((1, 2, -1), indices.eval())

  def test_duplicate_entries(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["hello", "hello"])
      feats = constant_op.constant(["hello", "hola"])
      _ = lookup.string_to_index(feats, mapping=mapping_strings)

      self.assertRaises(errors_impl.OpError,
                        lookup_ops.tables_initializer().run)

  def test_string_to_index_with_default_value(self):
    default_value = -42
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      feats = constant_op.constant(["salad", "surgery", "tarkus"])
      indices = lookup.string_to_index(
          feats, mapping=mapping_strings, default_value=default_value)
      self.assertRaises(errors_impl.OpError, indices.eval)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual((1, 2, default_value), indices.eval())


class IndexToStringTableFromFileTest(test.TestCase):

  def _createVocabFile(self, basename):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["brain", "salad", "surgery"]) + "\n")
    return vocabulary_file

  def test_index_to_string_table(self):
    vocabulary_file = self._createVocabFile("i2f_vocab1.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file)
      features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          features.eval())

  def test_index_to_string_table_with_default_value(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          features.eval())

  def test_index_to_string_table_with_vocab_size_too_small(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=2,
          default_value=default_value)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", default_value, default_value),
                          features.eval())

  def test_index_to_string_table_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))

      self.assertRaises(errors_impl.OpError, features.eval)
      init = lookup_ops.tables_initializer()
      self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                              "Invalid vocab_size", init.run)

  def test_index_to_string_table_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.cached_session():
      table = lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))

      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", b"UNK"), features.eval())


class IndexToStringTableFromTensorTest(test.TestCase):

  def test_index_to_string_table_from_tensor(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings)

      indices = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      features = table.lookup(indices)
      self.assertRaises(errors_impl.OpError, features.eval)
      lookup_ops.tables_initializer().run()

      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          features.eval())

  def test_duplicate_entries(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["hello", "hello"])
      table = lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings)
      indices = constant_op.constant([0, 1, 4], dtypes.int64)
      features = table.lookup(indices)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"hello", b"hello", b"UNK"), features.eval())

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      table = lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings, default_value=default_value)
      indices = constant_op.constant([1, 2, 4], dtypes.int64)
      features = table.lookup(indices)
      self.assertRaises(errors_impl.OpError, features.eval)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          features.eval())


class IndexToStringTest(test.TestCase):

  def test_index_to_string(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      indices = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      feats = lookup.index_to_string(indices, mapping=mapping_strings)

      self.assertRaises(errors_impl.OpError, feats.eval)
      lookup_ops.tables_initializer().run()

      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          feats.eval())

  def test_duplicate_entries(self):
    with self.cached_session():
      mapping_strings = constant_op.constant(["hello", "hello"])
      indices = constant_op.constant([0, 1, 4], dtypes.int64)
      feats = lookup.index_to_string(indices, mapping=mapping_strings)
      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"hello", b"hello", b"UNK"), feats.eval())

      self.assertRaises(errors_impl.OpError,
                        lookup_ops.tables_initializer().run)

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    with self.cached_session():
      mapping_strings = constant_op.constant(["brain", "salad", "surgery"])
      indices = constant_op.constant([1, 2, 4], dtypes.int64)
      feats = lookup.index_to_string(
          indices, mapping=mapping_strings, default_value=default_value)
      self.assertRaises(errors_impl.OpError, feats.eval)

      lookup_ops.tables_initializer().run()
      self.assertAllEqual((b"salad", b"surgery", default_value), feats.eval())


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
    table = lookup.HashTable(
        lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                   lookup.TextFileIndex.WHOLE_LINE,
                                   dtypes.int64,
                                   lookup.TextFileIndex.LINE_NUMBER),
        default_value)
    self.evaluate(table.initializer)

    output = table.lookup(constant_op.constant(["brain", "salad", "tank"]))

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testInitializeInt64Table(self):
    vocabulary_file = self._createVocabFile(
        "one_column_int64.txt", values=("42", "1", "-1000"))

    with self.cached_session():
      default_value = -1
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.int64,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value)
      table.initializer.run()

      output = table.lookup(
          constant_op.constant((42, 1, 11), dtype=dtypes.int64))

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testInitializeIndexTable(self):
    vocabulary_file = self._createVocabFile("one_column_2.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup.TextFileIndex.LINE_NUMBER
      value_index = lookup.TextFileIndex.WHOLE_LINE
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.int64,
                                     key_index, dtypes.string, value_index),
          default_value)
      table.initializer.run()

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      output = table.lookup(input_values)

      result = output.eval()
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], result)

  def testMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 1
      value_index = 2

      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     key_index, dtypes.int64, value_index),
          default_value)
      table.initializer.run()

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([1, 5, 6], result)

  def testInvalidDataTypeInMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 2
      value_index = 1
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     key_index, dtypes.int64, value_index),
          default_value)
      with self.assertRaisesOpError("is not a valid"):
        table.initializer.run()

  def testInvalidDataType(self):
    vocabulary_file = self._createVocabFile("one_column_3.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup.TextFileIndex.WHOLE_LINE
      value_index = lookup.TextFileIndex.LINE_NUMBER

      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(vocabulary_file, dtypes.int64,
                                       key_index, dtypes.string,
                                       value_index), default_value)

  def testInvalidIndex(self):
    vocabulary_file = self._createVocabFile("one_column_4.txt")
    with self.cached_session():
      default_value = -1
      key_index = 1  # second column of the line
      value_index = lookup.TextFileIndex.LINE_NUMBER
      table = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     key_index, dtypes.int64, value_index),
          default_value)

      with self.assertRaisesOpError("Invalid number of columns"):
        table.initializer.run()

  def testInitializeSameTableWithMultipleNodes(self):
    vocabulary_file = self._createVocabFile("one_column_5.txt")

    with self.cached_session() as sess:
      shared_name = "shared-one-columm"
      default_value = -1
      table1 = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)
      table2 = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)
      table3 = lookup.HashTable(
          lookup.TextFileInitializer(vocabulary_file, dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)

      lookup_ops.tables_initializer().run()

      input_string = constant_op.constant(["brain", "salad", "tank"])

      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testInitializeTableWithNoFilename(self):
    with self.cached_session():
      default_value = -1
      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(
                "", dtypes.string, lookup.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup.TextFileIndex.LINE_NUMBER),
            default_value)

  def testInitializeWithVocabSize(self):
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      vocabulary_file1 = self._createVocabFile("one_column6.txt")
      table1 = lookup.HashTable(
          lookup.TextFileInitializer(
              vocabulary_file1,
              dtypes.string,
              lookup.TextFileIndex.WHOLE_LINE,
              dtypes.int64,
              lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)

      # Initialize from file.
      table1.initializer.run()
      self.assertEquals(vocab_size, table1.size().eval())

      vocabulary_file2 = self._createVocabFile("one_column7.txt")
      vocab_size = 5
      table2 = lookup.HashTable(
          lookup.TextFileInitializer(
              vocabulary_file2,
              dtypes.string,
              lookup.TextFileIndex.WHOLE_LINE,
              dtypes.int64,
              lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)
      with self.assertRaisesOpError("Invalid vocab_size"):
        table2.initializer.run()

      vocab_size = 1
      vocabulary_file3 = self._createVocabFile("one_column3.txt")
      table3 = lookup.HashTable(
          lookup.TextFileInitializer(
              vocabulary_file3,
              dtypes.string,
              lookup.TextFileIndex.WHOLE_LINE,
              dtypes.int64,
              lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)

      # Smaller vocab size reads only vocab_size records.
      table3.initializer.run()
      self.assertEquals(vocab_size, table3.size().eval())

  def testFeedVocabularyName(self):
    vocabulary_file = self._createVocabFile("feed_vocabulary.txt")

    with self.cached_session():
      default_value = -1
      table = lookup.HashTable(
          lookup.TextFileInitializer("old_file.txt", dtypes.string,
                                     lookup.TextFileIndex.WHOLE_LINE,
                                     dtypes.int64,
                                     lookup.TextFileIndex.LINE_NUMBER),
          default_value)

      # Initialize with non existing file (old_file.txt) should fail.
      # TODO(yleon): Update message, which might change per FileSystem.
      with self.assertRaisesOpError("old_file.txt"):
        table.initializer.run()

      # Initialize the model feeding the vocabulary file.
      filenames = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
      table.initializer.run(feed_dict={filenames[0]: vocabulary_file})

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testInvalidFilenames(self):
    vocabulary_file = self._createVocabFile("filename_shape.txt")

    with self.cached_session():
      default_value = -1

      # Invalid data type
      other_type = constant_op.constant(1)
      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(
                other_type, dtypes.string, lookup.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup.TextFileIndex.LINE_NUMBER),
            default_value)

      # Non-scalar filename
      filenames = constant_op.constant([vocabulary_file, vocabulary_file])
      with self.assertRaises(ValueError):
        lookup.HashTable(
            lookup.TextFileInitializer(
                filenames, dtypes.string, lookup.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup.TextFileIndex.LINE_NUMBER),
            default_value)

  def testIdToStringTable(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = "UNK"
      vocab_size = 3
      table = lookup.HashTable(
          lookup.TextFileStringTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)

      table.initializer.run()

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)

      out = table.lookup(input_values)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], out.eval())
      self.assertEquals(vocab_size, table.size().eval())

  def testStringToIdTable(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)
      table.initializer.run()

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, -1], out.eval())
      self.assertEquals(vocab_size, table.size().eval())

  def testInt64ToIdTable(self):
    vocab_file = self._createVocabFile(
        "feat_to_id_3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
          default_value)
      table.initializer.run()

      out = table.lookup(
          constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64))
      self.assertAllEqual((0, 1, 2, -1), out.eval())
      self.assertEquals(vocab_size, table.size().eval())


class IdTableWithHashBucketsTest(test.TestCase):

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testStringIdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value),
          oov_buckets)

      table.initializer.run()

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, 3], out.eval())
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  def testInt32IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt", ("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
              default_value),
          oov_buckets,
          key_dtype=dtypes.int32)

      table.initializer.run()

      values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int32)

      out = table.lookup(values)
      self.assertAllEqual([0, 1, 2, 3], out.eval())
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  def testInt64IdTableWithHashBuckets(self):
    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
              default_value),
          oov_buckets)

      table.initializer.run()

      values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64)

      out = table.lookup(values)
      self.assertAllEqual([0, 1, 2, 3], out.eval())
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  def testStringIdTableWithOnlyHashBucket(self):
    with self.cached_session():
      oov_buckets = 5

      # Set a table that only uses hash buckets, for each input value returns
      # an id calculated by fingerprint("input") mod oov_buckets.
      table = lookup.IdTableWithHashBuckets(None, oov_buckets)
      table.initializer.run()

      values = constant_op.constant(("brain", "salad", "surgery"))

      out = table.lookup(values)
      self.assertAllEqual(
          [
              3,  # fingerprint("brain") mod 5.
              1,  # fingerprint("salad") mod 5.
              4  # fingerprint("surgery") mod 5
          ],
          out.eval())
      self.assertEquals(oov_buckets, table.size().eval())

  def testInt32IdTableWithOnlyHashBucket(self):
    with self.cached_session():
      oov_buckets = 5

      # Set a table that only uses hash buckets, for each input value returns
      # an id calculated by fingerprint("input") mod oov_buckets.
      table = lookup.IdTableWithHashBuckets(
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
          out.eval())
      self.assertEquals(oov_buckets, table.size().eval())

  def testFloat64IdTableWithOnlyHashBucket(self):
    with self.cached_session():
      with self.assertRaisesRegexp(TypeError, "Invalid key_dtype"):
        lookup.IdTableWithHashBuckets(
            None, num_oov_buckets=5, key_dtype=dtypes.float64)

  def testBoolIdTableWithOnlyHashBucket(self):
    with self.cached_session():
      with self.assertRaisesRegexp(TypeError, "Invalid key_dtype"):
        lookup.IdTableWithHashBuckets(
            None, num_oov_buckets=5, key_dtype=dtypes.bool)

  def testIdTableWithHashBucketsWithMultipleInitializers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    with self.cached_session() as sess:
      default_value = -1
      vocab_size = 3
      oov_buckets = 3

      vocab_table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)
      table1 = lookup.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=lookup.FastHashSpec,
          name="table1")

      table2 = lookup.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=lookup.StrongHashSpec((1, 2)),
          name="table2")

      lookup_ops.tables_initializer().run()

      input_string = constant_op.constant(
          ["fruit", "brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string)
      out2 = table2.lookup(input_string)

      out1, out2 = sess.run([out1, out2])
      self.assertAllEqual([5, 0, 1, 2, 5], out1)
      self.assertAllEqual([5, 0, 1, 2, 3], out2)
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())
      test_util.assert_ops_in_graph({
          "table1_Lookup/hash_bucket": "StringToHashBucketFast",
          "table2_Lookup/hash_bucket": "StringToHashBucketStrong",
      }, sess.graph)

  def testIdTableWithHashBucketsInitializationAcrossSessions(self):
    vocab_file = self._createVocabFile("feat_to_id_5.txt")
    shared_name = "across-sessions"
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table1 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value,
              shared_name=shared_name),
          oov_buckets)

      table1.initializer.run()

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string_1)

      self.assertAllEqual([0, 1, 2, 3], out1.eval())
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())

    with self.cached_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1

      # Underlying lookup table already initialized in previous session.
      # No need to call table2.initializer.run()
      table2 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value,
              shared_name=shared_name),
          oov_buckets)

      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out2 = table2.lookup(input_string_2)

      self.assertAllEqual([3, 1, 3], out2.eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())

  def testIdTableWithHashBucketsWithMultipleInitializersDifferentDefault(self):
    vocab_file = self._createVocabFile("feat_to_id_6.txt")
    with self.cached_session() as sess:
      default_value1 = -1
      vocab_size = 3
      oov_buckets = 0
      table1 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value1),
          oov_buckets)

      default_value2 = -2
      table2 = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value2),
          oov_buckets)

      lookup_ops.tables_initializer().run()

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])
      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out1 = table1.lookup(input_string_1)
      out2 = table2.lookup(input_string_2)

      out1, out2 = sess.run([out1, out2])
      self.assertAllEqual([0, 1, 2, -1], out1)
      self.assertAllEqual([-2, 1, -2], out2)
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())

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

      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=3),
              -1),
          1)
      table.initializer.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.KeyValueTensorInitializer(
                  (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64),
              -1),
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

  def testInt64SparseTensor(self):
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    with self.cached_session() as sess:
      sp_features = sparse_tensor.SparseTensor(
          constant_op.constant(input_indices, dtypes.int64),
          constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
          constant_op.constant(input_shape, dtypes.int64))

      table = lookup.IdTableWithHashBuckets(
          lookup.HashTable(
              lookup.KeyValueTensorInitializer(
                  (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64),
              -1),
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
      lookup_table = lookup.HashTable(
          lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)

      with self.assertRaises(TypeError):
        lookup.IdTableWithHashBuckets(
            lookup_table, oov_buckets, hasher_spec=1)

      table = lookup.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=lookup.HasherSpec("my-awesome-hash", None))

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      with self.assertRaises(ValueError):
        table.lookup(input_string)

      with self.assertRaises(ValueError):
        table = lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup.StrongHashSpec([]))

      with self.assertRaises(ValueError):
        table = lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup.StrongHashSpec([1, 2, 3]))

      with self.assertRaises(TypeError):
        table = lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=lookup.StrongHashSpec([None, 2]))


if __name__ == "__main__":
  test.main()
