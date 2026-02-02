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
"""Tests for tf.contrib.lookup.lookup_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


class HashTableOpTest(tf.test.TestCase):

  def testHashTable(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableFindHighRank(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant([["brain", "salad"], ["tank", "tarkus"]])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testHashTableInitWithPythonArrays(self):
    with self.test_session():
      default_val = -1
      keys = ["brain", "salad", "surgery"]
      values = [0, 1, 2]
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys,
                                                      values,
                                                      value_dtype=tf.int64),
          default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableInitWithNumPyArrays(self):
    with self.test_session():
      default_val = -1
      keys = np.array(["brain", "salad", "surgery"], dtype=np.str)
      values = np.array([0, 1, 2], dtype=np.int64)
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table.init.run()

      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testMultipleHashTables(self):
    with self.test_session() as sess:
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)

      table1 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table2 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table3 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)

      tf.initialize_all_tables().run()
      self.assertAllEqual(3, table1.size().eval())
      self.assertAllEqual(3, table2.size().eval())
      self.assertAllEqual(3, table3.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testHashTableWithTensorDefault(self):
    with self.test_session():
      default_val = tf.constant(-1, tf.int64)
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table.init.run()

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableWithSparseTensorInput(self):
    with self.test_session() as sess:
      default_val = tf.constant(-1, tf.int64)
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table.init.run()

      sp_indices = [[0, 0], [0, 1], [1, 0]]
      sp_shape = [2, 2]
      input_tensor = tf.SparseTensor(
          tf.constant(sp_indices, tf.int64),
          tf.constant(["brain", "salad", "tank"]),
          tf.constant(sp_shape, tf.int64))
      output = table.lookup(input_tensor)

      out_indices, out_values, out_shape = sess.run(output)

      self.assertAllEqual([0, 1, -1], out_values)
      self.assertAllEqual(sp_indices, out_indices)
      self.assertAllEqual(sp_shape, out_shape)

  def testSignatureMismatch(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table.init.run()

      input_string = tf.constant([1, 2, 3], tf.int64)
      with self.assertRaises(TypeError):
        table.lookup(input_string)

      with self.assertRaises(TypeError):
        tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, values), "UNK")

  def testDTypes(self):
    with self.test_session():
      default_val = -1
      with self.assertRaises(TypeError):
        tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                ["a"], [1], [tf.string], tf.int64), default_val)

  def testNotInitialized(self):
    with self.test_session():
      default_val = -1
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(
              ["a"], [1], value_dtype=tf.int64),
          default_val)

      input_string = tf.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      with self.assertRaisesOpError("Table not initialized"):
        output.eval()

  def testInitializeTwice(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
          default_val)
      table.init.run()

      with self.assertRaisesOpError("Table already initialized"):
        table.init.run()

  def testInitializationWithInvalidDimensions(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2, 3, 4], tf.int64)

      with self.assertRaises(ValueError):
        tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
            default_val)


class MutableHashTableOpTest(tf.test.TestCase):

  def testMutableHashTable(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

      exported_keys, exported_values = table.export()
      self.assertAllEqual([None], exported_keys.get_shape().as_list())
      self.assertAllEqual([None], exported_values.get_shape().as_list())

      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(exported_keys.eval())
      sorted_values = np.sort(exported_values.eval())
      self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
      self.assertAllEqual([0, 1, 2], sorted_values)

  def testMutableHashTableOfTensors(self):
    with self.test_session():
      default_val = tf.constant([-1, -1], tf.int64)
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([[0, 1], [2, 3], [4, 5]], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string, tf.int64,
                                                 default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)
      self.assertAllEqual([3, 2], output.get_shape())

      result = output.eval()
      self.assertAllEqual([[0, 1], [2, 3], [-1, -1]], result)

      exported_keys, exported_values = table.export()
      self.assertAllEqual([None], exported_keys.get_shape().as_list())
      self.assertAllEqual([None, 2], exported_values.get_shape().as_list())
      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(exported_keys.eval())
      sorted_values = np.sort(exported_values.eval())
      self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
      self.assertAllEqual([[4, 5], [2, 3], [0, 1]], sorted_values)

  def testMutableHashTableExportInsert(self):
    with self.test_session():
      default_val = tf.constant([-1, -1], tf.int64)
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([[0, 1], [2, 3], [4, 5]], tf.int64)
      table1 = tf.contrib.lookup.MutableHashTable(tf.string, tf.int64,
                                                  default_val)
      self.assertAllEqual(0, table1.size().eval())
      table1.insert(keys, values).run()
      self.assertAllEqual(3, table1.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      expected_output = [[0, 1], [2, 3], [-1, -1]]
      output1 = table1.lookup(input_string)
      self.assertAllEqual(expected_output, output1.eval())

      exported_keys, exported_values = table1.export()
      self.assertAllEqual(3, exported_keys.eval().size)
      self.assertAllEqual(6, exported_values.eval().size)

      # Populate a second table from the exported data
      table2 = tf.contrib.lookup.MutableHashTable(tf.string, tf.int64,
                                                  default_val)
      self.assertAllEqual(0, table2.size().eval())
      table2.insert(exported_keys, exported_values).run()
      self.assertAllEqual(3, table2.size().eval())

      # Verify lookup result is still the same
      output2 = table2.lookup(input_string)
      self.assertAllEqual(expected_output, output2.eval())

  def testMutableHashTableOfTensorsInvalidShape(self):
    with self.test_session():
      default_val = tf.constant([-1, -1], tf.int64)
      keys = tf.constant(["brain", "salad", "surgery"])
      table = tf.contrib.lookup.MutableHashTable(tf.string, tf.int64,
                                                 default_val)

      # Shape [6] instead of [3, 2]
      values = tf.constant([0, 1, 2, 3, 4, 5], tf.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Shape [2,3] instead of [3, 2]
      values = tf.constant([[0, 1, 2], [3, 4, 5]], tf.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Shape [2, 2] instead of [3, 2]
      values = tf.constant([[0, 1], [2, 3]], tf.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Shape [3, 1] instead of [3, 2]
      values = tf.constant([[0], [2], [4]], tf.int64)
      with self.assertRaisesOpError("Expected shape"):
        table.insert(keys, values).run()

      # Valid Insert
      values = tf.constant([[0, 1], [2, 3], [4, 5]], tf.int64)
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

  def testMutableHashTableInvalidDefaultValue(self):
    with self.test_session():
      default_val = tf.constant([[-1, -1]], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string, tf.int64,
                                                 default_val)
      with self.assertRaisesOpError("Default value must be a vector"):
        self.assertAllEqual(0, table.size().eval())

  def testMutableHashTableDuplicateInsert(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery", "brain"])
      values = tf.constant([0, 1, 2, 3], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([3, 1, -1], result)

  def testMutableHashTableFindHighRank(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant([["brain", "salad"], ["tank", "tarkus"]])
      output = table.lookup(input_string)
      self.assertAllEqual([2, 2], output.get_shape())

      result = output.eval()
      self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testMutableHashTableInsertHighRank(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant([["brain", "salad"], ["surgery", "tank"]])
      values = tf.constant([[0, 1], [2, 3]], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank", "tarkus"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, 3, -1], result)

  def testMutableHashTableOfTensorsFindHighRank(self):
    with self.test_session():
      default_val = tf.constant([-1, -1, -1], tf.int64)
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string, tf.int64,
                                                 default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant([["brain", "salad"], ["tank", "tarkus"]])
      output = table.lookup(input_string)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = output.eval()
      self.assertAllEqual(
          [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def testMultipleMutableHashTables(self):
    with self.test_session() as sess:
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)

      table1 = tf.contrib.lookup.MutableHashTable(tf.string,
                                                  tf.int64,
                                                  default_val)
      table2 = tf.contrib.lookup.MutableHashTable(tf.string,
                                                  tf.int64,
                                                  default_val)
      table3 = tf.contrib.lookup.MutableHashTable(tf.string,
                                                  tf.int64,
                                                  default_val)
      table1.insert(keys, values).run()
      table2.insert(keys, values).run()
      table3.insert(keys, values).run()

      self.assertAllEqual(3, table1.size().eval())
      self.assertAllEqual(3, table2.size().eval())
      self.assertAllEqual(3, table3.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testMutableHashTableWithTensorDefault(self):
    with self.test_session():
      default_val = tf.constant(-1, tf.int64)
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testSignatureMismatch(self):
    with self.test_session():
      default_val = -1
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.int64,
                                                 default_val)

      # insert with keys of the wrong type
      with self.assertRaises(TypeError):
        table.insert(tf.constant([4, 5, 6]), values).run()

      # insert with values of the wrong type
      with self.assertRaises(TypeError):
        table.insert(keys, tf.constant(["a", "b", "c"])).run()

      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      # lookup with keys of the wrong type
      input_string = tf.constant([1, 2, 3], tf.int64)
      with self.assertRaises(TypeError):
        table.lookup(input_string).eval()

      # default value of the wrong type
      with self.assertRaises(TypeError):
        tf.contrib.lookup.MutableHashTable(tf.string, tf.int64, "UNK")

  def testMutableHashTableStringFloat(self):
    with self.test_session():
      default_val = -1.5
      keys = tf.constant(["brain", "salad", "surgery"])
      values = tf.constant([0, 1.1, 2.2], tf.float32)
      table = tf.contrib.lookup.MutableHashTable(tf.string,
                                                 tf.float32,
                                                 default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllClose([0, 1.1, -1.5], result)

  def testMutableHashTableInt64String(self):
    with self.test_session():
      default_val = "n/a"
      keys = tf.constant([0, 1, 2], tf.int64)
      values = tf.constant(["brain", "salad", "surgery"])
      table = tf.contrib.lookup.MutableHashTable(tf.int64,
                                                 tf.string,
                                                 default_val)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant([0, 1, 3], tf.int64)
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual((b"brain", b"salad", b"n/a"), result)


class StringToIndexTest(tf.test.TestCase):

  def test_string_to_index(self):
    with self.test_session():
      mapping_strings = tf.constant(["brain", "salad", "surgery"])
      feats = tf.constant(["salad", "surgery", "tarkus"])
      indices = tf.contrib.lookup.string_to_index(feats,
                                                  mapping=mapping_strings)

      self.assertRaises(tf.OpError, indices.eval)
      tf.initialize_all_tables().run()

      self.assertAllEqual((1, 2, -1), indices.eval())

  def test_duplicate_entries(self):
    with self.test_session():
      mapping_strings = tf.constant(["hello", "hello"])
      feats = tf.constant(["hello", "hola"])
      indices = tf.contrib.lookup.string_to_index(feats,
                                                  mapping=mapping_strings)

      self.assertRaises(tf.OpError, tf.initialize_all_tables().run)

  def test_string_to_index_with_default_value(self):
    default_value = -42
    with self.test_session():
      mapping_strings = tf.constant(["brain", "salad", "surgery"])
      feats = tf.constant(["salad", "surgery", "tarkus"])
      indices = tf.contrib.lookup.string_to_index(feats,
                                                  mapping=mapping_strings,
                                                  default_value=default_value)
      self.assertRaises(tf.OpError, indices.eval)

      tf.initialize_all_tables().run()
      self.assertAllEqual((1, 2, default_value), indices.eval())


class IndexToStringTest(tf.test.TestCase):

  def test_index_to_string(self):
    with self.test_session():
      mapping_strings = tf.constant(["brain", "salad", "surgery"])
      indices = tf.constant([0, 1, 2, 3], tf.int64)
      feats = tf.contrib.lookup.index_to_string(indices,
                                                mapping=mapping_strings)

      self.assertRaises(tf.OpError, feats.eval)
      tf.initialize_all_tables().run()

      self.assertAllEqual(
          (b"brain", b"salad", b"surgery", b"UNK"), feats.eval())

  def test_duplicate_entries(self):
    with self.test_session():
      mapping_strings = tf.constant(["hello", "hello"])
      indices = tf.constant([0, 1, 4], tf.int64)
      feats = tf.contrib.lookup.index_to_string(indices,
                                                mapping=mapping_strings)
      tf.initialize_all_tables().run()
      self.assertAllEqual((b"hello", b"hello", b"UNK"), feats.eval())

      self.assertRaises(tf.OpError, tf.initialize_all_tables().run)

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    with self.test_session():
      mapping_strings = tf.constant(["brain", "salad", "surgery"])
      indices = tf.constant([1, 2, 4], tf.int64)
      feats = tf.contrib.lookup.index_to_string(indices,
                                                mapping=mapping_strings,
                                                default_value=default_value)
      self.assertRaises(tf.OpError, feats.eval)

      tf.initialize_all_tables().run()
      self.assertAllEqual((b"salad", b"surgery", default_value), feats.eval())


class InitializeTableFromFileOpTest(tf.test.TestCase):

  def _createVocabFile(self, basename):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["brain", "salad", "surgery"]) + "\n")
    return vocabulary_file

  def testInitializeTable(self):
    vocabulary_file = self._createVocabFile("one_column_1.txt")

    with self.test_session():
      default_value = -1
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              vocabulary_file, tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER), default_value)
      table.init.run()

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testInitializeIndexTable(self):
    vocabulary_file = self._createVocabFile("one_column_2.txt")

    with self.test_session():
      default_value = "UNK"
      key_index = tf.contrib.lookup.TextFileIndex.LINE_NUMBER
      value_index = tf.contrib.lookup.TextFileIndex.WHOLE_LINE
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(vocabulary_file, tf.int64,
                                                key_index, tf.string,
                                                value_index), default_value)
      table.init.run()

      input_values = tf.constant([0, 1, 2, 3], tf.int64)
      output = table.lookup(input_values)

      result = output.eval()
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], result)

  def testMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.test_session():
      default_value = -1
      key_index = 1
      value_index = 2

      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(vocabulary_file, tf.string,
                                                key_index, tf.int64,
                                                value_index), default_value)
      table.init.run()

      input_string = tf.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([1, 5, 6], result)

  def testInvalidDataTypeInMultiColumn(self):
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.test_session():
      default_value = -1
      key_index = 2
      value_index = 1
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(vocabulary_file, tf.string,
                                                key_index, tf.int64,
                                                value_index), default_value)
      with self.assertRaisesOpError("is not a valid"):
        table.init.run()

  def testInvalidDataType(self):
    vocabulary_file = self._createVocabFile("one_column_3.txt")

    with self.test_session():
      default_value = "UNK"
      key_index = tf.contrib.lookup.TextFileIndex.WHOLE_LINE
      value_index = tf.contrib.lookup.TextFileIndex.LINE_NUMBER

      with self.assertRaises(ValueError):
        tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(vocabulary_file, tf.int64,
                                                  key_index, tf.string,
                                                  value_index), default_value)

  def testInvalidIndex(self):
    vocabulary_file = self._createVocabFile("one_column_4.txt")
    with self.test_session():
      default_value = -1
      key_index = 1  # second column of the line
      value_index = tf.contrib.lookup.TextFileIndex.LINE_NUMBER
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(vocabulary_file, tf.string,
                                                key_index, tf.int64,
                                                value_index), default_value)

      with self.assertRaisesOpError("Invalid number of columns"):
        table.init.run()

  def testInitializeSameTableWithMultipleNodes(self):
    vocabulary_file = self._createVocabFile("one_column_5.txt")

    with self.test_session() as sess:
      shared_name = "shared-one-columm"
      default_value = -1
      table1 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              vocabulary_file, tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)
      table2 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              vocabulary_file, tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)
      table3 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              vocabulary_file, tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER),
          default_value,
          shared_name=shared_name)

      tf.initialize_all_tables().run()

      input_string = tf.constant(["brain", "salad", "tank"])

      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testInitializeTableWithNoFilename(self):
    with self.test_session():
      default_value = -1
      with self.assertRaises(ValueError):
        tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(
                "", tf.string, tf.contrib.lookup.TextFileIndex.WHOLE_LINE,
                tf.int64, tf.contrib.lookup.TextFileIndex.LINE_NUMBER),
            default_value)

  def testInitializeWithVocabSize(self):
    with self.test_session():
      default_value = -1
      vocab_size = 3
      vocabulary_file1 = self._createVocabFile("one_column6.txt")
      table1 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              vocabulary_file1,
              tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE,
              tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)

      # Initialize from file.
      table1.init.run()
      self.assertEquals(vocab_size, table1.size().eval())

      vocabulary_file2 = self._createVocabFile("one_column7.txt")
      vocab_size = 5
      table2 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              vocabulary_file2,
              tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE,
              tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)
      with self.assertRaisesOpError("Invalid vocab_size"):
        table2.init.run()

      vocab_size = 1
      vocabulary_file3 = self._createVocabFile("one_column3.txt")
      table3 = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              vocabulary_file3,
              tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE,
              tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER,
              vocab_size=vocab_size),
          default_value)

      # Smaller vocab size reads only vocab_size records.
      table3.init.run()
      self.assertEquals(vocab_size, table3.size().eval())

  def testFeedVocabularyName(self):
    vocabulary_file = self._createVocabFile("feed_vocabulary.txt")

    with self.test_session():
      default_value = -1
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileInitializer(
              "old_file.txt", tf.string,
              tf.contrib.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
              tf.contrib.lookup.TextFileIndex.LINE_NUMBER), default_value)

      # Initialize with non existing file (old_file.txt) should fail.
      # TODO(yleon): Update message, which might change per FileSystem.
      with self.assertRaisesOpError("old_file.txt"):
        table.init.run()

      # Initialize the model feeding the vocabulary file.
      filenames = tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)
      table.init.run(feed_dict={filenames[0]: vocabulary_file})

      input_string = tf.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testInvalidFilenames(self):
    vocabulary_file = self._createVocabFile("filename_shape.txt")

    with self.test_session():
      default_value = -1

      # Invalid data type
      other_type = tf.constant(1)
      with self.assertRaises(ValueError):
        tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(
                other_type, tf.string,
                tf.contrib.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
                tf.contrib.lookup.TextFileIndex.LINE_NUMBER), default_value)

      # Non-scalar filename
      filenames = tf.constant([vocabulary_file, vocabulary_file])
      with self.assertRaises(ValueError):
        tf.contrib.lookup.HashTable(
            tf.contrib.lookup.TextFileInitializer(
                filenames, tf.string,
                tf.contrib.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
                tf.contrib.lookup.TextFileIndex.LINE_NUMBER), default_value)

  def testIdToStringTable(self):
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.test_session():
      default_value = "UNK"
      vocab_size = 3
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileStringTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)

      table.init.run()

      input_values = tf.constant([0, 1, 2, 3], tf.int64)

      out = table.lookup(input_values)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], out.eval())
      self.assertEquals(vocab_size, table.size().eval())

  def testStringToIdTable(self):
    vocab_file = self._createVocabFile("feat_to_id_2.txt")
    with self.test_session():
      default_value = -1
      vocab_size = 3
      table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileIdTableInitializer(vocab_file,
                                                       vocab_size=vocab_size),
          default_value)
      table.init.run()

      input_string = tf.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, -1], out.eval())
      self.assertEquals(vocab_size, table.size().eval())


if __name__ == "__main__":
  tf.test.main()
