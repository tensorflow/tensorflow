# Copyright 2016 Google Inc. All Rights Reserved.
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
              ["a"],
              [1],
              value_dtype=tf.int64),
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
            tf.contrib.lookup.KeyValueTensorInitializer(keys,
                                                        values), default_val)


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

      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"), feats.eval())

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


if __name__ == "__main__":
  tf.test.main()
