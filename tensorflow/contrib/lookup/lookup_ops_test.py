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
import tempfile
import numpy as np
import six
import tensorflow as tf

from tensorflow.python.framework import test_util


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

  def testMultipleSessions(self):
    # Start a server
    server = tf.train.Server(
        {"local0": ["localhost:0"]}, protocol="grpc", start=True)
    # Create two sessions sharing the same state
    session1 = tf.Session(server.target)
    session2 = tf.Session(server.target)

    default_val = -1
    keys = tf.constant(["brain", "salad", "surgery"])
    values = tf.constant([0, 1, 2], tf.int64)
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
        default_val,
        name="t1")

    # Init the table in the first session.
    with session1:
      table.init.run()
      self.assertAllEqual(3, table.size().eval())

    # Init the table in the second session and verify that we do not get a
    # "Table already initialized" error.
    with session2:
      table.init.run()
      self.assertAllEqual(3, table.size().eval())


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

  def testSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.test_session(graph=tf.Graph()) as sess:
      v0 = tf.Variable(10.0, name="v0")
      v1 = tf.Variable(20.0, name="v1")

      default_val = -1
      keys = tf.constant(["b", "c", "d"], tf.string)
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableHashTable(
          tf.string, tf.int64, default_val, name="t1", checkpoint=True)

      save = tf.train.Saver()
      tf.global_variables_initializer().run()

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.test_session(graph=tf.Graph()) as sess:
      v0 = tf.Variable(-1.0, name="v0")
      v1 = tf.Variable(-1.0, name="v1")
      default_val = -1
      table = tf.contrib.lookup.MutableHashTable(
          tf.string, tf.int64, default_val, name="t1", checkpoint=True)
      table.insert(
          tf.constant(["a", "c"], tf.string),
          tf.constant([12, 24], tf.int64)).run()
      self.assertAllEqual(2, table.size().eval())

      save = tf.train.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, v0.eval())
      self.assertEqual(20.0, v1.eval())

      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["a", "b", "c", "d", "e"], tf.string)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], output.eval())

  def testSharing(self):
    # Start a server to store the table state
    server = tf.train.Server(
        {"local0": ["localhost:0"]}, protocol="grpc", start=True)
    # Create two sessions sharing the same state
    session1 = tf.Session(server.target)
    session2 = tf.Session(server.target)

    table = tf.contrib.lookup.MutableHashTable(
        tf.int64, tf.string, "-", name="t1")

    # Populate the table in the first session
    with session1:
      self.assertAllEqual(0, table.size().eval())

      keys = tf.constant([11, 12], tf.int64)
      values = tf.constant(["a", "b"])
      table.insert(keys, values).run()
      self.assertAllEqual(2, table.size().eval())

      output = table.lookup(tf.constant([11, 12, 13], tf.int64))
      self.assertAllEqual([b"a", b"b", b"-"], output.eval())

    # Verify that we can access the shared data from the second session
    with session2:
      self.assertAllEqual(2, table.size().eval())

      output = table.lookup(tf.constant([10, 11, 12], tf.int64))
      self.assertAllEqual([b"-", b"a", b"b"], output.eval())

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


class MutableDenseHashTableOpTest(tf.test.TestCase):

  def testBasic(self):
    with self.test_session():
      keys = tf.constant([11, 12, 13], tf.int64)
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64, tf.int64, default_value=-1, empty_key=0)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant([11, 12, 15], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testLookupUnknownShape(self):
    with self.test_session():
      keys = tf.constant([11, 12, 13], tf.int64)
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64, tf.int64, default_value=-1, empty_key=0)

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      placeholder_keys = tf.placeholder(tf.int64)
      output = table.lookup(placeholder_keys)
      self.assertAllEqual(None, output.get_shape())
      result = output.eval({placeholder_keys: [11, 12, 15]})
      self.assertAllEqual([0, 1, -1], result)

  def testMapStringToFloat(self):
    with self.test_session():
      keys = tf.constant(["a", "b", "c"], tf.string)
      values = tf.constant([0.0, 1.1, 2.2], tf.float32)
      default_value = tf.constant(-1.5, tf.float32)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.string, tf.float32, default_value=default_value, empty_key="")
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(["a", "b", "d"], tf.string)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllClose([0, 1.1, -1.5], result)

  def testMapInt64ToFloat(self):
    for float_dtype in [tf.float32, tf.float64]:
      with self.test_session():
        keys = tf.constant([11, 12, 13], tf.int64)
        values = tf.constant([0.0, 1.1, 2.2], float_dtype)
        default_value = tf.constant(-1.5, float_dtype)
        table = tf.contrib.lookup.MutableDenseHashTable(
            tf.int64, float_dtype, default_value=default_value, empty_key=0)
        self.assertAllEqual(0, table.size().eval())

        table.insert(keys, values).run()
        self.assertAllEqual(3, table.size().eval())

        input_string = tf.constant([11, 12, 15], tf.int64)
        output = table.lookup(input_string)
        self.assertAllEqual([3], output.get_shape())

        result = output.eval()
        self.assertAllClose([0, 1.1, -1.5], result)

  def testVectorValues(self):
    with self.test_session():
      keys = tf.constant([11, 12, 13], tf.int64)
      values = tf.constant([[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]], tf.int64)
      default_value = tf.constant([-1, -2, -3, -4], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=0,
          initial_num_buckets=4)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(4, len(table.export()[0].eval()))

      table.insert(
          tf.constant([14], tf.int64),
          tf.constant([[2, 3, 4, 5]], tf.int64)).run()
      self.assertAllEqual(4, table.size().eval())
      self.assertAllEqual(8, len(table.export()[0].eval()))

      input_string = tf.constant([11, 12, 15], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3, 4], output.get_shape())

      result = output.eval()
      self.assertAllEqual([[0, 1, 2, 3], [3, 4, 5, 6], [-1, -2, -3, -4]],
                          result)

  def testVectorKeys(self):
    with self.test_session():
      keys = tf.constant([[0, 1], [1, 2], [1, 3]], tf.int64)
      values = tf.constant([10, 11, 12], tf.int64)
      empty_key = tf.constant([0, 3], tf.int64)
      default_value = tf.constant(-1, tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=empty_key,
          initial_num_buckets=8)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      table.insert(
          tf.constant([[0, 0]], tf.int64), tf.constant([13], tf.int64)).run()
      self.assertAllEqual(4, table.size().eval())
      self.assertAllEqual(8, len(table.export()[0].eval()))

      input_string = tf.constant([[0, 1], [1, 2], [0, 2]], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([10, 11, -1], result)

  def testResize(self):
    with self.test_session():
      keys = tf.constant([11, 12, 13], tf.int64)
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=-1,
          empty_key=0,
          initial_num_buckets=4)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(4, len(table.export()[0].eval()))

      keys2 = tf.constant([13, 14, 15, 16, 17], tf.int64)
      values2 = tf.constant([3, 4, 5, 6, 7], tf.int64)

      table.insert(keys2, values2).run()
      self.assertAllEqual(7, table.size().eval())
      self.assertAllEqual(16, len(table.export()[0].eval()))

      keys3 = tf.constant([10, 11, 12, 13, 14, 15, 16, 17, 18], tf.int64)
      output = table.lookup(keys3)
      self.assertAllEqual([-1, 0, 1, 3, 4, 5, 6, 7, -1], output.eval())

  def testExport(self):
    with self.test_session():
      keys = tf.constant([11, 12, 13], tf.int64)
      values = tf.constant([1, 2, 3], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=-1,
          empty_key=100,
          initial_num_buckets=8)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      exported_keys, exported_values = table.export()
      self.assertAllEqual([None], exported_keys.get_shape().as_list())
      self.assertAllEqual([None], exported_values.get_shape().as_list())

      np_keys = exported_keys.eval()
      np_values = exported_values.eval()

      self.assertAllEqual(8, len(np_keys))
      self.assertAllEqual(8, len(np_values))

      # pair up keys and values, drop extra added dimension
      pairs = np.dstack((np_keys.flatten(), np_values.flatten()))[0]
      # sort by key
      pairs = pairs[pairs[:, 0].argsort()]
      self.assertAllEqual([[11, 1], [12, 2], [13, 3], [100, 0], [100, 0],
                           [100, 0], [100, 0], [100, 0]], pairs)

  def testSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.test_session(graph=tf.Graph()) as sess:
      default_value = -1
      empty_key = 0
      keys = tf.constant([11, 12, 13], tf.int64)
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32)

      save = tf.train.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.test_session(graph=tf.Graph()) as sess:
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          tf.constant([11, 14], tf.int64),
          tf.constant([12, 24], tf.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = tf.train.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = tf.constant([10, 11, 12, 13, 14], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], output.eval())

  def testVectorSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "vector_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.test_session(graph=tf.Graph()) as sess:
      empty_key = tf.constant([11, 13], tf.int64)
      default_value = tf.constant([-1, -2], tf.int64)
      keys = tf.constant([[11, 12], [11, 14], [13, 14]], tf.int64)
      values = tf.constant([[0, 1], [2, 3], [4, 5]], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32)

      save = tf.train.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.test_session(graph=tf.Graph()) as sess:
      empty_key = tf.constant([11, 13], tf.int64)
      default_value = tf.constant([-1, -2], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          tf.constant([[11, 12], [13, 15]], tf.int64),
          tf.constant([[21, 22], [23, 24]], tf.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = tf.train.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = tf.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -2], [4, 5], [-1, -2]],
                          output.eval())

  def testVectorScalarSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), "vector_scalar_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.test_session(graph=tf.Graph()) as sess:
      empty_key = tf.constant([11, 13], tf.int64)
      default_value = tf.constant(-1, tf.int64)
      keys = tf.constant([[11, 12], [11, 14], [13, 14]], tf.int64)
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=32)

      save = tf.train.Saver()

      self.assertAllEqual(0, table.size().eval())
      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertTrue(isinstance(val, six.string_types))
      self.assertEqual(save_path, val)

    with self.test_session(graph=tf.Graph()) as sess:
      empty_key = tf.constant([11, 13], tf.int64)
      default_value = tf.constant(-1, tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=default_value,
          empty_key=empty_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=64)
      table.insert(
          tf.constant([[11, 12], [13, 15]], tf.int64),
          tf.constant([3, 4], tf.int64)).run()
      self.assertAllEqual(2, table.size().eval())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = tf.train.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size().eval())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = tf.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([0, 1, -1, 2, -1], output.eval())

  def testReprobe(self):
    with self.test_session():
      # Insert 6 keys into a table with 8 buckets.
      # The values are chosen to make sure collisions occur when using GCC STL
      keys = tf.constant([11, 12, 13, 19, 20, 21], tf.int64)
      values = tf.constant([51, 52, 53, 54, 55, 56], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=-1,
          empty_key=0,
          initial_num_buckets=8)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(6, table.size().eval())

      input_string = tf.constant([10, 11, 12, 13, 14, 19, 20, 21, 22], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([9], output.get_shape())

      result = output.eval()
      self.assertAllEqual([-1, 51, 52, 53, -1, 54, 55, 56, -1], result)

  def testCustomEmptyKey(self):
    with self.test_session():
      keys = tf.constant([11, 0, 13], tf.int64)
      values = tf.constant([0, 1, 2], tf.int64)
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64, tf.int64, default_value=-1, empty_key=12)
      self.assertAllEqual(0, table.size().eval())

      table.insert(keys, values).run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant([11, 0, 15], tf.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([3], output.get_shape())

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testErrors(self):
    with self.test_session():
      table = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64, tf.int64, default_value=-1, empty_key=0)

      # Inserting the empty key returns an error
      keys = tf.constant([11, 0], tf.int64)
      values = tf.constant([0, 1], tf.int64)
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "empty_key"):
        table.insert(keys, values).run()

      # Looking up the empty key returns an error
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "empty_key"):
        table.lookup(keys).eval()

      # Arbitrary tensors of keys are not supported
      keys = tf.constant([[11, 0], [12, 1]], tf.int64)
      values = tf.constant([[11, 0], [12, 1]], tf.int64)
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   "Expected key shape"):
        table.lookup(keys).eval()
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   "Expected key shape"):
        table.insert(keys, values).run()

      table2 = tf.contrib.lookup.MutableDenseHashTable(
          tf.int64,
          tf.int64,
          default_value=-1,
          empty_key=17,
          initial_num_buckets=12)
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   "Number of buckets must be"):
        self.assertAllEqual(0, table2.size().eval())


class StringToIndexTableFromFile(tf.test.TestCase):

  def _createVocabFile(self, basename):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["brain", "salad", "surgery"]) + "\n")
    return vocabulary_file

  def test_string_to_index_table_from_file(self):
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1)
      ids = table.lookup(tf.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(tf.OpError, ids.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_string_to_index_table_from_file_with_default_value(self):
    default_value = -42
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      ids = table.lookup(tf.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(tf.OpError, ids.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((1, 2, default_value), ids.eval())

  def test_string_to_index_table_from_file_with_oov_buckets(self):
    vocabulary_file = self._createVocabFile("f2i_vocab3.txt")
    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_file(
          vocabulary_file=vocabulary_file, num_oov_buckets=1000)
      ids = table.lookup(tf.constant(["salad", "surgery", "tarkus", "toccata"]))

      self.assertRaises(tf.OpError, ids.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual(
          (
              1,  # From vocabulary file.
              2,  # From vocabulary file.
              867,  # 3 + fingerprint("tarkus") mod 300.
              860),  # 3 + fingerprint("toccata") mod 300.
          ids.eval())

  def test_string_to_index_table_from_file_with_only_oov_buckets(self):
    self.assertRaises(
        ValueError,
        tf.contrib.lookup.string_to_index_table_from_file,
        vocabulary_file=None)

  def test_string_to_index_table_from_file_with_vocab_size_too_small(self):
    vocabulary_file = self._createVocabFile("f2i_vocab5.txt")
    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=2)
      ids = table.lookup(tf.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(tf.OpError, ids.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((1, -1, -1), ids.eval())
      self.assertEqual(2, table.size().eval())

  def test_string_to_index_table_from_file_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                              "Invalid vocab_size", table.init.run)

  def test_string_to_index_table_from_file_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")

    self.assertRaises(
        ValueError,
        tf.contrib.lookup.string_to_index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      ids = table.lookup(tf.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(tf.OpError, ids.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((1, 2, -1), ids.eval())
      self.assertEqual(3, table.size().eval())

  def test_string_to_index_table_from_file_with_invalid_hashers(self):
    vocabulary_file = self._createVocabFile("invalid_hasher.txt")
    with self.test_session():
      with self.assertRaises(TypeError):
        tf.contrib.lookup.string_to_index_table_from_file(
            vocabulary_file=vocabulary_file,
            vocab_size=3,
            num_oov_buckets=1,
            hasher_spec=1)

      table = tf.contrib.lookup.string_to_index_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=3,
          num_oov_buckets=1,
          hasher_spec=tf.contrib.lookup.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        tf.constant(["salad", "surgery", "tarkus"]))


class StringToIndexTableFromTensor(tf.test.TestCase):

  def test_string_to_index_table_from_tensor_with_tensor_init(self):
    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_tensor(
          mapping=["brain", "salad", "surgery"], num_oov_buckets=1)
      ids = table.lookup(tf.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(tf.OpError, ids.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((1, 2, 3), ids.eval())

  def test_string_to_index_table_from_tensor_with_default_value(self):
    default_value = -42
    with self.test_session():
      table = tf.contrib.lookup.string_to_index_table_from_tensor(
          mapping=["brain", "salad", "surgery"], default_value=default_value)
      ids = table.lookup(tf.constant(["salad", "surgery", "tarkus"]))

      self.assertRaises(tf.OpError, ids.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((1, 2, default_value), ids.eval())

  def test_string_to_index_table_from_tensor_with_only_oov_buckets(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping=None, num_oov_buckets=1)

  def test_string_to_index_table_from_tensor_with_invalid_hashers(self):
    with self.test_session():
      with self.assertRaises(TypeError):
        tf.contrib.lookup.string_to_index_table_from_tensor(
            mapping=["brain", "salad", "surgery"],
            num_oov_buckets=1,
            hasher_spec=1)

      table = tf.contrib.lookup.string_to_index_table_from_tensor(
          mapping=["brain", "salad", "surgery"],
          num_oov_buckets=1,
          hasher_spec=tf.contrib.lookup.HasherSpec("my-awesome-hash", None))

      self.assertRaises(ValueError, table.lookup,
                        tf.constant(["salad", "surgery", "tarkus"]))


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


class IndexToStringTableFromFileTest(tf.test.TestCase):

  def _createVocabFile(self, basename):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["brain", "salad", "surgery"]) + "\n")
    return vocabulary_file

  def test_index_to_string_table(self):
    vocabulary_file = self._createVocabFile("i2f_vocab1.txt")
    with self.test_session():
      table = tf.contrib.lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file)
      features = table.lookup(tf.constant([0, 1, 2, 3], tf.int64))
      self.assertRaises(tf.OpError, features.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          features.eval())

  def test_index_to_string_table_with_default_value(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.test_session():
      table = tf.contrib.lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, default_value=default_value)
      features = table.lookup(tf.constant([1, 2, 4], tf.int64))
      self.assertRaises(tf.OpError, features.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          features.eval())

  def test_index_to_string_table_with_vocab_size_too_small(self):
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    with self.test_session():
      table = tf.contrib.lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=2,
          default_value=default_value)
      features = table.lookup(tf.constant([1, 2, 4], tf.int64))
      self.assertRaises(tf.OpError, features.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((b"salad", default_value, default_value),
                          features.eval())

  def test_index_to_string_table_with_vocab_size_too_large(self):
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.test_session():
      table = tf.contrib.lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      features = table.lookup(tf.constant([1, 2, 4], tf.int64))

      self.assertRaises(tf.OpError, features.eval)
      init = tf.initialize_all_tables()
      self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                              "Invalid vocab_size", init.run)

  def test_index_to_string_table_with_vocab_size(self):
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.test_session():
      table = tf.contrib.lookup.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=3)
      features = table.lookup(tf.constant([1, 2, 4], tf.int64))

      self.assertRaises(tf.OpError, features.eval)
      tf.initialize_all_tables().run()
      self.assertAllEqual((b"salad", b"surgery", b"UNK"), features.eval())


class IndexToStringTableFromTensorTest(tf.test.TestCase):

  def test_index_to_string_table_from_tensor(self):
    with self.test_session():
      mapping_strings = tf.constant(["brain", "salad", "surgery"])
      table = tf.contrib.lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings)

      indices = tf.constant([0, 1, 2, 3], tf.int64)
      features = table.lookup(indices)
      self.assertRaises(tf.OpError, features.eval)
      tf.initialize_all_tables().run()

      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          features.eval())

  def test_duplicate_entries(self):
    with self.test_session():
      mapping_strings = tf.constant(["hello", "hello"])
      table = tf.contrib.lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings)
      indices = tf.constant([0, 1, 4], tf.int64)
      features = table.lookup(indices)
      tf.initialize_all_tables().run()
      self.assertAllEqual((b"hello", b"hello", b"UNK"), features.eval())

  def test_index_to_string_with_default_value(self):
    default_value = b"NONE"
    with self.test_session():
      mapping_strings = tf.constant(["brain", "salad", "surgery"])
      table = tf.contrib.lookup.index_to_string_table_from_tensor(
          mapping=mapping_strings, default_value=default_value)
      indices = tf.constant([1, 2, 4], tf.int64)
      features = table.lookup(indices)
      self.assertRaises(tf.OpError, features.eval)

      tf.initialize_all_tables().run()
      self.assertAllEqual((b"salad", b"surgery", default_value),
                          features.eval())


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


class IdTableWithHashBucketsTest(tf.test.TestCase):

  def _createVocabFile(self, basename):
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["brain", "salad", "surgery"]) + "\n")
    return vocabulary_file

  def testIdTableWithHashBucketsInit(self):
    vocab_file = self._createVocabFile("feat_to_id_3.txt")
    with self.test_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table = tf.contrib.lookup.IdTableWithHashBuckets(
          tf.contrib.lookup.HashTable(
              tf.contrib.lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value),
          oov_buckets)

      table.init.run()

      input_string = tf.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, 3], out.eval())
      self.assertEquals(vocab_size + oov_buckets, table.size().eval())

  def testIdTableWithOnlyHashBucket(self):
    with self.test_session():
      oov_buckets = 5

      # Set a table that only uses hash buckets, for each input value returns
      # an id calculated by fingerprint("input") mod oov_buckets.
      table = tf.contrib.lookup.IdTableWithHashBuckets(None, oov_buckets)
      table.init.run()

      input_string = tf.constant(["brain", "salad", "surgery"])

      out = table.lookup(input_string)
      self.assertAllEqual(
          [
              3,  # fingerprint("brain") mod 5.
              1,  # fingerprint("salad") mod 5.
              4  # fingerprint("surgery") mod 5
          ],
          out.eval())
      self.assertEquals(oov_buckets, table.size().eval())

  def testIdTableWithHashBucketsWithMultipleInitializers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    with self.test_session() as sess:
      default_value = -1
      vocab_size = 3
      oov_buckets = 3

      vocab_table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)
      table1 = tf.contrib.lookup.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=tf.contrib.lookup.FastHashSpec,
          name="table1")

      table2 = tf.contrib.lookup.IdTableWithHashBuckets(
          vocab_table,
          oov_buckets,
          hasher_spec=tf.contrib.lookup.StrongHashSpec((1, 2)),
          name="table2")

      tf.initialize_all_tables().run()

      input_string = tf.constant(["fruit", "brain", "salad", "surgery", "UNK"])

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
    with self.test_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      table1 = tf.contrib.lookup.IdTableWithHashBuckets(
          tf.contrib.lookup.HashTable(
              tf.contrib.lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value,
              shared_name=shared_name),
          oov_buckets)

      table1.init.run()

      input_string_1 = tf.constant(["brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string_1)

      self.assertAllEqual([0, 1, 2, 3], out1.eval())
      self.assertEquals(vocab_size + oov_buckets, table1.size().eval())

    with self.test_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1

      # Underlying lookup table already initialized in previous session.
      # No need to call table2.init.run()
      table2 = tf.contrib.lookup.IdTableWithHashBuckets(
          tf.contrib.lookup.HashTable(
              tf.contrib.lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value,
              shared_name=shared_name),
          oov_buckets)

      input_string_2 = tf.constant(["fruit", "salad", "UNK"])

      out2 = table2.lookup(input_string_2)

      self.assertAllEqual([3, 1, 3], out2.eval())
      self.assertEquals(vocab_size + oov_buckets, table2.size().eval())

  def testIdTableWithHashBucketsWithMultipleInitializersDifferentDefault(self):
    vocab_file = self._createVocabFile("feat_to_id_6.txt")
    with self.test_session() as sess:
      default_value1 = -1
      vocab_size = 3
      oov_buckets = 0
      table1 = tf.contrib.lookup.IdTableWithHashBuckets(
          tf.contrib.lookup.HashTable(
              tf.contrib.lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value1),
          oov_buckets)

      default_value2 = -2
      table2 = tf.contrib.lookup.IdTableWithHashBuckets(
          tf.contrib.lookup.HashTable(
              tf.contrib.lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=vocab_size),
              default_value2),
          oov_buckets)

      tf.initialize_all_tables().run()

      input_string_1 = tf.constant(["brain", "salad", "surgery", "UNK"])
      input_string_2 = tf.constant(["fruit", "salad", "UNK"])

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
    with self.test_session() as sess:
      sp_features = tf.SparseTensor(
          tf.constant(input_indices, tf.int64),
          tf.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                      tf.string), tf.constant(input_shape, tf.int64))

      table = tf.contrib.lookup.IdTableWithHashBuckets(
          tf.contrib.lookup.HashTable(
              tf.contrib.lookup.TextFileIdTableInitializer(
                  vocab_file, vocab_size=3),
              -1),
          1)
      table.init.run()

      sp_ids = table.lookup(sp_features)

      self.assertAllEqual([5], sp_ids.values._shape_as_list())

      sp_ids_ind, sp_ids_val, sp_ids_shape = sess.run(
          [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

      self.assertAllEqual(input_indices, sp_ids_ind)
      self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
      self.assertAllEqual(input_shape, sp_ids_shape)

  def testIdTableWithHashBucketsWithInvalidHashers(self):
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    with self.test_session():
      default_value = -1
      vocab_size = 3
      oov_buckets = 1
      lookup_table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          default_value)

      with self.assertRaises(TypeError):
        tf.contrib.lookup.IdTableWithHashBuckets(
            lookup_table, oov_buckets, hasher_spec=1)

      table = tf.contrib.lookup.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=tf.contrib.lookup.HasherSpec("my-awesome-hash", None))

      input_string = tf.constant(["brain", "salad", "surgery", "UNK"])

      with self.assertRaises(ValueError):
        table.lookup(input_string)

      with self.assertRaises(ValueError):
        table = tf.contrib.lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=tf.contrib.lookup.StrongHashSpec([]))

      with self.assertRaises(ValueError):
        table = tf.contrib.lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=tf.contrib.lookup.StrongHashSpec([1, 2, 3]))

      with self.assertRaises(TypeError):
        table = tf.contrib.lookup.IdTableWithHashBuckets(
            lookup_table,
            oov_buckets,
            hasher_spec=tf.contrib.lookup.StrongHashSpec([None, 2]))


if __name__ == "__main__":
  tf.test.main()
