"""Tests for lookup table ops from tf."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class HashTableOpTest(tf.test.TestCase):

  def testHashTable(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      # Initialize with keys and values tensors.
      keys = tf.constant(['brain', 'salad', 'surgery'])
      values = tf.constant([0, 1, 2], tf.int64)
      init = table.initialize_from(keys, values)
      init.run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(['brain', 'salad', 'tank'])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableInitWithPythonArrays(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)
      # Empty table.
      self.assertAllEqual(0, table.size().eval())

      # Initialize with keys and values tensors.
      keys = ['brain', 'salad', 'surgery']
      values = [0, 1, 2]
      init = table.initialize_from(keys, values)
      init.run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(['brain', 'salad', 'tank'])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testHashTableInitWithNumPyArrays(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      # Initialize with keys and values tensors.
      keys = np.array(['brain', 'salad', 'surgery'], dtype=np.str)
      values = np.array([0, 1, 2], dtype=np.int64)
      init = table.initialize_from(keys, values)
      init.run()
      self.assertAllEqual(3, table.size().eval())

      input_string = tf.constant(['brain', 'salad', 'tank'])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testMultipleHashTables(self):
    with self.test_session() as sess:
      shared_name = ''
      default_val = -1
      table1 = tf.HashTable(tf.string, tf.int64, default_val, shared_name)
      table2 = tf.HashTable(tf.string, tf.int64, default_val, shared_name)
      table3 = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      keys = tf.constant(['brain', 'salad', 'surgery'])
      values = tf.constant([0, 1, 2], tf.int64)
      table1.initialize_from(keys, values)
      table2.initialize_from(keys, values)
      table3.initialize_from(keys, values)

      tf.initialize_all_tables().run()
      self.assertAllEqual(3, table1.size().eval())
      self.assertAllEqual(3, table2.size().eval())
      self.assertAllEqual(3, table3.size().eval())

      input_string = tf.constant(['brain', 'salad', 'tank'])
      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = sess.run([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testHashTableWithTensorDefault(self):
    with self.test_session():
      shared_name = ''
      default_val = tf.constant(-1, tf.int64)
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      # Initialize with keys and values tensors.
      keys = tf.constant(['brain', 'salad', 'surgery'])
      values = tf.constant([0, 1, 2], tf.int64)
      init = table.initialize_from(keys, values)
      init.run()

      input_string = tf.constant(['brain', 'salad', 'tank'])
      output = table.lookup(input_string)

      result = output.eval()
      self.assertAllEqual([0, 1, -1], result)

  def testSignatureMismatch(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      # Initialize with keys and values tensors.
      keys = tf.constant(['brain', 'salad', 'surgery'])
      values = tf.constant([0, 1, 2], tf.int64)
      init = table.initialize_from(keys, values)
      init.run()

      input_string = tf.constant([1, 2, 3], tf.int64)
      with self.assertRaises(TypeError):
        table.lookup(input_string)

      with self.assertRaises(TypeError):
        tf.HashTable(tf.string, tf.int64, 'UNK', shared_name)

  def testDTypes(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      with self.assertRaises(TypeError):
        tf.HashTable([tf.string], tf.string, default_val, shared_name)

  def testNotInitialized(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      input_string = tf.constant(['brain', 'salad', 'surgery'])
      output = table.lookup(input_string)

      with self.assertRaisesOpError('Table not initialized'):
        output.eval()

  def testInitializeTwice(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      # Initialize with keys and values tensors.
      keys = tf.constant(['brain', 'salad', 'surgery'])
      values = tf.constant([0, 1, 2], tf.int64)
      init = table.initialize_from(keys, values)
      init.run()

      with self.assertRaisesOpError('Table already initialized'):
        init.run()

  def testInitializationWithInvalidDimensions(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      # Initialize with keys and values tensors.
      keys = tf.constant(['brain', 'salad', 'surgery'])
      values = tf.constant([0, 1, 2, 3, 4], tf.int64)
      with self.assertRaises(ValueError):
        table.initialize_from(keys, values)

  def testInitializationWithInvalidDataTypes(self):
    with self.test_session():
      shared_name = ''
      default_val = -1
      table = tf.HashTable(tf.string, tf.int64, default_val, shared_name)

      # Initialize with keys and values tensors.
      keys = [0, 1, 2]
      values = ['brain', 'salad', 'surgery']
      with self.assertRaises(TypeError):
        table.initialize_from(keys, values)


if __name__ == '__main__':
  tf.test.main()
