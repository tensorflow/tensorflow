# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for simple_hash_table."""

import os.path
import tempfile

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.simple_hash_table import simple_hash_table
from tensorflow.python.eager import def_function
# This pylint disable is only needed for internal google users
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class SimpleHashTableTest(tf.test.TestCase, parameterized.TestCase):

  # Helper function using "create, find, insert, find, remove, find
  def _use_table(self, key_dtype, value_dtype):
    hash_table = simple_hash_table.SimpleHashTable(key_dtype, value_dtype, 111)
    result1 = hash_table.find(1, -999)
    hash_table.insert(1, 100)
    result2 = hash_table.find(1, -999)
    hash_table.remove(1)
    result3 = hash_table.find(1, -999)
    results = tf.stack((result1, result2, result3))
    return results  # expect [-999, 100, -999]

  # Test of "create, find, insert, find" in eager mode.
  @parameterized.named_parameters(('int32_float', tf.int32, float),
                                  ('int64_int32', tf.int64, tf.int32))
  def test_find_insert_find_eager(self, key_dtype, value_dtype):
    results = self._use_table(key_dtype, value_dtype)
    self.assertAllClose(results, [-999, 100, -999])

  # Test of "create, find, insert, find" in a tf.function. Note that the
  # creation and use of the ref-counted resource occurs inside a single
  # self.evaluate.
  @parameterized.named_parameters(('int32_float', tf.int32, float),
                                  ('int64_int32', tf.int64, tf.int32))
  def test_find_insert_find_tf_function(self, key_dtype, value_dtype):
    results = def_function.function(
        lambda: self._use_table(key_dtype, value_dtype))
    self.assertAllClose(self.evaluate(results), [-999.0, 100.0, -999.0])

  # strings for key and value
  def test_find_insert_find_strings_eager(self):
    default = 'Default'
    foo = 'Foo'
    bar = 'Bar'
    hash_table = simple_hash_table.SimpleHashTable(tf.string, tf.string,
                                                   default)
    result1 = hash_table.find(foo, default)
    self.assertEqual(result1, default)
    hash_table.insert(foo, bar)
    result2 = hash_table.find(foo, default)
    self.assertEqual(result2, bar)

  def test_export(self):
    table = simple_hash_table.SimpleHashTable(
        tf.int64, tf.int64, default_value=-1)
    table.insert(1, 100)
    table.insert(2, 200)
    table.insert(3, 300)
    keys, values = self.evaluate(table.export())
    self.assertAllEqual(sorted(keys), [1, 2, 3])
    self.assertAllEqual(sorted(values), [100, 200, 300])

  def test_import(self):
    table = simple_hash_table.SimpleHashTable(
        tf.int64, tf.int64, default_value=-1)
    keys = tf.constant([1, 2, 3], dtype=tf.int64)
    values = tf.constant([100, 200, 300], dtype=tf.int64)
    table.do_import(keys, values)
    self.assertEqual(table.find(1), 100)
    self.assertEqual(table.find(2), 200)
    self.assertEqual(table.find(3), 300)
    self.assertEqual(table.find(9), -1)

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self):
    save_dir = os.path.join(self.get_temp_dir(), 'save_restore')
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), 'hash')

    # TODO(b/203097231) is there an alternative that is not __internal__?
    root = tf.__internal__.tracking.AutoTrackable()

    default_value = -1
    root.table = simple_hash_table.SimpleHashTable(
        tf.int64, tf.int64, default_value=default_value)

    @def_function.function(input_signature=[tf.TensorSpec((), tf.int64)])
    def lookup(key):
      return root.table.find(key)

    root.lookup = lookup

    root.table.insert(1, 100)
    root.table.insert(2, 200)
    root.table.insert(3, 300)
    self.assertEqual(root.lookup(2), 200)
    self.assertAllEqual(3, len(self.evaluate(root.table.export()[0])))
    tf.saved_model.save(root, save_path)

    del root
    loaded = tf.saved_model.load(save_path)
    self.assertEqual(loaded.lookup(2), 200)
    self.assertEqual(loaded.lookup(10), -1)


if __name__ == '__main__':
  tf.test.main()
