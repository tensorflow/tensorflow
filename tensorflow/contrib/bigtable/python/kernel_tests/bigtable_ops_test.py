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
"""Tests for Bigtable Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import bigtable
from tensorflow.contrib.bigtable.ops import gen_bigtable_ops
from tensorflow.contrib.bigtable.ops import gen_bigtable_test_ops
from tensorflow.contrib.util import loader
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.util import compat

_bigtable_so = loader.load_op_library(
    resource_loader.get_path_to_datafile("_bigtable_test.so"))


class BigtableOpsTest(test.TestCase):
  COMMON_ROW_KEYS = ["r1", "r2", "r3"]
  COMMON_VALUES = ["v1", "v2", "v3"]

  def setUp(self):
    self._client = gen_bigtable_test_ops.bigtable_test_client()
    table = gen_bigtable_ops.bigtable_table(self._client, "testtable")
    self._table = bigtable.BigTable("testtable", None, table)

  def _makeSimpleDataset(self):
    output_rows = dataset_ops.Dataset.from_tensor_slices(self.COMMON_ROW_KEYS)
    output_values = dataset_ops.Dataset.from_tensor_slices(self.COMMON_VALUES)
    return dataset_ops.Dataset.zip((output_rows, output_values))

  def _writeCommonValues(self, sess):
    output_ds = self._makeSimpleDataset()
    write_op = self._table.write(output_ds, ["cf1"], ["c1"])
    sess.run(write_op)

  def runReadKeyTest(self, read_ds):
    itr = read_ds.make_initializable_iterator()
    n = itr.get_next()
    expected = list(self.COMMON_ROW_KEYS)
    expected.reverse()
    with self.test_session() as sess:
      self._writeCommonValues(sess)
      sess.run(itr.initializer)
      for i in range(3):
        output = sess.run(n)
        want = expected.pop()
        self.assertEqual(
            compat.as_bytes(want), compat.as_bytes(output),
            "Unequal at step %d: want: %s, got: %s" % (i, want, output))

  def testReadPrefixKeys(self):
    self.runReadKeyTest(self._table.keys_by_prefix_dataset("r"))

  def testReadRangeKeys(self):
    self.runReadKeyTest(self._table.keys_by_range_dataset("r1", "r4"))

  def runScanTest(self, read_ds):
    itr = read_ds.make_initializable_iterator()
    n = itr.get_next()
    expected_keys = list(self.COMMON_ROW_KEYS)
    expected_keys.reverse()
    expected_values = list(self.COMMON_VALUES)
    expected_values.reverse()
    with self.test_session() as sess:
      self._writeCommonValues(sess)
      sess.run(itr.initializer)
      for i in range(3):
        output = sess.run(n)
        want = expected_keys.pop()
        self.assertEqual(
            compat.as_bytes(want), compat.as_bytes(output[0]),
            "Unequal keys at step %d: want: %s, got: %s" % (i, want, output[0]))
        want = expected_values.pop()
        self.assertEqual(
            compat.as_bytes(want), compat.as_bytes(output[1]),
            "Unequal values at step: %d: want: %s, got: %s" % (i, want,
                                                               output[1]))

  def testScanPrefixStringCol(self):
    self.runScanTest(self._table.scan_prefix("r", cf1="c1"))

  def testScanPrefixListCol(self):
    self.runScanTest(self._table.scan_prefix("r", cf1=["c1"]))

  def testScanRangeStringCol(self):
    self.runScanTest(self._table.scan_range("r1", "r4", cf1="c1"))

  def testScanRangeListCol(self):
    self.runScanTest(self._table.scan_range("r1", "r4", cf1=["c1"]))

  def testLookup(self):
    ds = self._table.keys_by_prefix_dataset("r")
    ds = ds.apply(self._table.lookup_columns(cf1="c1"))
    itr = ds.make_initializable_iterator()
    n = itr.get_next()
    expected_keys = list(self.COMMON_ROW_KEYS)
    expected_values = list(self.COMMON_VALUES)
    expected_tuples = zip(expected_keys, expected_values)
    with self.test_session() as sess:
      self._writeCommonValues(sess)
      sess.run(itr.initializer)
      for i, elem in enumerate(expected_tuples):
        output = sess.run(n)
        self.assertEqual(
            compat.as_bytes(elem[0]), compat.as_bytes(output[0]),
            "Unequal keys at step %d: want: %s, got: %s" %
            (i, compat.as_bytes(elem[0]), compat.as_bytes(output[0])))
        self.assertEqual(
            compat.as_bytes(elem[1]), compat.as_bytes(output[1]),
            "Unequal values at step %d: want: %s, got: %s" %
            (i, compat.as_bytes(elem[1]), compat.as_bytes(output[1])))

  def testSampleKeys(self):
    ds = self._table.sample_keys()
    itr = ds.make_initializable_iterator()
    n = itr.get_next()
    expected_key = self.COMMON_ROW_KEYS[0]
    with self.test_session() as sess:
      self._writeCommonValues(sess)
      sess.run(itr.initializer)
      output = sess.run(n)
      self.assertEqual(
          compat.as_bytes(self.COMMON_ROW_KEYS[0]), compat.as_bytes(output),
          "Unequal keys: want: %s, got: %s" % (compat.as_bytes(
              self.COMMON_ROW_KEYS[0]), compat.as_bytes(output)))
      output = sess.run(n)
      self.assertEqual(
          compat.as_bytes(self.COMMON_ROW_KEYS[2]), compat.as_bytes(output),
          "Unequal keys: want: %s, got: %s" % (compat.as_bytes(
              self.COMMON_ROW_KEYS[2]), compat.as_bytes(output)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(n)


if __name__ == "__main__":
  test.main()
