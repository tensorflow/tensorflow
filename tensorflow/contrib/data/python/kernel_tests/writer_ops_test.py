# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.data.python.ops import writers
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class TFRecordWriterTest(test_base.DatasetTestBase):

  def setUp(self):
    super(TFRecordWriterTest, self).setUp()
    self._num_records = 7
    self.filename = array_ops.placeholder(dtypes.string, shape=[])
    self.compression_type = array_ops.placeholder_with_default("", shape=[])

    input_dataset = readers.TFRecordDataset([self.filename],
                                            self.compression_type)
    self.writer = writers.TFRecordWriter(
        self._outputFilename(), self.compression_type).write(input_dataset)

  def _record(self, i):
    return compat.as_bytes("Record %d" % (i))

  def _createFile(self, options=None):
    filename = self._inputFilename()
    writer = python_io.TFRecordWriter(filename, options)
    for i in range(self._num_records):
      writer.write(self._record(i))
    writer.close()
    return filename

  def _inputFilename(self):
    return os.path.join(self.get_temp_dir(), "tf_record.in.txt")

  def _outputFilename(self):
    return os.path.join(self.get_temp_dir(), "tf_record.out.txt")

  def testWrite(self):
    with self.cached_session() as sess:
      sess.run(
          self.writer, feed_dict={
              self.filename: self._createFile(),
          })
    for i, r in enumerate(tf_record.tf_record_iterator(self._outputFilename())):
      self.assertAllEqual(self._record(i), r)

  def testWriteZLIB(self):
    options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.ZLIB)
    with self.cached_session() as sess:
      sess.run(
          self.writer,
          feed_dict={
              self.filename: self._createFile(options),
              self.compression_type: "ZLIB",
          })
    for i, r in enumerate(
        tf_record.tf_record_iterator(self._outputFilename(), options=options)):
      self.assertAllEqual(self._record(i), r)

  def testWriteGZIP(self):
    options = tf_record.TFRecordOptions(tf_record.TFRecordCompressionType.GZIP)
    with self.cached_session() as sess:
      sess.run(
          self.writer,
          feed_dict={
              self.filename: self._createFile(options),
              self.compression_type: "GZIP",
          })
    for i, r in enumerate(
        tf_record.tf_record_iterator(self._outputFilename(), options=options)):
      self.assertAllEqual(self._record(i), r)

  def testFailDataset(self):
    with self.assertRaises(TypeError):
      writers.TFRecordWriter(self._outputFilename(),
                             self.compression_type).write("whoops")

  def testFailDType(self):
    input_dataset = dataset_ops.Dataset.from_tensors(10)
    with self.assertRaises(TypeError):
      writers.TFRecordWriter(self._outputFilename(),
                             self.compression_type).write(input_dataset)

  def testFailShape(self):
    input_dataset = dataset_ops.Dataset.from_tensors([["hello"], ["world"]])
    with self.assertRaises(TypeError):
      writers.TFRecordWriter(self._outputFilename(),
                             self.compression_type).write(input_dataset)


if __name__ == "__main__":
  test.main()
