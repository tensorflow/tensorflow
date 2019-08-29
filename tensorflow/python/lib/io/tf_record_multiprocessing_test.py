# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Multiprocessing tests for TFRecordWriter and tf_record_iterator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import test
from tensorflow.python.util import compat

TFRecordCompressionType = tf_record.TFRecordCompressionType


def ChildProcess(writer, rs):
  for r in rs:
    writer.write(r)
  writer.flush()


class TFRecordWriterCloseAndFlushTests(test.TestCase):
  """TFRecordWriter close and flush tests."""

  # pylint: disable=arguments-differ
  def setUp(self, compression_type=TFRecordCompressionType.NONE):
    super(TFRecordWriterCloseAndFlushTests, self).setUp()
    self._fn = os.path.join(self.get_temp_dir(), "tf_record_writer_test.txt")
    self._options = tf_record.TFRecordOptions(compression_type)
    self._writer = tf_record.TFRecordWriter(self._fn, self._options)
    self._num_records = 20

  def _Record(self, r):
    return compat.as_bytes("Record %d" % r)

  def testFlush(self):
    """test Flush."""
    records = [self._Record(i) for i in range(self._num_records)]

    write_process = multiprocessing.Process(
        target=ChildProcess, args=(self._writer, records))
    write_process.start()
    write_process.join()
    actual = list(tf_record.tf_record_iterator(self._fn, self._options))
    self.assertListEqual(actual, records)


if __name__ == "__main__":
  test.main()
