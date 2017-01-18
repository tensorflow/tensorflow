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
"""Tests for record_input_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


class RecordInputOpTest(test.TestCase):

  def generateTestData(self, prefix, n, m):
    for i in range(n):
      f = os.path.join(self.get_temp_dir(), prefix + "." + str(i))
      w = tf_record.TFRecordWriter(f)

      for j in range(m):
        w.write("{0:0{width}}".format(i * m + j, width=10).encode("utf-8"))

    w.close()

  def testRecordInputSimple(self):
    with self.test_session() as sess:
      self.generateTestData("basic", 1, 1)

      yield_op = data_flow_ops.RecordInput(
          file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
          parallelism=1,
          buffer_size=1,
          batch_size=1,
          name="record_input").get_yield_op()

      self.assertEqual(sess.run(yield_op), b"0000000000")

  def testRecordInputEpochs(self):
    files = 100
    records_per_file = 100
    with self.test_session() as sess:
      self.generateTestData("basic", files, records_per_file)

      records = data_flow_ops.RecordInput(
          file_pattern=os.path.join(self.get_temp_dir(), "basic.*"),
          parallelism=2,
          buffer_size=2000,
          batch_size=1,
          shift_ratio=0.33,
          seed=10,
          name="record_input")

      yield_op = records.get_yield_op()

      # cycle over 3 epochs and make sure we never duplicate
      for _ in range(3):
        epoch_set = set()
        for _ in range(files * records_per_file):
          r = sess.run(yield_op)
          self.assertTrue(r[0] not in epoch_set)
          epoch_set.add(r[0])


if __name__ == "__main__":
  test.main()
