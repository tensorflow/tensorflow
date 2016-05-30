# Copyright 2015 Google Inc. All Rights Reserved.
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
"""Tests for tf.bitcast."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


class BitcastTest(tf.test.TestCase):

  def _testBitcast(self, x, datatype, shape):
    with self.test_session():
      tf_ans = tf.bitcast(x, datatype)
      out = tf_ans.eval()
      buff_after = memoryview(out).tobytes()
      buff_before = memoryview(x).tobytes()
      self.assertEqual(buff_before, buff_after)
      self.assertEqual(tf_ans.get_shape(), shape)
      self.assertEqual(tf_ans.dtype, datatype)

  def testSmaller(self):
    x = np.random.rand(3, 2)
    datatype = tf.int8
    shape = [3, 2, 8]
    self._testBitcast(x, datatype, shape)

  def testLarger(self):
    x = np.arange(16, dtype=np.int8).reshape([4, 4])
    datatype = tf.int32
    shape = [4]
    self._testBitcast(x, datatype, shape)

  def testSameDtype(self):
    x = np.random.rand(3, 4)
    shape = [3, 4]
    self._testBitcast(x, x.dtype, shape)

  def testSameSize(self):
    x = np.random.rand(3, 4)
    shape = [3, 4]
    self._testBitcast(x, tf.int64, shape)

  def testErrors(self):
    x = np.zeros([1, 1], np.int8)
    datatype = tf.int32
    with self.assertRaisesRegexp(ValueError, "Cannot bitcast due to shape"):
      tf.bitcast(x, datatype, None)

  def testEmpty(self):
    x = np.ones([], np.int32)
    datatype = tf.int8
    shape = [4]
    self._testBitcast(x, datatype, shape)

  def testUnknown(self):
    x = tf.placeholder(tf.float32)
    datatype = tf.int8
    tf.bitcast(x, datatype, None)


if __name__ == "__main__":
  tf.test.main()
