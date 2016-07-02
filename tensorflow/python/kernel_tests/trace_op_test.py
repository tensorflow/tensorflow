# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf


class TraceTest(tf.test.TestCase):

  def setUp(self):
    x = numpy.random.seed(0)

  def traceOp(self, x, dtype, expected_ans, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.trace(x.astype(dtype))
      out = tf_ans.eval()
    self.assertAllClose(out, expected_ans)

  def testEmptyTensor(self):
    x = numpy.array([])
    self.assertRaises(ValueError, self.traceOp, x, numpy.float32, 0)

  def testRankOneTensor(self):
    x = numpy.array([1,2,3])
    self.assertRaises(ValueError, self.traceOp, x, numpy.float32, 0)

  def testRankTwoIntTensor(self):
    x = numpy.array(
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]])
    expected_ans = 6
    self.traceOp(x, numpy.int32, expected_ans)
    self.traceOp(x, numpy.int64, expected_ans)

  def testRankTwoFloatTensor(self):
    x = numpy.array(
        [[1.1, 0, 0],
         [0, 2.2, 0],
         [0, 0, 3.3]])
    expected_ans = 6.6
    self.traceOp(x, numpy.float32, expected_ans)
    self.traceOp(x, numpy.float64, expected_ans)

  def testRankThreeFloatTensor(self):
    x = numpy.random.rand(2, 2, 2)
    self.assertRaises(ValueError, self.traceOp, x, numpy.float32, 0)

  def testRankFourFloatTensor(self):
    x = numpy.random.rand(2, 2, 2, 2)
    self.assertRaises(ValueError, self.traceOp, x, numpy.float32, 0)


if __name__ == "__main__":
  tf.test.main()
