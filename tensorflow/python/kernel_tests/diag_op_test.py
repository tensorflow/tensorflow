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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf


class GenerateIdentityTensorTest(tf.test.TestCase):

  def diagOp(self, diag, dtype, expected_ans, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.diag(tf.convert_to_tensor(diag.astype(dtype)))
      out = tf_ans.eval()
      tf_ans_inv = tf.diag_part(expected_ans)
      inv_out = tf_ans_inv.eval()
    self.assertAllClose(out, expected_ans)
    self.assertAllClose(inv_out, diag)
    self.assertShapeEqual(expected_ans, tf_ans)
    self.assertShapeEqual(diag, tf_ans_inv)

  def testEmptyTensor(self):
    x = numpy.array([])
    expected_ans = numpy.empty([0, 0])
    self.diagOp(x, numpy.int32, expected_ans)

  def testRankOneIntTensor(self):
    x = numpy.array([1, 2, 3])
    expected_ans = numpy.array(
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]])
    self.diagOp(x, numpy.int32, expected_ans)
    self.diagOp(x, numpy.int64, expected_ans)

  def testRankOneFloatTensor(self):
    x = numpy.array([1.1, 2.2, 3.3])
    expected_ans = numpy.array(
        [[1.1, 0, 0],
         [0, 2.2, 0],
         [0, 0, 3.3]])
    self.diagOp(x, numpy.float32, expected_ans)
    self.diagOp(x, numpy.float64, expected_ans)

  def testRankTwoIntTensor(self):
    x = numpy.array([[1, 2, 3], [4, 5, 6]])
    expected_ans = numpy.array(
        [[[[1, 0, 0], [0, 0, 0]],
          [[0, 2, 0], [0, 0, 0]],
          [[0, 0, 3], [0, 0, 0]]],
         [[[0, 0, 0], [4, 0, 0]],
          [[0, 0, 0], [0, 5, 0]],
          [[0, 0, 0], [0, 0, 6]]]])
    self.diagOp(x, numpy.int32, expected_ans)
    self.diagOp(x, numpy.int64, expected_ans)

  def testRankTwoFloatTensor(self):
    x = numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    expected_ans = numpy.array(
        [[[[1.1, 0, 0], [0, 0, 0]],
          [[0, 2.2, 0], [0, 0, 0]],
          [[0, 0, 3.3], [0, 0, 0]]],
         [[[0, 0, 0], [4.4, 0, 0]],
          [[0, 0, 0], [0, 5.5, 0]],
          [[0, 0, 0], [0, 0, 6.6]]]])
    self.diagOp(x, numpy.float32, expected_ans)
    self.diagOp(x, numpy.float64, expected_ans)

  def testRankThreeFloatTensor(self):
    x = numpy.array([[[1.1, 2.2], [3.3, 4.4]],
                     [[5.5, 6.6], [7.7, 8.8]]])
    expected_ans = numpy.array(
        [[[[[[1.1, 0], [0, 0]], [[0, 0], [0, 0]]],
           [[[0, 2.2], [0, 0]], [[0, 0], [0, 0]]]],
          [[[[0, 0], [3.3, 0]], [[0, 0], [0, 0]]],
           [[[0, 0], [0, 4.4]], [[0, 0], [0, 0]]]]],
         [[[[[0, 0], [0, 0]], [[5.5, 0], [0, 0]]],
           [[[0, 0], [0, 0]], [[0, 6.6], [0, 0]]]],
          [[[[0, 0], [0, 0]], [[0, 0], [7.7, 0]]],
           [[[0, 0], [0, 0]], [[0, 0], [0, 8.8]]]]]])
    self.diagOp(x, numpy.float32, expected_ans)
    self.diagOp(x, numpy.float64, expected_ans)

class DiagPartOpTest(tf.test.TestCase):

  def setUp(self):
    x = numpy.random.seed(0)

  def diagPartOp(self, tensor, dtpe, expected_ans, use_gpu=False):
    with self.test_session(use_gpu=use_gpu):
      tf_ans_inv = tf.diag_part(tensor)
      inv_out = tf_ans_inv.eval()
    self.assertAllClose(inv_out, expected_ans)
    self.assertShapeEqual(expected_ans, tf_ans_inv)

  def testRankTwoFloatTensor(self):
    x = numpy.random.rand(3, 3)
    i = numpy.arange(3)
    expected_ans = x[i, i]
    self.diagPartOp(x, numpy.float32, expected_ans)
    self.diagPartOp(x, numpy.float64, expected_ans)

  def testRankFourFloatTensor(self):
    x = numpy.random.rand(2, 3, 2, 3)
    i = numpy.arange(2)[:, None]
    j = numpy.arange(3)
    expected_ans = x[i, j, i, j]
    self.diagPartOp(x, numpy.float32, expected_ans)
    self.diagPartOp(x, numpy.float64, expected_ans)
    
  def testRankSixFloatTensor(self):
    x = numpy.random.rand(2, 2, 2, 2, 2, 2)
    i = numpy.arange(2)[:, None, None]
    j = numpy.arange(2)[:, None]
    k = numpy.arange(2)
    expected_ans = x[i, j, k, i, j, k]
    self.diagPartOp(x, numpy.float32, expected_ans)
    self.diagPartOp(x, numpy.float64, expected_ans)

  def testOddRank(self):
    w = numpy.random.rand(2)
    x = numpy.random.rand(2, 2, 2)
    y = numpy.random.rand(2, 2, 2, 2, 2)
    z = numpy.random.rand(2, 2, 2, 2, 2, 2, 2)
    self.assertRaises(ValueError, self.diagPartOp, w, numpy.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, numpy.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, y, numpy.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, z, numpy.float32, 0)
    
  def testUnevenDimensions(self):
    w = numpy.random.rand(2, 5)
    x = numpy.random.rand(2, 1, 2, 3)
    y = numpy.random.rand(2, 1, 2, 1, 2, 5)
    z = numpy.random.rand(2, 2, 2, 2, 2, 2, 2, 2)
    self.assertRaises(ValueError, self.diagPartOp, w, numpy.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, x, numpy.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, y, numpy.float32, 0)
    self.assertRaises(ValueError, self.diagPartOp, z, numpy.float32, 0)

if __name__ == "__main__":
  tf.test.main()
