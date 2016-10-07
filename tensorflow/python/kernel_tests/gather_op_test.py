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

"""Tests for tensorflow.ops.tf.gather."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class GatherTest(tf.test.TestCase):
  use_gpu = False

  def testScalar1D(self):
    with self.test_session(use_gpu=self.use_gpu):
      params = tf.constant([0, 1, 2, 3, 7, 5])
      indices = tf.constant(4)
      gather_t = tf.gather(params, indices)
      gather_val = gather_t.eval()
    self.assertAllEqual(7, gather_val)
    self.assertEqual([], gather_t.get_shape())

  def testScalar2D(self):
    with self.test_session(use_gpu=self.use_gpu):
      params = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                     [9, 10, 11], [12, 13, 14]])
      indices = tf.constant(2)
      gather_t = tf.gather(params, indices)
      gather_val = gather_t.eval()
    self.assertAllEqual([6, 7, 8], gather_val)
    self.assertEqual([3], gather_t.get_shape())

  def testSimpleTwoD32(self):
    with self.test_session(use_gpu=self.use_gpu):
      params = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                                     [9, 10, 11], [12, 13, 14]])
      indices = tf.constant([0, 4, 0, 2])
      gather_t = tf.gather(params, indices)
      gather_val = gather_t.eval()
    self.assertAllEqual([[0, 1, 2], [12, 13, 14], [0, 1, 2], [6, 7, 8]],
                        gather_val)
    self.assertEqual([4, 3], gather_t.get_shape())

  def testHigherRank(self):
    np.random.seed(1)
    # We check that scalar and empty shapes work as well
    for shape in (7, 0), (4, 3, 2):
      for indices_shape in (), (0,), (3, 0), (3, 5):
        params = np.random.randn(*shape)
        indices = np.random.randint(shape[0], size=indices_shape)
        with self.test_session(use_gpu=self.use_gpu):
          tf_params = tf.constant(params)
          tf_indices = tf.constant(indices)
          gather = tf.gather(tf_params, tf_indices)
          self.assertAllEqual(params[indices], gather.eval())
          self.assertEqual(indices.shape + params.shape[1:], gather.get_shape())
          # Test gradients
          gather_grad = np.random.randn(*gather.get_shape().as_list())
          params_grad, indices_grad = tf.gradients(
              gather, [tf_params, tf_indices], gather_grad)
          self.assertEqual(indices_grad, None)
          self.assertEqual(type(params_grad), tf.IndexedSlices)
          params_grad = tf.convert_to_tensor(params_grad)
          correct_params_grad = np.zeros(shape)
          for i, g in zip(indices.flat,
                          gather_grad.reshape((indices.size,) + shape[1:])):
            correct_params_grad[i] += g
          self.assertAllClose(correct_params_grad, params_grad.eval())

  def testUnknownIndices(self):
    params = tf.constant([[0, 1, 2]])
    indices = tf.placeholder(tf.int32)
    gather_t = tf.gather(params, indices)
    self.assertEqual(None, gather_t.get_shape())

  def testBadIndices(self):
    with self.test_session(use_gpu=False):
      params = [0, 1, 2]
      indices = [[7]]
      gather = tf.gather(params, indices)
      with self.assertRaisesOpError(r"indices\[0,0\] = 7 is not in \[0, 3\)"):
        gather.eval()

  def testEmptySlices(self):
    with self.test_session(use_gpu=self.use_gpu):
      for dtype in np.float32, np.float64:
        for itype in np.int32, np.int64:
          params = np.zeros((7, 0), dtype=dtype)
          indices = np.array([3, 4], dtype=itype)
          gather = tf.gather(params, indices)
          self.assertAllEqual(gather.eval(), np.zeros((2, 0)))


class GatherGpuTest(GatherTest):
  use_gpu = True


if __name__ == "__main__":
  tf.test.main()
