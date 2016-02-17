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

"""Tests for Softplus and SoftplusGrad."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SoftplusTest(tf.test.TestCase):

  def _npSoftplus(self, np_features):
    return np.log(1 + np.exp(np_features))

  def _testSoftplus(self, np_features, use_gpu=False):
    np_softplus = self._npSoftplus(np_features)
    with self.test_session(use_gpu=use_gpu):
      softplus = tf.nn.softplus(np_features)
      tf_softplus = softplus.eval()
    self.assertAllClose(np_softplus, tf_softplus)
    self.assertShapeEqual(np_softplus, softplus)

  def testNumbers(self):
    for t in [np.float, np.double]:
      self._testSoftplus(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=False)
      self._testSoftplus(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
          use_gpu=True)

  def testGradient(self):
    with self.test_session():
      x = tf.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5], name="x")
      y = tf.nn.softplus(x, name="softplus")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32, order="F")
      err = tf.test.compute_gradient_error(x,
                                           [2, 5],
                                           y,
                                           [2, 5],
                                           x_init_value=x_init)
    print("softplus (float) gradient err = ", err)
    self.assertLess(err, 1e-4)


if __name__ == "__main__":
  tf.test.main()
