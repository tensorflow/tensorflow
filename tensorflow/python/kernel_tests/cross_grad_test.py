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
"""Tests for tensorflow.ops.nn_ops.Cross."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class CrossOpTest(tf.test.TestCase):

  def testGradientRandomValues(self):
    with self.test_session():
      us = [2, 3]
      u = tf.reshape([0.854, -0.616, 0.767, 0.725, -0.927, 0.159], shape=us)
      v = tf.reshape([-0.522, 0.755, 0.407, -0.652, 0.241, 0.247], shape=us)
      s = tf.cross(u, v)
      jacob_u, jacob_v = tf.test.compute_gradient([u, v], [us, us], s, us)

    self.assertAllClose(jacob_u[0], jacob_u[1], rtol=1e-3, atol=1e-3)
    self.assertAllClose(jacob_v[0], jacob_v[1], rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  tf.test.main()
