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

"""Functional tests for Pack Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class PackOpTest(tf.test.TestCase):

  def testSimple(self):
    np.random.seed(7)
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
          data = np.random.randn(*shape)
          # Convert [data[0], data[1], ...] separately to tensorflow
          # TODO(irving): Remove list() once we handle maps correctly
          xs = list(map(tf.constant, data))
          # Pack back into a single tensorflow tensor
          c = tf.pack(xs)
          self.assertAllEqual(c.eval(), data)

  def testGradients(self):
    np.random.seed(7)
    for use_gpu in False, True:
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape)
        shapes = [shape[1:]] * shape[0]
        with self.test_session(use_gpu=use_gpu):
          # TODO(irving): Remove list() once we handle maps correctly
          xs = list(map(tf.constant, data))
          c = tf.pack(xs)
          err = tf.test.compute_gradient_error(xs, shapes, c, shape)
          self.assertLess(err, 1e-6)

  def testZeroSize(self):
    # Verify that pack doesn't crash for zero size inputs
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        for shape in (0,), (3,0), (0, 3):
          x = np.zeros((2,) + shape)
          p = tf.pack(list(x)).eval()
          self.assertAllEqual(p, x)


if __name__ == "__main__":
  tf.test.main()
