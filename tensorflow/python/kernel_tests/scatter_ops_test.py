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

"""Tests for tensorflow.ops.tf.scatter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class ScatterTest(tf.test.TestCase):

  def _VariableRankTest(self, np_scatter, tf_scatter):
    np.random.seed(8)
    with self.test_session():
      for indices_shape in (), (2,), (2, 3), (2, 3, 4):
        for extra_shape in (), (5,), (5, 6):
          # Generate random indices with no duplicates for easy numpy comparison
          size = np.prod(indices_shape, dtype=np.int32)
          indices = np.arange(2 * size)
          np.random.shuffle(indices)
          indices = indices[:size].reshape(indices_shape)
          updates = np.random.randn(*(indices_shape + extra_shape))
          old = np.random.randn(*((2 * size,) + extra_shape))
        # Scatter via numpy
        new = old.copy()
        np_scatter(new, indices, updates)
        # Scatter via tensorflow
        ref = tf.Variable(old)
        ref.initializer.run()
        tf_scatter(ref, indices, updates).eval()
        # Compare
        self.assertAllClose(ref.eval(), new)

  def testVariableRankUpdate(self):
    def update(ref, indices, updates):
      ref[indices] = updates
    self._VariableRankTest(update, tf.scatter_update)

  def testVariableRankAdd(self):
    def add(ref, indices, updates):
      ref[indices] += updates
    self._VariableRankTest(add, tf.scatter_add)

  def testVariableRankSub(self):
    def sub(ref, indices, updates):
      ref[indices] -= updates
    self._VariableRankTest(sub, tf.scatter_sub)

  def testBooleanScatterUpdate(self):
    with self.test_session() as session:
      var = tf.Variable([True, False])
      update0 = tf.scatter_update(var, 1, True)
      update1 = tf.scatter_update(var, tf.constant(0, dtype=tf.int64), False)
      var.initializer.run()

      session.run([update0, update1])

      self.assertAllEqual([False, True], var.eval())


if __name__ == "__main__":
  tf.test.main()
