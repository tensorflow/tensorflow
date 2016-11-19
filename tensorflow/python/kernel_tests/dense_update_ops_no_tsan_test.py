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

"""Tests for state updating ops that may have benign race conditions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class AssignOpTest(tf.test.TestCase):

  # NOTE(mrry): We exclude thess tests from the TSAN TAP target, because they
  #   contain benign and deliberate data races when multiple threads update
  #   the same parameters without a lock.
  def testParallelUpdateWithoutLocking(self):
    with self.test_session() as sess:
      ones_t = tf.fill([1024, 1024], 1.0)
      p = tf.Variable(tf.zeros([1024, 1024]))
      adds = [tf.assign_add(p, ones_t, use_locking=False)
              for _ in range(20)]
      tf.global_variables_initializer().run()

      def run_add(add_op):
        sess.run(add_op)
      threads = [self.checkedThread(target=run_add, args=(add_op,))
                 for add_op in adds]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()
      ones = np.ones((1024, 1024)).astype(np.float32)
      self.assertTrue((vals >= ones).all())
      self.assertTrue((vals <= ones * 20).all())

  def testParallelAssignWithoutLocking(self):
    with self.test_session() as sess:
      ones_t = tf.fill([1024, 1024], float(1))
      p = tf.Variable(tf.zeros([1024, 1024]))
      assigns = [tf.assign(p, tf.mul(ones_t, float(i)), False)
                 for i in range(1, 21)]
      tf.global_variables_initializer().run()

      def run_assign(assign_op):
        sess.run(assign_op)
      threads = [self.checkedThread(target=run_assign, args=(assign_op,))
                 for assign_op in assigns]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

      vals = p.eval()

      # Assert every element is taken from one of the assignments.
      self.assertTrue((vals > 0).all())
      self.assertTrue((vals <= 20).all())


if __name__ == "__main__":
  tf.test.main()
