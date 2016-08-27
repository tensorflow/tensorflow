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
"""Tests for moving_average_optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import six
import tensorflow as tf


class MovingAverageOptimizerTest(tf.test.TestCase):

  def testRun(self):
    for dtype in [tf.half, tf.float32, tf.float64]:
      with self.test_session() as sess:
        orig_val0 = [1.0, 2.0]
        orig_val1 = [3.0, 4.0]
        var0 = tf.Variable(orig_val0, name='var0', dtype=dtype)
        var1 = tf.Variable(orig_val1, name='var1', dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)

        opt = tf.contrib.opt.MovingAverageOptimizer(
            tf.train.GradientDescentOptimizer(learning_rate=2.0),
            average_decay=0.5)
        save_path = os.path.join(self.get_temp_dir(), 'model')
        var_list = [var0, var1]
        update = opt.apply_gradients(
            list(six.moves.zip([grads0, grads1], [var0, var1])))
        train_saver = opt.swapping_saver(var_list)
        inference_saver = tf.train.Saver(var_list)
        tf.initialize_all_variables().run()
        # Fetch params to validate initial values
        # Step 1: the momentum accumulators were. So we should see a normal
        # update: v -= grad * learning_rate
        update.run()
        self.assertAllCloseAccordingToType([0.8, 1.8], var0.eval())
        self.assertAllCloseAccordingToType([2.98, 3.98], var1.eval())
        update.run()
        # Test that the swapping saver save/restore operation is identity.
        self.assertAllCloseAccordingToType([0.6, 1.6], var0.eval())
        self.assertAllCloseAccordingToType([2.96, 3.96], var1.eval())
        train_saver.save(sess, save_path)
        train_saver.restore(sess, save_path)
        val0 = var0.eval()
        val1 = var1.eval()
        self.assertAllCloseAccordingToType([0.6, 1.6], val0)
        self.assertAllCloseAccordingToType([2.96, 3.96], val1)
        # Test that the normal saver will have the averaged variables.
        # We test that the average values are between the original value and the
        # most recent variable values (since they are an average of the two).
        inference_saver.restore(sess, save_path)
        avg_val0 = var0.eval()
        avg_val1 = var1.eval()
        for i in six.moves.range(len(val0)):
          self.assertLess(val0[i], avg_val0[i])
          self.assertLess(avg_val0[i], orig_val0[i])
          self.assertLess(val1[i], avg_val1[i])
          self.assertLess(avg_val1[i], orig_val1[i])

  def testFailWhenSaverCreatedBeforeInitialized(self):
    with self.test_session():
      var = tf.Variable([1.0], name='var', dtype=tf.float32)
      opt = tf.contrib.opt.MovingAverageOptimizer(
          tf.train.GradientDescentOptimizer(learning_rate=2.0))
      # We didn't call apply_gradients yet.
      # This will raise an exception.
      with self.assertRaises(RuntimeError):
        _ = opt.swapping_saver([var])


if __name__ == '__main__':
  tf.test.main()
