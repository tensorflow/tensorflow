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

# pylint: disable=g-bad-import-order,unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class LinearTest(tf.test.TestCase):

  def testLinear(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(1.0)):
        x = tf.zeros([1, 2])
        l = tf.nn.rnn_cell.linear([x], 2, False)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([l], {x.name: np.array([[1., 2.]])})
        self.assertAllClose(res[0], [[3.0, 3.0]])

        # Checks prevent you from accidentally creating a shared function.
        with self.assertRaises(ValueError) as exc:
          l1 = tf.nn.rnn_cell.linear([x], 2, False)
        self.assertEqual(str(exc.exception)[:12], "Over-sharing")

        # But you can create a new one in a new scope and share the variables.
        with tf.variable_scope("l1") as new_scope:
          l1 = tf.nn.rnn_cell.linear([x], 2, False)
        with tf.variable_scope(new_scope, reuse=True):
          tf.nn.rnn_cell.linear([l1], 2, False)
        self.assertEqual(len(tf.trainable_variables()), 2)


if __name__ == "__main__":
  tf.test.main()
