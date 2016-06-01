# pylint: disable=g-bad-file-header
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
"""Dropout tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import ops


class DropoutTest(tf.test.TestCase):
  """Dropout tests."""

  def test_dropout_float(self):
    with self.test_session() as session:
      x = tf.placeholder(tf.float32, [5, 5])
      ops.dropout(x, 0.5)
      probs = tf.get_collection(ops.DROPOUTS)
      session.run(tf.initialize_all_variables())
      self.assertEqual(len(probs), 1)
      self.assertEqual(session.run(probs[0]), 0.5)

  def test_dropout_tensor(self):
    with self.test_session():
      x = tf.placeholder(tf.float32, [5, 5])
      y = tf.get_variable("prob", [], initializer=tf.constant_initializer(0.5))
      ops.dropout(x, y)
      probs = tf.get_collection(ops.DROPOUTS)
      self.assertEqual(probs, [y])


if __name__ == "__main__":
  tf.test.main()
