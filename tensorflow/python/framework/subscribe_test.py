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

"""Tests for tf.subscribe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import subscribe
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class SubscribeTest(test_util.TensorFlowTestCase):

  def testSideEffect(self):
    a = tf.constant(1)
    b = tf.constant(1)
    c = tf.add(a, b)
    with tf.control_dependencies([c]):
      d = tf.constant(42)
    n = tf.neg(c)

    shared = []

    def sub(t):
      shared.append(t)
      return t

    c = subscribe.subscribe(c, lambda t: tf.py_func(sub, [t], [t.dtype]))

    with self.test_session() as sess:
      c_out = sess.run([c])
      n_out = sess.run([n])
      d_out = sess.run([d])

    self.assertEquals(n_out, [-2])
    self.assertEquals(c_out, [2])
    self.assertEquals(d_out, [42])
    self.assertEquals(shared, [2, 2, 2])


if __name__ == '__main__':
  googletest.main()
