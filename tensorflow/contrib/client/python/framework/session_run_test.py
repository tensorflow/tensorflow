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
"""session_run tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class SessionRunTest(tf.test.TestCase):

  def testRegisterFetchFeedConversionFuncs(self):
    class SquaredTensor(object):
      def __init__(self, tensor):
        self.sq = tf.square(tensor)

    fetch_fn = lambda squared_tensor: ([squared_tensor.sq], lambda val: val[0])
    feed_fn1 = lambda feed, feed_val: [(feed.sq, feed_val)]
    feed_fn2 = lambda feed: [feed.sq]

    tf.contrib.client.register_session_fetch_feed_conversion_functions(
        SquaredTensor, fetch_fn, feed_fn1, feed_fn2)
    with self.assertRaises(ValueError):
      tf.contrib.client.register_session_fetch_feed_conversion_functions(
          SquaredTensor, fetch_fn, feed_fn1, feed_fn2)
    with self.test_session() as sess:
      np1 = np.array([1.0, 1.5, 2.0, 2.5])
      np2 = np.array([3.0, 3.5, 4.0, 4.5])
      squared_tensor = SquaredTensor(np2)
      squared_eval = sess.run(squared_tensor)
      self.assertAllClose(np2 * np2, squared_eval)
      squared_eval = sess.run(squared_tensor, feed_dict={
        squared_tensor : np1 * np1})
      self.assertAllClose(np1 * np1, squared_eval)

      partial_run = sess.partial_run_setup([squared_tensor], [])
      squared_eval = sess.partial_run(partial_run, squared_tensor)
      self.assertAllClose(np2 * np2, squared_eval)

if __name__ == '__main__':
  tf.test.main()
