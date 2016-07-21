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
"""Tests for bucketization_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BucketizationOpTest(tf.test.TestCase):

  def test_normal_usecase(self):
    op = tf.contrib.layers.bucketize(
        tf.constant([-5, 0, 2, 3, 5, 8, 10, 11, 12]),
        boundaries=[0, 3, 8, 11])
    expected_out = [0, 1, 1, 2, 2, 3, 3, 4, 4]
    with self.test_session() as sess:
      self.assertAllEqual(expected_out, sess.run(op))

  def test_invalid_boundaries_order(self):
    op = tf.contrib.layers.bucketize(
        tf.constant([-5, 0]), boundaries=[0, 8, 3, 11])
    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(op)


if __name__ == "__main__":
  tf.test.main()
