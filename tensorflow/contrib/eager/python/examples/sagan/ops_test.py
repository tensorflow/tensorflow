# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for auxiliary operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.sagan import ops


class OpsTest(tf.test.TestCase):

  def test_flatten_hw(self):
    """Test `flatten_hw` function with mock object."""

    batch_size = 1
    # Default NCHW format
    if tf.test.is_gpu_available():
      x = tf.random_normal(shape=(batch_size, 3, 4, 4))
      y = ops.flatten_hw(x, data_format="channels_first")
      self.assertEqual(y.shape, (batch_size, 4 * 4, 3))

    # NHWC format
    x = tf.random_normal(shape=(batch_size, 4, 4, 3))
    y = ops.flatten_hw(x, data_format="channels_last")
    self.assertEqual(y.shape, (batch_size, 4 * 4, 3))

  def test_broaden_hw(self):
    """Test `broaden_hw` function with mock object."""

    batch_size = 1
    # NHWC format
    x = tf.random_normal(shape=[batch_size, 4 * 4 * 16])
    y = ops.broaden_hw(x, h=4, w=4, c=16, data_format="channels_last")
    self.assertEqual(y.shape, (batch_size, 4, 4, 16))

    # Default NCHW format
    if tf.test.is_gpu_available():
      y = ops.broaden_hw(x, h=4, w=4, c=16, data_format="channels_first")
      self.assertEqual(y.shape, (batch_size, 16, 4, 4))


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
