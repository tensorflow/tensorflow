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
"""Tests for basic ops used in eager mode RevNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import ops
tfe = tf.contrib.eager


class OpsTest(tf.test.TestCase):

  def test_downsample(self):
    """Test `possible_down_sample` function with mock object."""

    batch_size = 100
    # NHWC format
    x = tf.random_normal(shape=[batch_size, 32, 32, 3])
    # HW doesn't change but number of features increased
    y = ops.downsample(x, filters=5, strides=(1, 1), axis=3)
    self.assertEqual(y.shape, [batch_size, 32, 32, 5])
    # Feature map doesn't change but HW reduced
    y = ops.downsample(x, filters=3, strides=(2, 2), axis=3)
    self.assertEqual(y.shape, [batch_size, 16, 16, 3])
    # Number of feature increased and HW reduced
    y = ops.downsample(x, filters=5, strides=(2, 2), axis=3)
    self.assertEqual(y.shape, [batch_size, 16, 16, 5])

    # Test gradient flow
    x = tf.random_normal(shape=[batch_size, 32, 32, 3])
    with tfe.GradientTape() as tape:
      tape.watch(x)
      y = ops.downsample(x, filters=3, strides=(1, 1))
    self.assertEqual(y.shape, x.shape)
    dy = tf.random_normal(shape=[batch_size, 32, 32, 3])
    grad, = tape.gradient(y, [x], output_gradients=[dy])
    self.assertEqual(grad.shape, x.shape)

    # Default NCHW format
    if tf.test.is_gpu_available():
      x = tf.random_normal(shape=[batch_size, 3, 32, 32])
      # HW doesn't change but feature map reduced
      y = ops.downsample(x, filters=5, strides=(1, 1))
      self.assertEqual(y.shape, [batch_size, 5, 32, 32])
      # Feature map doesn't change but HW reduced
      y = ops.downsample(x, filters=3, strides=(2, 2))
      self.assertEqual(y.shape, [batch_size, 3, 16, 16])
      # Both feature map and HW reduced
      y = ops.downsample(x, filters=5, strides=(2, 2))
      self.assertEqual(y.shape, [batch_size, 5, 16, 16])

      # Test gradient flow
      x = tf.random_normal(shape=[batch_size, 3, 32, 32])
      with tfe.GradientTape() as tape:
        tape.watch(x)
        y = ops.downsample(x, filters=3, strides=(1, 1))
      self.assertEqual(y.shape, x.shape)
      dy = tf.random_normal(shape=[batch_size, 3, 32, 32])
      grad, = tape.gradient(y, [x], output_gradients=[dy])
      self.assertEqual(grad.shape, x.shape)


if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
