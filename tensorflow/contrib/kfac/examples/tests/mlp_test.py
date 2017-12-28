# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for mlp.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.kfac.examples import mlp


class MlpTest(tf.test.TestCase):

  def testFcLayer(self):
    with tf.Graph().as_default():
      pre, act, (w, b) = mlp.fc_layer(
          layer_id=1, inputs=tf.zeros([5, 3]), output_size=10)
      self.assertShapeEqual(np.zeros([5, 10]), pre)
      self.assertShapeEqual(np.zeros([5, 10]), act)
      self.assertShapeEqual(np.zeros([3, 10]), tf.convert_to_tensor(w))
      self.assertShapeEqual(np.zeros([10]), tf.convert_to_tensor(b))
      self.assertIsInstance(w, tf.Variable)
      self.assertIsInstance(b, tf.Variable)
      self.assertIn("fc_1/", w.op.name)
      self.assertIn("fc_1/", b.op.name)

  def testTrainMnist(self):
    with tf.Graph().as_default():
      # Ensure model training doesn't crash.
      #
      # Ideally, we should check that accuracy increases as the model converges,
      # but that takes a non-trivial amount of compute.
      mlp.train_mnist(data_dir=None, num_epochs=1, use_fake_data=True)

  def testTrainMnistMultitower(self):
    with tf.Graph().as_default():
      # Ensure model training doesn't crash.
      mlp.train_mnist_multitower(
          data_dir=None, num_epochs=1, num_towers=2, use_fake_data=True)


if __name__ == "__main__":
  tf.test.main()
