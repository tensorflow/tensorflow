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
"""Tests for mnist.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.kfac.examples import mnist


class MnistTest(tf.test.TestCase):

  def testValues(self):
    """Ensure values are in their expected range."""
    with tf.Graph().as_default():
      examples, labels = mnist.load_mnist(
          data_dir=None, num_epochs=1, batch_size=64, use_fake_data=True)

      with self.test_session() as sess:
        examples_, labels_ = sess.run([examples, labels])
        self.assertTrue(np.all((0 <= examples_) & (examples_ < 1)))
        self.assertTrue(np.all((0 <= labels_) & (labels_ < 10)))

  def testFlattenedShapes(self):
    """Ensure images are flattened into their appropriate shape."""
    with tf.Graph().as_default():
      examples, labels = mnist.load_mnist(
          data_dir=None,
          num_epochs=1,
          batch_size=64,
          flatten_images=True,
          use_fake_data=True)

      with self.test_session() as sess:
        examples_, labels_ = sess.run([examples, labels])
        self.assertEqual(examples_.shape, (64, 784))
        self.assertEqual(labels_.shape, (64,))

  def testNotFlattenedShapes(self):
    """Ensure non-flattened images are their appropriate shape."""
    with tf.Graph().as_default():
      examples, labels = mnist.load_mnist(
          data_dir=None,
          num_epochs=1,
          batch_size=64,
          flatten_images=False,
          use_fake_data=True)

      with self.test_session() as sess:
        examples_, labels_ = sess.run([examples, labels])
        self.assertEqual(examples_.shape, (64, 28, 28, 1))
        self.assertEqual(labels_.shape, (64,))


if __name__ == '__main__':
  tf.test.main()
