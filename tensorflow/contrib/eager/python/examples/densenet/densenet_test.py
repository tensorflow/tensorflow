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
"""Tests for various Densenet architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.densenet import densenet


class DensenetTest(tf.test.TestCase):

  def test_bottleneck_true(self):
    depth = 7
    growth_rate = 2
    num_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    batch_size = 1

    model = densenet.DenseNet(depth, growth_rate, num_blocks,
                              output_classes, num_layers_in_each_block,
                              bottleneck=True, compression=0.5,
                              weight_decay=1e-4, dropout_rate=0,
                              pool_initial=False, include_top=True)

    rand_input = tf.random_uniform((batch_size, 32, 32, 3))
    output_shape = model(rand_input).shape
    self.assertEqual(output_shape, (batch_size, output_classes))

  def test_bottleneck_false(self):
    depth = 7
    growth_rate = 2
    num_blocks = 3
    output_classes = 10
    num_layers_in_each_block = -1
    batch_size = 1

    model = densenet.DenseNet(depth, growth_rate, num_blocks,
                              output_classes, num_layers_in_each_block,
                              bottleneck=False, compression=0.5,
                              weight_decay=1e-4, dropout_rate=0,
                              pool_initial=False, include_top=True)

    rand_input = tf.random_uniform((batch_size, 32, 32, 3))
    output_shape = model(rand_input).shape
    self.assertEqual(output_shape, (batch_size, output_classes))

  def test_pool_initial_true(self):
    depth = 7
    growth_rate = 2
    num_blocks = 4
    output_classes = 10
    num_layers_in_each_block = [1, 2, 2, 1]
    batch_size = 1

    model = densenet.DenseNet(depth, growth_rate, num_blocks,
                              output_classes, num_layers_in_each_block,
                              bottleneck=True, compression=0.5,
                              weight_decay=1e-4, dropout_rate=0,
                              pool_initial=True, include_top=True)

    rand_input = tf.random_uniform((batch_size, 32, 32, 3))
    output_shape = model(rand_input).shape
    self.assertEqual(output_shape, (batch_size, output_classes))

if __name__ == '__main__':
  tf.enable_eager_execution()
  tf.test.main()
