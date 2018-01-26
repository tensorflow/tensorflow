# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.layers import maxout
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers

from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
import numpy as np

"""
Contains the maxout layer tests
"""


class MaxOutTest(test.TestCase):
  def test_simple(self):
    inputs = random_ops.random_uniform((64, 10, 36), seed=1)
    graph = maxout.maxout(inputs, num_units=3)
    self.assertEqual(graph.get_shape().as_list(), [64, 10, 3])

  def test_fully_connected(self):
    inputs = random_ops.random_uniform((64, 50), seed=1)
    graph = core_layers.dense(inputs, 50)
    graph = maxout.maxout(graph, num_units=10)
    self.assertEqual(graph.get_shape().as_list(), [64, 10])

  def test_nchw(self):
    inputs = random_ops.random_uniform((10, 100, 100, 3), seed=1)
    graph = conv_layers.conv2d(inputs, 10, 3, padding="SAME")
    graph = maxout.maxout(graph, num_units=1)
    self.assertEqual(graph.get_shape().as_list(), [10, 100, 100, 1])

  def test_invalid_shape(self):
    inputs = random_ops.random_uniform((10, 100, 100, 3), seed=1)
    graph = conv_layers.conv2d(inputs, 3, 10, strides=(1, 1))
    with self.assertRaisesRegexp(ValueError, 'number of features'):
      graph = maxout.maxout(graph, num_units=2)

if __name__ == '__main__':
  test.main()
