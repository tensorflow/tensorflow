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
"""Tests for imagingvision.intelligence.tensorflow.model_pruning.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.model_pruning.python.layers import core_layers
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class MaskedConvolutionLayerTest(test.TestCase):

  def setUp(self):
    super(MaskedConvolutionLayerTest, self).setUp()
    self.height, self.width = 7, 9

  def testInvalidRank3(self):
    input_tensor = array_ops.ones((self.height, self.width, 3))
    with self.assertRaisesRegexp(ValueError, 'rank'):
      layers.masked_conv2d(input_tensor, 32, 3)

  def testInvalidRank5(self):
    input_tensor = array_ops.ones((8, 8, self.height, self.width, 3))
    with self.assertRaisesRegexp(ValueError, 'rank'):
      layers.masked_conv2d(input_tensor, 32, 3)

  def testSingleConvMaskAdded(self):
    kernel_size = 3
    input_depth, output_depth = 8, 32
    input_tensor = array_ops.ones((8, self.height, self.width, input_depth))
    layers.masked_conv2d(input_tensor, output_depth, kernel_size)

    masks = ops.get_collection(core_layers.MASK_COLLECTION)
    self.assertEqual(len(masks), 1)
    self.assertListEqual(masks[0].get_shape().as_list(),
                         [kernel_size, kernel_size, input_depth, output_depth])

    masked_weight = ops.get_collection(core_layers.MASKED_WEIGHT_COLLECTION)
    self.assertEqual(len(masked_weight), 1)
    self.assertListEqual(masked_weight[0].get_shape().as_list(),
                         [kernel_size, kernel_size, input_depth, output_depth])

  def testMultipleConvMaskAdded(self):
    number_of_layers = 5

    kernel_size = 3
    base_depth = 4
    depth_step = 7

    input_tensor = array_ops.ones((8, self.height, self.width, base_depth))

    top_layer = input_tensor

    for ix in range(number_of_layers):
      top_layer = layers.masked_conv2d(top_layer, base_depth +
                                       (ix + 1) * depth_step, kernel_size)

    masks = ops.get_collection(core_layers.MASK_COLLECTION)
    self.assertEqual(len(masks), number_of_layers)
    for ix in range(number_of_layers):
      self.assertListEqual(masks[ix].get_shape().as_list(), [
          kernel_size, kernel_size, base_depth + ix * depth_step,
          base_depth + (ix + 1) * depth_step
      ])

    masked_weight = ops.get_collection(core_layers.MASKED_WEIGHT_COLLECTION)
    self.assertEqual(len(masked_weight), number_of_layers)
    for ix in range(number_of_layers):
      self.assertListEqual(masked_weight[ix].get_shape().as_list(), [
          kernel_size, kernel_size, base_depth + ix * depth_step,
          base_depth + (ix + 1) * depth_step
      ])


class MaskedFullyConnectedLayerTest(test.TestCase):

  def testSingleFCMaskAdded(self):
    input_depth, output_depth = 8, 32
    input_tensor = array_ops.ones((5, input_depth))
    layers.masked_fully_connected(input_tensor, output_depth)

    masks = ops.get_collection(core_layers.MASK_COLLECTION)
    self.assertEqual(len(masks), 1)
    self.assertListEqual(masks[0].get_shape().as_list(),
                         [input_depth, output_depth])

    masked_weight = ops.get_collection(core_layers.MASKED_WEIGHT_COLLECTION)
    self.assertEqual(len(masked_weight), 1)
    self.assertListEqual(masked_weight[0].get_shape().as_list(),
                         [input_depth, output_depth])

  def testMultipleConvMaskAdded(self):
    number_of_layers = 5

    base_depth = 4
    depth_step = 7

    input_tensor = array_ops.ones((8, base_depth))

    top_layer = input_tensor

    for ix in range(number_of_layers):
      top_layer = layers.masked_fully_connected(top_layer, base_depth +
                                                (ix + 1) * depth_step)

    masks = ops.get_collection(core_layers.MASK_COLLECTION)
    self.assertEqual(len(masks), number_of_layers)
    for ix in range(number_of_layers):
      self.assertListEqual(masks[ix].get_shape().as_list(), [
          base_depth + ix * depth_step, base_depth + (ix + 1) * depth_step
      ])

    masked_weight = ops.get_collection(core_layers.MASKED_WEIGHT_COLLECTION)
    self.assertEqual(len(masked_weight), number_of_layers)
    for ix in range(number_of_layers):
      self.assertListEqual(masked_weight[ix].get_shape().as_list(), [
          base_depth + ix * depth_step, base_depth + (ix + 1) * depth_step
      ])


if __name__ == '__main__':
  test.main()
