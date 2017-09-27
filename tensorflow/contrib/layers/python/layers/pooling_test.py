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
"""Tests for tf.layers.pooling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from random import randint

from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class PoolingTest(test.TestCase):


  def testSpatialPyramidPoolingDefaultDimensionForBins(self):
    height, width, channel = 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(None, height, width, channel))
    layer = pooling_layers.SpatialPyramidPooling()
    output = layer.apply(images)
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    self.assertListEqual(output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])

  def testSpatialPyramidPoolingCustomDimensionForBins(self):
    height, width, channel = 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(None, height, width, channel))
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [None, channel * expected_output_size_for_each_channel])

  def testSpatialPyramidPoolingBatchSizeGiven(self):
    batch_size, height, width, channel = 4, 5, 6, 3
    images = array_ops.placeholder(dtype='float32',
                                   shape=(batch_size, height, width, channel))
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output = layer.apply(images)
    self.assertListEqual(output.get_shape().as_list(), [batch_size, channel * expected_output_size_for_each_channel])

  def testSpatialPyramidPoolingAssertOutDimensionFixedForAnyInput(self):
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5])
    expected_output_size_for_each_channel = sum(d * d for d in layer.dimensions)
    output_arrays = []
    check_for_images = 10
    batch_size, channel = 2, 3
    for _ in range(check_for_images):
      height, width = randint(0, 9), randint(0, 9)
      images = array_ops.placeholder(dtype='float32',
                                     shape=(batch_size, height, width, channel))
      output = layer.apply(images)
      output_arrays.append(output.get_shape().as_list())
    self.assertListEqual(output_arrays,
                         [[batch_size, channel * expected_output_size_for_each_channel]] * check_for_images)

  def testSpatialPyramidPoolingComputeOutputShape(self):
    batch_size, height, width, channel = 4, 5, 6, 3
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5])
    image = array_ops.placeholder(dtype='float32',
                                   shape=(batch_size, height, width, channel))
    output_shape = layer._compute_output_shape(input_shape=image._shape)
    self.assertListEqual(output_shape.as_list() , [None, 200])

  def testSpatialPyramidPoolingMode(self):
    batch_size, height, width, channel = 4, 5, 6, 3
    mode = 'invalid_mode'
    layer = pooling_layers.SpatialPyramidPooling(dimensions=[3, 4, 5], mode=mode)
    images = array_ops.placeholder(dtype='float32',
                                   shape=(batch_size, height, width, channel))
    with self.assertRaisesRegexp(
            ValueError, "Mode must be either 'max' or 'avg'. Got '{}'".format(mode)):
      layer.apply(images)



if __name__ == '__main__':
  test.main()
