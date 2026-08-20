# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras convolutional layers."""

from tensorflow.python.keras.layers import convolutional as keras_layers
from tensorflow.python.platform import test


class SeparableConv2DTransposeTest(test.TestCase):
  """Tests for SeparableConv2DTranspose layer."""

  def test_separable_conv2d_transpose_construction(self):
    layer = keras_layers.SeparableConv2DTranspose(
        filters=3,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        output_padding=(1, 1),
        depth_multiplier=2,
        activation='relu',
        use_bias=True)
    
    self.assertEqual(layer.filters, 3)
    self.assertEqual(layer.kernel_size, (3, 3))
    self.assertEqual(layer.strides, (2, 2))
    self.assertEqual(layer.padding, 'same')
    self.assertEqual(layer.output_padding, (1, 1))
    self.assertEqual(layer.depth_multiplier, 2)
    self.assertEqual(layer.activation.__name__, 'relu')
    self.assertTrue(layer.use_bias)

  def test_separable_conv2d_transpose_build(self):
    layer = keras_layers.SeparableConv2DTranspose(
        filters=4,
        kernel_size=(2, 2),
        depth_multiplier=3,
        use_bias=True)
    
    # input_shape: (batch, height, width, channels)
    layer.build(input_shape=(2, 8, 8, 5))
    
    # Check weight shapes
    # depthwise_kernel: [filter_height, filter_width, out_channels, depth_multiplier] => [2, 2, 4, 3]
    self.assertEqual(layer.depthwise_kernel.shape, (2, 2, 4, 3))
    # pointwise_kernel: [1, 1, depth_multiplier * out_channels, in_channels] => [1, 1, 12, 5]
    self.assertEqual(layer.pointwise_kernel.shape, (1, 1, 12, 5))
    # bias: [out_channels] => [4]
    self.assertEqual(layer.bias.shape, (4,))

  def test_separable_conv2d_transpose_invalid_output_padding(self):
    with self.assertRaisesRegex(
        ValueError, 'must be greater than output padding'):
      keras_layers.SeparableConv2DTranspose(
          filters=3,
          kernel_size=2,
          strides=(1, 1),
          output_padding=(2, 2))


if __name__ == '__main__':
  test.main()
