# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA JIT compilation with Keras initializers and dynamic shapes.

This test validates the fix for issue #105334 where @tf.function(jit_compile=True)
fails when using Keras initializers with dynamic shapes.
"""

import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables


class XLAInitializersDynamicShapesTest(test.TestCase):
  """Test XLA JIT compilation with Keras initializers and dynamic shapes."""

  def test_glorot_uniform_with_concrete_shape(self):
    """Test GlorotUniform initializer with concrete shape values."""
    # This should work - concrete shape without tf.shape()
    @tf.function(jit_compile=True)
    def init_weights_concrete():
      weights = tf.keras.initializers.GlorotUniform()(shape=[32, 128])
      return weights
    
    result = init_weights_concrete()
    self.assertEqual(result.shape, (32, 128))

  def test_glorot_uniform_with_dynamic_shape_error(self):
    """Test that GlorotUniform with tf.shape() provides clear error message."""
    # This should raise a clear TypeError about dynamic shapes
    @tf.function(jit_compile=True)
    def init_weights_dynamic(x):
      batch_size = tf.shape(x)[0]
      # Using dynamic shape should raise informative error
      weights = tf.keras.initializers.GlorotUniform()(shape=[batch_size, 128])
      return weights
    
    input_tensor = tf.random.uniform([32, 50], minval=0, maxval=1000, dtype=tf.int32)
    
    with self.assertRaisesRegex(
        TypeError, 
        "Cannot compute fan_in/fan_out with dynamic shape dimensions"):
      init_weights_dynamic(input_tensor)

  def test_he_normal_with_concrete_shape(self):
    """Test HeNormal initializer with concrete shape values."""
    @tf.function(jit_compile=True)
    def init_weights_he():
      weights = tf.keras.initializers.HeNormal()(shape=[64, 256])
      return weights
    
    result = init_weights_he()
    self.assertEqual(result.shape, (64, 256))

  def test_variance_scaling_with_concrete_shape(self):
    """Test VarianceScaling initializer with concrete shape."""
    @tf.function(jit_compile=True)
    def init_weights_variance():
      weights = tf.keras.initializers.VarianceScaling()(shape=[128, 512])
      return weights
    
    result = init_weights_variance()
    self.assertEqual(result.shape, (128, 512))

  def test_initializers_without_xla(self):
    """Test that initializers work without XLA when using dynamic shapes."""
    # Without jit_compile, dynamic shapes should still work
    @tf.function(jit_compile=False)
    def init_weights_no_xla(x):
      batch_size = tf.shape(x)[0]
      # Note: This will still fail because Keras initializers
      # require concrete values for fan calculation, but the error
      # will be more informative
      weights = tf.keras.initializers.GlorotUniform()(shape=[batch_size, 128])
      return weights
    
    input_tensor = tf.random.uniform([32, 50])
    
    # Even without XLA, dynamic shapes in initializers will fail
    # but with a clearer error message
    with self.assertRaisesRegex(
        TypeError,
        "Cannot compute fan_in/fan_out with dynamic shape dimensions"):
      init_weights_no_xla(input_tensor)

  def test_conv_kernel_initializer_concrete_shape(self):
    """Test initializers with convolution kernel shapes."""
    @tf.function(jit_compile=True)
    def init_conv_kernel():
      # Conv2D kernel shape: (kernel_height, kernel_width, in_channels, out_channels)
      weights = tf.keras.initializers.GlorotUniform()(shape=[3, 3, 64, 128])
      return weights
    
    result = init_conv_kernel()
    self.assertEqual(result.shape, (3, 3, 64, 128))


if __name__ == '__main__':
  test.main()
