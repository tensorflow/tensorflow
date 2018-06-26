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
"""Tests for basic building blocks used in eager mode RevNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.eager.python.examples.revnet import blocks


def _validate_block_call_channels_last(block_factory, test):
  """Generic testing function for `channels_last` data format.

  Completes a set of tests varying data format, stride, and batch normalization
  configured train vs test time.
  Args:
    block_factory: constructor of one of blocks.InitBlock, blocks.FinalBlock,
      blocks._ResidualInner
    test: tf.test.TestCase object
  """
  with tf.device("/cpu:0"):  # NHWC format
    input_shape = (224, 224, 32)
    data_shape = (16,) + input_shape
    x = tf.random_normal(shape=data_shape)

    # Stride 1
    block = block_factory(
        filters=64,
        strides=(1, 1),
        input_shape=input_shape,
        data_format="channels_last")
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 224, 224, 64))
    test.assertNotAllClose(y_tr, y_ev)

    # Stride of 2
    block = block_factory(
        filters=64,
        strides=(2, 2),
        input_shape=input_shape,
        data_format="channels_last")
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 112, 112, 64))
    test.assertNotAllClose(y_tr, y_ev)


def _validate_block_call_channels_first(block_factory, test):
  """Generic testing function for `channels_first` data format.

  Completes a set of tests varying data format, stride, and batch normalization
  configured train vs test time.
  Args:
    block_factory: constructor of one of blocks.InitBlock, blocks.FinalBlock,
      blocks._ResidualInner
    test: tf.test.TestCase object
  """
  if not tf.test.is_gpu_available():
    test.skipTest("GPU not available")

  with tf.device("/gpu:0"):  # Default NCHW format
    input_shape = (32, 224, 224)
    data_shape = (16,) + input_shape
    x = tf.random_normal(shape=data_shape)

    # Stride of 1
    block = block_factory(filters=64, strides=(1, 1), input_shape=input_shape)
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 64, 224, 224))
    test.assertNotAllClose(y_tr, y_ev)

    # Stride of 2
    block = block_factory(filters=64, strides=(2, 2), input_shape=input_shape)
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 64, 112, 112))
    test.assertNotAllClose(y_tr, y_ev)


class RevBlockTest(tf.test.TestCase):

  def test_call_channels_first(self):
    """Test `call` function with `channels_first` data format."""
    if not tf.test.is_gpu_available():
      self.skipTest("GPU not available")

    with tf.device("/gpu:0"):  # Default NCHW format
      input_shape = (32, 224, 224)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)

      # Stride of 1
      block = blocks.RevBlock(
          n_res=3, filters=64, strides=(1, 1), input_shape=input_shape)
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, (16, 64, 224, 224))
      self.assertNotAllClose(y_tr, y_ev)

      # Stride of 2
      block = blocks.RevBlock(
          n_res=3, filters=64, strides=(2, 2), input_shape=input_shape)
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, [16, 64, 112, 112])
      self.assertNotAllClose(y_tr, y_ev)

  def test_call_channels_last(self):
    """Test `call` function with `channels_last` data format."""
    with tf.device("/cpu:0"):  # NHWC format
      input_shape = (224, 224, 32)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)

      # Stride 1
      block = blocks.RevBlock(
          n_res=3,
          filters=64,
          strides=(1, 1),
          input_shape=input_shape,
          data_format="channels_last")
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, (16, 224, 224, 64))
      self.assertNotAllClose(y_tr, y_ev)

      # Stride of 2
      block = blocks.RevBlock(
          n_res=3,
          filters=64,
          strides=(2, 2),
          input_shape=input_shape,
          data_format="channels_last")
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, (16, 112, 112, 64))
      self.assertNotAllClose(y_tr, y_ev)

  def test_backward_grads_and_vars_channels_first(self):
    """Test `backward` function with `channels_first` data format."""
    if not tf.test.is_gpu_available():
      self.skipTest("GPU not available")

    with tf.device("/gpu:0"):  # Default NCHW format
      input_shape = (32, 224, 224)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)

      # Stride 1
      y = tf.random_normal(shape=data_shape)
      dy = tf.random_normal(shape=data_shape)
      block = blocks.RevBlock(
          n_res=3, filters=32, strides=(1, 1), input_shape=input_shape)
      dy, grads, vars_ = block.backward_grads_and_vars(x, y, dy)
      self.assertEqual(dy.shape, x.shape)
      self.assertTrue(isinstance(grads, list))
      self.assertTrue(isinstance(vars_, list))

      # Stride 2
      y = tf.random_normal(shape=(16, 32, 112, 112))
      dy = tf.random_normal(shape=(16, 32, 112, 112))
      block = blocks.RevBlock(
          n_res=3, filters=32, strides=(2, 2), input_shape=input_shape)
      dy, grads, vars_ = block.backward_grads_and_vars(x, y, dy)
      self.assertEqual(dy.shape, x.shape)
      self.assertTrue(isinstance(grads, list))
      self.assertTrue(isinstance(vars_, list))

  def test_backward_grads_and_vars_channels_last(self):
    """Test `backward` function with `channels_last` data format."""
    with tf.device("/cpu:0"):  # NHWC format
      input_shape = (224, 224, 32)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)

      # Stride 1
      y = tf.random_normal(shape=data_shape)
      dy = tf.random_normal(shape=data_shape)
      block = blocks.RevBlock(
          n_res=3,
          filters=32,
          strides=(1, 1),
          input_shape=input_shape,
          data_format="channels_last")
      dy, grads, vars_ = block.backward_grads_and_vars(x, y, dy)
      self.assertEqual(dy.shape, x.shape)
      self.assertTrue(isinstance(grads, list))
      self.assertTrue(isinstance(vars_, list))

      # Stride 2
      y = tf.random_normal(shape=(16, 112, 112, 32))
      dy = tf.random_normal(shape=(16, 112, 112, 32))
      block = blocks.RevBlock(
          n_res=3,
          filters=32,
          strides=(2, 2),
          input_shape=input_shape,
          data_format="channels_last")
      dy, grads, vars_ = block.backward_grads_and_vars(x, y, dy)
      self.assertEqual(dy.shape, x.shape)
      self.assertTrue(isinstance(grads, list))
      self.assertTrue(isinstance(vars_, list))


class _ResidualTest(tf.test.TestCase):

  def test_call(self):
    """Test `call` function.

    Varying downsampling and data format options.
    """

    _validate_block_call_channels_first(blocks._Residual, self)
    _validate_block_call_channels_last(blocks._Residual, self)

  def test_backward_channels_first(self):
    """Test `backward` function with `channels_first` data format."""
    if not tf.test.is_gpu_available():
      self.skipTest("GPU not available")

    with tf.device("/gpu:0"):  # Default NCHW format
      input_shape = (16, 224, 224)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)
      residual = blocks._Residual(
          filters=16, strides=(1, 1), input_shape=input_shape)

      y_tr, y_ev = residual(x, training=True), residual(x, training=False)
      x_ = residual.backward(y_ev, training=False)
      self.assertAllClose(x, x_)
      x_ = residual.backward(y_tr, training=True)  # This updates moving avg
      self.assertAllClose(x, x_)

  def test_backward_channels_last(self):
    """Test `backward` function with `channels_last` data format."""
    with tf.device("/cpu:0"):  # NHWC format
      input_shape = (224, 224, 16)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)
      residual = blocks._Residual(
          filters=16,
          strides=(1, 1),
          input_shape=input_shape,
          data_format="channels_last")

      y_tr, y_ev = residual(x, training=True), residual(x, training=False)
      x_ = residual.backward(y_ev, training=False)
      self.assertAllClose(x, x_, rtol=1e-4, atol=1e-4)
      x_ = residual.backward(y_tr, training=True)  # This updates moving avg
      self.assertAllClose(x, x_, rtol=1e-4, atol=1e-4)

  def test_backward_grads_and_vars_channels_first(self):
    """Test `backward_grads` function with `channels_first` data format."""
    if not tf.test.is_gpu_available():
      self.skipTest("GPU not available")

    with tf.device("/gpu:0"):  # Default NCHW format
      input_shape = (16, 224, 224)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)
      dy = tf.random_normal(shape=data_shape)
      residual = blocks._Residual(
          filters=16, strides=(1, 1), input_shape=input_shape)

      vars_and_vals = residual.get_moving_stats()
      dx_tr, grads_tr, vars_tr = residual.backward_grads_and_vars(
          x, dy=dy, training=True)
      dx_ev, grads_ev, vars_ev = residual.backward_grads_and_vars(
          x, dy=dy, training=False)
      self.assertNotAllClose(dx_tr, dx_ev)
      self.assertTrue(isinstance(grads_tr, list))
      self.assertTrue(isinstance(grads_ev, list))
      self.assertTrue(isinstance(vars_tr, list))
      self.assertTrue(isinstance(vars_ev, list))
      for grad_tr, var_tr, grad_ev, var_ev in zip(grads_tr, vars_tr, grads_ev,
                                                  vars_ev):
        self.assertEqual(grad_tr.shape, grad_ev.shape)
        self.assertEqual(var_tr.shape, var_ev.shape)
        self.assertEqual(grad_tr.shape, var_tr.shape)

      # Compare against the true gradient computed by the tape
      residual.restore_moving_stats(vars_and_vals)
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = residual(x, training=True)
      grads = tape.gradient(
          y, [x] + residual.trainable_variables, output_gradients=[dy])
      dx_tr_true, grads_tr_true = grads[0], grads[1:]

      del tape

      self.assertAllClose(dx_tr, dx_tr_true, rtol=1e-4, atol=1e-4)
      self.assertAllClose(grads_tr, grads_tr_true, rtol=1e-4, atol=1e-4)

  def test_backward_grads_and_vars_channels_last(self):
    """Test `backward_grads` function with `channels_last` data format."""
    with tf.device("/cpu:0"):  # NHWC format
      input_shape = (224, 224, 16)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)
      dy = tf.random_normal(shape=data_shape)
      residual = blocks._Residual(
          filters=16,
          strides=(1, 1),
          input_shape=input_shape,
          data_format="channels_last")

      dx_tr, grads_tr, vars_tr = residual.backward_grads_and_vars(
          x, dy=dy, training=True)
      dx_ev, grads_ev, vars_ev = residual.backward_grads_and_vars(
          x, dy=dy, training=False)
      self.assertNotAllClose(dx_tr, dx_ev)
      self.assertTrue(isinstance(grads_tr, list))
      self.assertTrue(isinstance(grads_ev, list))
      self.assertTrue(isinstance(vars_tr, list))
      self.assertTrue(isinstance(vars_ev, list))
      for grad_tr, var_tr, grad_ev, var_ev in zip(grads_tr, vars_tr, grads_ev,
                                                  vars_ev):
        self.assertEqual(grad_tr.shape, grad_ev.shape)
        self.assertEqual(var_tr.shape, var_ev.shape)
        self.assertEqual(grad_tr.shape, var_tr.shape)


class _ResidualInnerTest(tf.test.TestCase):

  def test_call(self):
    """Test `call` function."""

    _validate_block_call_channels_first(blocks._ResidualInner, self)
    _validate_block_call_channels_last(blocks._ResidualInner, self)


class _BottleneckResidualInner(tf.test.TestCase):

  def test_call(self):
    """Test `call` function."""

    _validate_block_call_channels_first(blocks._BottleneckResidualInner, self)
    _validate_block_call_channels_last(blocks._BottleneckResidualInner, self)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
