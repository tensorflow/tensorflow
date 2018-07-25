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


def compute_degree(g1, g2, eps=1e-7):
  """Compute the degree between two vectors using their usual inner product."""

  def _dot(u, v):
    return tf.reduce_sum(u * v)

  g1_norm = tf.sqrt(_dot(g1, g1))
  g2_norm = tf.sqrt(_dot(g2, g2))
  if g1_norm.numpy() == 0 and g2_norm.numpy() == 0:
    cosine = 1. - eps
  else:
    g1_norm = 1. if g1_norm.numpy() == 0 else g1_norm
    g2_norm = 1. if g2_norm.numpy() == 0 else g2_norm
    cosine = _dot(g1, g2) / g1_norm / g2_norm
    # Restrict to arccos range
    cosine = tf.minimum(tf.maximum(cosine, eps - 1.), 1. - eps)
  degree = tf.acos(cosine) * 180. / 3.141592653589793

  return degree


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
    input_shape = (8, 8, 128)
    data_shape = (16,) + input_shape
    x = tf.random_normal(shape=data_shape)

    # Stride 1
    block = block_factory(
        filters=128,
        strides=(1, 1),
        input_shape=input_shape,
        data_format="channels_last")
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 8, 8, 128))
    test.assertNotAllClose(y_tr, y_ev)

    # Stride of 2
    block = block_factory(
        filters=128,
        strides=(2, 2),
        input_shape=input_shape,
        data_format="channels_last")
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 4, 4, 128))
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
    input_shape = (128, 8, 8)
    data_shape = (16,) + input_shape
    x = tf.random_normal(shape=data_shape)

    # Stride of 1
    block = block_factory(filters=128, strides=(1, 1), input_shape=input_shape)
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 128, 8, 8))
    test.assertNotAllClose(y_tr, y_ev)

    # Stride of 2
    block = block_factory(filters=128, strides=(2, 2), input_shape=input_shape)
    y_tr, y_ev = block(x, training=True), block(x, training=False)
    test.assertEqual(y_tr.shape, y_ev.shape)
    test.assertEqual(y_ev.shape, (16, 128, 4, 4))
    test.assertNotAllClose(y_tr, y_ev)


class RevBlockTest(tf.test.TestCase):

  def test_call_channels_first(self):
    """Test `call` function with `channels_first` data format."""
    if not tf.test.is_gpu_available():
      self.skipTest("GPU not available")

    with tf.device("/gpu:0"):  # Default NCHW format
      input_shape = (128, 8, 8)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)

      # Stride of 1
      block = blocks.RevBlock(
          n_res=3, filters=128, strides=(1, 1), input_shape=input_shape)
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, (16, 128, 8, 8))
      self.assertNotAllClose(y_tr, y_ev)

      # Stride of 2
      block = blocks.RevBlock(
          n_res=3, filters=128, strides=(2, 2), input_shape=input_shape)
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, [16, 128, 4, 4])
      self.assertNotAllClose(y_tr, y_ev)

  def test_call_channels_last(self):
    """Test `call` function with `channels_last` data format."""
    with tf.device("/cpu:0"):  # NHWC format
      input_shape = (8, 8, 128)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape)

      # Stride 1
      block = blocks.RevBlock(
          n_res=3,
          filters=128,
          strides=(1, 1),
          input_shape=input_shape,
          data_format="channels_last")
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, (16, 8, 8, 128))
      self.assertNotAllClose(y_tr, y_ev)

      # Stride of 2
      block = blocks.RevBlock(
          n_res=3,
          filters=128,
          strides=(2, 2),
          input_shape=input_shape,
          data_format="channels_last")
      y_tr, y_ev = block(x, training=True), block(x, training=False)
      self.assertEqual(y_tr.shape, y_ev.shape)
      self.assertEqual(y_ev.shape, (16, 4, 4, 128))
      self.assertNotAllClose(y_tr, y_ev)

  def _check_grad_angle(self, grads, grads_true, atol=1e0):
    """Check the angle between two list of vectors are all close."""
    for g1, g2 in zip(grads, grads_true):
      degree = compute_degree(g1, g2)
      self.assertLessEqual(degree, atol)

  def test_backward_grads_and_vars_channels_first(self):
    """Test `backward` function with `channels_first` data format."""
    if not tf.test.is_gpu_available():
      self.skipTest("GPU not available")

    with tf.device("/gpu:0"):  # Default NCHW format
      # Stride 1
      input_shape = (128, 8, 8)
      data_shape = (16,) + input_shape
      x = tf.random_normal(shape=data_shape, dtype=tf.float64)
      dy = tf.random_normal(shape=data_shape, dtype=tf.float64)
      block = blocks.RevBlock(
          n_res=3,
          filters=128,
          strides=(1, 1),
          input_shape=input_shape,
          fused=False,
          dtype=tf.float64)
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = block(x, training=True)
      # Compute grads from reconstruction
      dx, dw, vars_ = block.backward_grads_and_vars(x, y, dy, training=True)
      # Compute true grads
      grads = tape.gradient(y, [x] + vars_, output_gradients=dy)
      dx_true, dw_true = grads[0], grads[1:]
      self.assertAllClose(dx_true, dx)
      self.assertAllClose(dw_true, dw)
      self._check_grad_angle(dx_true, dx)
      self._check_grad_angle(dw_true, dw)

      # Stride 2
      x = tf.random_normal(shape=data_shape, dtype=tf.float64)
      dy = tf.random_normal(shape=(16, 128, 4, 4), dtype=tf.float64)
      block = blocks.RevBlock(
          n_res=3,
          filters=128,
          strides=(2, 2),
          input_shape=input_shape,
          fused=False,
          dtype=tf.float64)
      with tf.GradientTape() as tape:
        tape.watch(x)
        y = block(x, training=True)
      # Compute grads from reconstruction
      dx, dw, vars_ = block.backward_grads_and_vars(x, y, dy, training=True)
      # Compute true grads
      grads = tape.gradient(y, [x] + vars_, output_gradients=dy)
      dx_true, dw_true = grads[0], grads[1:]
      self.assertAllClose(dx_true, dx)
      self.assertAllClose(dw_true, dw)
      self._check_grad_angle(dx_true, dx)
      self._check_grad_angle(dw_true, dw)


class _ResidualTest(tf.test.TestCase):

  def test_call(self):
    """Test `call` function.

    Varying downsampling and data format options.
    """

    _validate_block_call_channels_first(blocks._Residual, self)
    _validate_block_call_channels_last(blocks._Residual, self)

  def test_backward_grads_and_vars_channels_first(self):
    """Test `backward_grads` function with `channels_first` data format."""
    if not tf.test.is_gpu_available():
      self.skipTest("GPU not available")

    with tf.device("/gpu:0"):  # Default NCHW format
      input_shape = (128, 8, 8)
      data_shape = (16,) + input_shape
      # Use double precision for testing
      x_true = tf.random_normal(shape=data_shape, dtype=tf.float64)
      dy = tf.random_normal(shape=data_shape, dtype=tf.float64)
      residual = blocks._Residual(
          filters=128,
          strides=(1, 1),
          input_shape=input_shape,
          fused=False,
          dtype=tf.float64)

      with tf.GradientTape() as tape:
        x_true = tf.identity(x_true)
        tape.watch(x_true)
        y = residual(x_true, training=True)

      # Gradients computed due to reversibility
      x, dx, dw, vars_ = residual.backward_grads_and_vars(
          y, dy=dy, training=True)

      # True gradients computed by the tape
      grads = tape.gradient(y, [x_true] + vars_, output_gradients=dy)
      dx_true, dw_true = grads[0], grads[1:]

      self.assertAllClose(x_true, x)
      self.assertAllClose(dx_true, dx)
      self.assertAllClose(dw_true, dw)


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
