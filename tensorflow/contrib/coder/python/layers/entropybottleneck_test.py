# -*- coding: utf-8 -*-
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
"""Tests of EntropyBottleneck class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.coder.python.layers import entropybottleneck

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class EntropyBottleneckTest(test.TestCase):

  def test_noise(self):
    # Tests that the noise added is uniform noise between -0.5 and 0.5.
    inputs = array_ops.placeholder(dtypes.float32, (None, 1))
    layer = entropybottleneck.EntropyBottleneck()
    noisy, _ = layer(inputs, training=True)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      values = np.linspace(-50, 50, 100)[:, None]
      noisy, = sess.run([noisy], {inputs: values})
      self.assertFalse(np.allclose(values, noisy, rtol=0, atol=.49))
      self.assertAllClose(values, noisy, rtol=0, atol=.5)

  def test_quantization(self):
    # Tests that inputs are quantized to full integer values, even after
    # quantiles have been updated.
    inputs = array_ops.placeholder(dtypes.float32, (None, 1))
    layer = entropybottleneck.EntropyBottleneck(optimize_integer_offset=False)
    quantized, _ = layer(inputs, training=False)
    opt = gradient_descent.GradientDescentOptimizer(learning_rate=1)
    self.assertTrue(len(layer.losses) == 1)
    step = opt.minimize(layer.losses[0])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(step)
      values = np.linspace(-50, 50, 100)[:, None]
      quantized, = sess.run([quantized], {inputs: values})
      self.assertAllClose(np.around(values), quantized, rtol=0, atol=1e-6)

  def test_quantization_optimized_offset(self):
    # Tests that inputs are not quantized to full integer values after quantiles
    # have been updated. However, the difference between input and output should
    # be between -0.5 and 0.5, and the offset must be consistent.
    inputs = array_ops.placeholder(dtypes.float32, (None, 1))
    layer = entropybottleneck.EntropyBottleneck(optimize_integer_offset=True)
    quantized, _ = layer(inputs, training=False)
    opt = gradient_descent.GradientDescentOptimizer(learning_rate=1)
    self.assertTrue(len(layer.losses) == 1)
    step = opt.minimize(layer.losses[0])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(step)
      values = np.linspace(-50, 50, 100)[:, None]
      quantized, = sess.run([quantized], {inputs: values})
      self.assertAllClose(values, quantized, rtol=0, atol=.5)
      diff = np.ravel(np.around(values) - quantized) % 1
      self.assertAllClose(diff, np.full_like(diff, diff[0]), rtol=0, atol=5e-6)
      self.assertNotEqual(diff[0], 0)

  def test_codec(self):
    # Tests that inputs are compressed and decompressed correctly, and quantized
    # to full integer values, even after quantiles have been updated.
    inputs = array_ops.placeholder(dtypes.float32, (1, None, 1))
    layer = entropybottleneck.EntropyBottleneck(
        data_format="channels_last", init_scale=60,
        optimize_integer_offset=False)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, array_ops.shape(inputs)[1:])
    opt = gradient_descent.GradientDescentOptimizer(learning_rate=1)
    self.assertTrue(len(layer.losses) == 1)
    step = opt.minimize(layer.losses[0])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(step)
      self.assertTrue(len(layer.updates) == 1)
      sess.run(layer.updates[0])
      values = np.linspace(-50, 50, 100)[None, :, None]
      decoded, = sess.run([decoded], {inputs: values})
      self.assertAllClose(np.around(values), decoded, rtol=0, atol=1e-6)

  def test_codec_optimized_offset(self):
    # Tests that inputs are compressed and decompressed correctly, and not
    # quantized to full integer values after quantiles have been updated.
    # However, the difference between input and output should be between -0.5
    # and 0.5, and the offset must be consistent.
    inputs = array_ops.placeholder(dtypes.float32, (1, None, 1))
    layer = entropybottleneck.EntropyBottleneck(
        data_format="channels_last", init_scale=60,
        optimize_integer_offset=True)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, array_ops.shape(inputs)[1:])
    opt = gradient_descent.GradientDescentOptimizer(learning_rate=1)
    self.assertTrue(len(layer.losses) == 1)
    step = opt.minimize(layer.losses[0])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(step)
      self.assertTrue(len(layer.updates) == 1)
      sess.run(layer.updates[0])
      values = np.linspace(-50, 50, 100)[None, :, None]
      decoded, = sess.run([decoded], {inputs: values})
      self.assertAllClose(values, decoded, rtol=0, atol=.5)
      diff = np.ravel(np.around(values) - decoded) % 1
      self.assertAllClose(diff, np.full_like(diff, diff[0]), rtol=0, atol=5e-6)
      self.assertNotEqual(diff[0], 0)

  def test_codec_clipping(self):
    # Tests that inputs are compressed and decompressed correctly, and clipped
    # to the expected range.
    inputs = array_ops.placeholder(dtypes.float32, (1, None, 1))
    layer = entropybottleneck.EntropyBottleneck(
        data_format="channels_last", init_scale=40)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, array_ops.shape(inputs)[1:])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertTrue(len(layer.updates) == 1)
      sess.run(layer.updates[0])
      values = np.linspace(-50, 50, 100)[None, :, None]
      decoded, = sess.run([decoded], {inputs: values})
      expected = np.clip(np.around(values), -40, 40)
      self.assertAllClose(expected, decoded, rtol=0, atol=1e-6)

  def test_channels_last(self):
    # Test the layer with more than one channel and multiple input dimensions,
    # with the channels in the last dimension.
    inputs = array_ops.placeholder(dtypes.float32, (None, None, None, 2))
    layer = entropybottleneck.EntropyBottleneck(
        data_format="channels_last", init_scale=50)
    noisy, _ = layer(inputs, training=True)
    quantized, _ = layer(inputs, training=False)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, array_ops.shape(inputs)[1:])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertTrue(len(layer.updates) == 1)
      sess.run(layer.updates[0])
      values = 5 * np.random.normal(size=(7, 5, 3, 2))
      noisy, quantized, decoded = sess.run(
          [noisy, quantized, decoded], {inputs: values})
      self.assertAllClose(values, noisy, rtol=0, atol=.5)
      self.assertAllClose(values, quantized, rtol=0, atol=.5)
      self.assertAllClose(values, decoded, rtol=0, atol=.5)

  def test_channels_first(self):
    # Test the layer with more than one channel and multiple input dimensions,
    # with the channel dimension right after the batch dimension.
    inputs = array_ops.placeholder(dtypes.float32, (None, 3, None, None))
    layer = entropybottleneck.EntropyBottleneck(
        data_format="channels_first", init_scale=50)
    noisy, _ = layer(inputs, training=True)
    quantized, _ = layer(inputs, training=False)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, array_ops.shape(inputs)[1:])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertTrue(len(layer.updates) == 1)
      sess.run(layer.updates[0])
      values = 5 * np.random.normal(size=(2, 3, 5, 7))
      noisy, quantized, decoded = sess.run(
          [noisy, quantized, decoded], {inputs: values})
      self.assertAllClose(values, noisy, rtol=0, atol=.5)
      self.assertAllClose(values, quantized, rtol=0, atol=.5)
      self.assertAllClose(values, decoded, rtol=0, atol=.5)

  def test_compress(self):
    # Test compression and decompression, and produce test data for
    # `test_decompress`. If you set the constant at the end to `True`, this test
    # will fail and the log will contain the new test data.
    inputs = array_ops.placeholder(dtypes.float32, (2, 3, 10))
    layer = entropybottleneck.EntropyBottleneck(
        data_format="channels_first", filters=(), init_scale=2)
    bitstrings = layer.compress(inputs)
    decoded = layer.decompress(bitstrings, array_ops.shape(inputs)[1:])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertTrue(len(layer.updates) == 1)
      sess.run(layer.updates[0])
      values = 5 * np.random.uniform(size=(2, 3, 10)) - 2.5
      bitstrings, quantized_cdf, decoded = sess.run(
          [bitstrings, layer._quantized_cdf, decoded], {inputs: values})
      self.assertAllClose(values, decoded, rtol=0, atol=.5)
      # Set this constant to `True` to log new test data for `test_decompress`.
      if False:  # pylint:disable=using-constant-test
        assert False, (bitstrings, quantized_cdf, decoded)

  # Data generated by `test_compress`.
  # pylint:disable=g-inconsistent-quotes,bad-whitespace
  bitstrings = np.array([
      b'\x1e\xbag}\xc2\xdaN\x8b\xbd.',
      b'\x8dF\xf0%\x1cv\xccllW'
  ], dtype=object)

  quantized_cdf = np.array([
      [    0, 15636, 22324, 30145, 38278, 65536],
      [    0, 19482, 26927, 35052, 42904, 65535],
      [    0, 21093, 28769, 36919, 44578, 65536]
  ], dtype=np.int32)

  expected = np.array([
      [[-2.,  1.,  0., -2., -1., -2., -2., -2.,  2., -1.],
       [ 1.,  2.,  1.,  0., -2., -2.,  1.,  2.,  0.,  1.],
       [ 2.,  0., -2.,  2.,  0., -1., -2.,  0.,  2.,  0.]],
      [[ 1.,  2.,  0., -1.,  1.,  2.,  1.,  1.,  2., -2.],
       [ 2., -1., -1.,  0., -1.,  2.,  0.,  2., -2.,  2.],
       [ 2., -2., -2., -1., -2.,  1., -2.,  0.,  0.,  0.]]
  ], dtype=np.float32)
  # pylint:enable=g-inconsistent-quotes,bad-whitespace

  def test_decompress(self):
    # Test that decompression of values compressed with a previous version
    # works, i.e. that the file format doesn't change across revisions.
    bitstrings = array_ops.placeholder(dtypes.string)
    input_shape = array_ops.placeholder(dtypes.int32)
    quantized_cdf = array_ops.placeholder(dtypes.int32)
    layer = entropybottleneck.EntropyBottleneck(
        data_format="channels_first", filters=(), dtype=dtypes.float32)
    layer.build(self.expected.shape)
    layer._quantized_cdf = quantized_cdf
    decoded = layer.decompress(bitstrings, input_shape[1:])
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      decoded, = sess.run([decoded], {
          bitstrings: self.bitstrings, input_shape: self.expected.shape,
          quantized_cdf: self.quantized_cdf})
      self.assertAllClose(self.expected, decoded, rtol=0, atol=1e-6)

  def test_build_decompress(self):
    # Test that layer can be built when `decompress` is the first call to it.
    bitstrings = array_ops.placeholder(dtypes.string)
    input_shape = array_ops.placeholder(dtypes.int32, shape=[3])
    layer = entropybottleneck.EntropyBottleneck(dtype=dtypes.float32)
    layer.decompress(bitstrings, input_shape[1:], channels=5)
    self.assertTrue(layer.built)

  def test_pmf_normalization(self):
    # Test that probability mass functions are normalized correctly.
    layer = entropybottleneck.EntropyBottleneck(dtype=dtypes.float32)
    layer.build((None, 10))
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      pmf, = sess.run([layer._pmf])
      self.assertAllClose(np.ones(10), np.sum(pmf, axis=-1), rtol=0, atol=1e-6)

  def test_visualize(self):
    # Test that summary op can be constructed.
    layer = entropybottleneck.EntropyBottleneck(dtype=dtypes.float32)
    layer.build((None, 10))
    summary = layer.visualize()
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run([summary])

  def test_normalization(self):
    # Test that densities are normalized correctly.
    inputs = array_ops.placeholder(dtypes.float32, (None, 1))
    layer = entropybottleneck.EntropyBottleneck(filters=(2,))
    _, likelihood = layer(inputs, training=True)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      x = np.repeat(np.arange(-200, 201), 1000)[:, None]
      likelihood, = sess.run([likelihood], {inputs: x})
      self.assertEqual(x.shape, likelihood.shape)
      integral = np.sum(likelihood) * .001
      self.assertAllClose(1, integral, rtol=0, atol=1e-4)

  def test_entropy_estimates(self):
    # Test that entropy estimates match actual range coding.
    inputs = array_ops.placeholder(dtypes.float32, (1, None, 1))
    layer = entropybottleneck.EntropyBottleneck(
        filters=(2, 3), data_format="channels_last")
    _, likelihood = layer(inputs, training=True)
    diff_entropy = math_ops.reduce_sum(math_ops.log(likelihood)) / -np.log(2)
    _, likelihood = layer(inputs, training=False)
    disc_entropy = math_ops.reduce_sum(math_ops.log(likelihood)) / -np.log(2)
    bitstrings = layer.compress(inputs)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertTrue(len(layer.updates) == 1)
      sess.run(layer.updates[0])
      diff_entropy, disc_entropy, bitstrings = sess.run(
          [diff_entropy, disc_entropy, bitstrings],
          {inputs: np.random.normal(size=(1, 10000, 1))})
      codelength = 8 * sum(len(bitstring) for bitstring in bitstrings)
      self.assertAllClose(diff_entropy, disc_entropy, rtol=5e-3, atol=0)
      self.assertAllClose(disc_entropy, codelength, rtol=5e-3, atol=0)
      self.assertGreater(codelength, disc_entropy)


if __name__ == "__main__":
  test.main()
