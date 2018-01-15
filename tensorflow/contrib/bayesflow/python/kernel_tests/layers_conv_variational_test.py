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
"""Tests for convolutional Bayesian layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.bayesflow.python.ops import layers_conv_variational as prob_layers_lib
from tensorflow.contrib.bayesflow.python.ops import layers_util as prob_layers_util
from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


class Counter(object):
  """Helper class to manage incrementing a counting `int`."""

  def __init__(self):
    self._value = -1

  @property
  def value(self):
    return self._value

  def __call__(self):
    self._value += 1
    return self._value


class MockDistribution(independent_lib.Independent):
  """Monitors DenseVariational calls to the underlying distribution."""

  def __init__(self, result_sample, result_log_prob, loc=None, scale=None):
    self.result_sample = result_sample
    self.result_log_prob = result_log_prob
    self.result_loc = loc
    self.result_scale = scale
    self.result_distribution = normal_lib.Normal(loc=0.0, scale=1.0)
    if loc is not None and scale is not None:
      self.result_distribution = normal_lib.Normal(loc=self.result_loc,
                                                   scale=self.result_scale)
    self.called_log_prob = Counter()
    self.called_sample = Counter()
    self.called_loc = Counter()
    self.called_scale = Counter()

  def log_prob(self, *args, **kwargs):
    self.called_log_prob()
    return self.result_log_prob

  def sample(self, *args, **kwargs):
    self.called_sample()
    return self.result_sample

  @property
  def distribution(self):  # for dummy check on Independent(Normal)
    return self.result_distribution

  @property
  def loc(self):
    self.called_loc()
    return self.result_loc

  @property
  def scale(self):
    self.called_scale()
    return self.result_scale


class MockKLDivergence(object):
  """Monitors layer calls to the divergence implementation."""

  def __init__(self, result):
    self.result = result
    self.args = []
    self.called = Counter()

  def __call__(self, *args, **kwargs):
    self.called()
    self.args.append(args)
    return self.result


class ConvVariational(test.TestCase):

  def _testKLPenaltyKernel(self, layer_class):
    with self.test_session():
      layer = layer_class(filters=2, kernel_size=3)
      if layer_class == prob_layers_lib.Conv1DVariational:
        inputs = random_ops.random_uniform([2, 3, 1], seed=1)
      elif layer_class == prob_layers_lib.Conv2DVariational:
        inputs = random_ops.random_uniform([2, 3, 3, 1], seed=1)
      elif layer_class == prob_layers_lib.Conv3DVariational:
        inputs = random_ops.random_uniform([2, 3, 3, 3, 1], seed=1)

      # No keys.
      losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(losses), 0)
      self.assertListEqual(layer.losses, losses)

      _ = layer(inputs)

      # Yes keys.
      losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(losses), 1)
      self.assertListEqual(layer.losses, losses)

  def _testKLPenaltyBoth(self, layer_class):
    def _make_normal(dtype, *args):  # pylint: disable=unused-argument
      return normal_lib.Normal(
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.))
    with self.test_session():
      layer = layer_class(
          filters=2,
          kernel_size=3,
          bias_posterior_fn=prob_layers_util.default_mean_field_normal_fn(),
          bias_prior_fn=_make_normal)
      if layer_class == prob_layers_lib.Conv1DVariational:
        inputs = random_ops.random_uniform([2, 3, 1], seed=1)
      elif layer_class == prob_layers_lib.Conv2DVariational:
        inputs = random_ops.random_uniform([2, 3, 3, 1], seed=1)
      elif layer_class == prob_layers_lib.Conv3DVariational:
        inputs = random_ops.random_uniform([2, 3, 3, 3, 1], seed=1)

      # No keys.
      losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(losses), 0)
      self.assertListEqual(layer.losses, losses)

      _ = layer(inputs)

      # Yes keys.
      losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(losses), 2)
      self.assertListEqual(layer.losses, losses)

  def _testConvVariational(self, layer_class):
    batch_size, depth, height, width, channels, filters = 2, 4, 4, 4, 3, 5
    with self.test_session() as sess:
      seed = Counter()
      if layer_class == prob_layers_lib.Conv1DVariational:
        inputs = random_ops.random_uniform(
            [batch_size, width, channels], seed=seed())
        kernel_size = (2,)
      elif layer_class == prob_layers_lib.Conv2DVariational:
        inputs = random_ops.random_uniform(
            [batch_size, height, width, channels], seed=seed())
        kernel_size = (2, 2)
      elif layer_class == prob_layers_lib.Conv3DVariational:
        inputs = random_ops.random_uniform(
            [batch_size, depth, height, width, channels], seed=seed())
        kernel_size = (2, 2, 2)

      kernel_shape = kernel_size + (channels, filters)
      kernel_posterior = MockDistribution(
          result_log_prob=random_ops.random_uniform(kernel_shape, seed=seed()),
          result_sample=random_ops.random_uniform(kernel_shape, seed=seed()))
      kernel_prior = MockDistribution(
          result_log_prob=random_ops.random_uniform(kernel_shape, seed=seed()),
          result_sample=random_ops.random_uniform(kernel_shape, seed=seed()))
      kernel_divergence = MockKLDivergence(
          result=random_ops.random_uniform(kernel_shape, seed=seed()))

      bias_size = (filters,)
      bias_posterior = MockDistribution(
          result_log_prob=random_ops.random_uniform(bias_size, seed=seed()),
          result_sample=random_ops.random_uniform(bias_size, seed=seed()))
      bias_prior = MockDistribution(
          result_log_prob=random_ops.random_uniform(bias_size, seed=seed()),
          result_sample=random_ops.random_uniform(bias_size, seed=seed()))
      bias_divergence = MockKLDivergence(
          result=random_ops.random_uniform(bias_size, seed=seed()))

      convolution_op = nn_ops.Convolution(
          tensor_shape.TensorShape(inputs.shape),
          filter_shape=tensor_shape.TensorShape(kernel_shape),
          padding="SAME")
      expected_outputs = convolution_op(inputs, kernel_posterior.result_sample)
      expected_outputs = nn.bias_add(expected_outputs,
                                     bias_posterior.result_sample,
                                     data_format="NHWC")

      layer = layer_class(
          filters=filters,
          kernel_size=kernel_size,
          padding="SAME",
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          kernel_prior_fn=lambda *args: kernel_prior,
          kernel_divergence_fn=kernel_divergence,
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          bias_prior_fn=lambda *args: bias_prior,
          bias_divergence_fn=bias_divergence)

      outputs = layer(inputs)

      kl_penalty = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)

      [
          expected_outputs_, actual_outputs_,
          expected_kernel_, actual_kernel_,
          expected_kernel_divergence_, actual_kernel_divergence_,
          expected_bias_, actual_bias_,
          expected_bias_divergence_, actual_bias_divergence_,
      ] = sess.run([
          expected_outputs, outputs,
          kernel_posterior.result_sample, layer.kernel_posterior_tensor,
          kernel_divergence.result, kl_penalty[0],
          bias_posterior.result_sample, layer.bias_posterior_tensor,
          bias_divergence.result, kl_penalty[1],
      ])

      self.assertAllClose(
          expected_kernel_, actual_kernel_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_bias_, actual_bias_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_outputs_, actual_outputs_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_kernel_divergence_, actual_kernel_divergence_,
          rtol=1e-6, atol=0.)
      self.assertAllClose(
          expected_bias_divergence_, actual_bias_divergence_,
          rtol=1e-6, atol=0.)

      self.assertAllEqual(
          [[kernel_posterior.distribution,
            kernel_prior.distribution,
            kernel_posterior.result_sample]],
          kernel_divergence.args)

      self.assertAllEqual(
          [[bias_posterior.distribution,
            bias_prior.distribution,
            bias_posterior.result_sample]],
          bias_divergence.args)

  def testKLPenaltyKernelConv1DVariational(self):
    self._testKLPenaltyKernel(prob_layers_lib.Conv1DVariational)

  def testKLPenaltyKernelConv2DVariational(self):
    self._testKLPenaltyKernel(prob_layers_lib.Conv2DVariational)

  def testKLPenaltyKernelConv3DVariational(self):
    self._testKLPenaltyKernel(prob_layers_lib.Conv3DVariational)

  def testKLPenaltyBothConv1DVariational(self):
    self._testKLPenaltyBoth(prob_layers_lib.Conv1DVariational)

  def testKLPenaltyBothConv2DVariational(self):
    self._testKLPenaltyBoth(prob_layers_lib.Conv2DVariational)

  def testKLPenaltyBothConv3DVariational(self):
    self._testKLPenaltyBoth(prob_layers_lib.Conv3DVariational)

  def testConv1DVariational(self):
    self._testConvVariational(prob_layers_lib.Conv1DVariational)

  def testConv2DVariational(self):
    self._testConvVariational(prob_layers_lib.Conv2DVariational)

  def testConv3DVariational(self):
    self._testConvVariational(prob_layers_lib.Conv3DVariational)


if __name__ == "__main__":
  test.main()
