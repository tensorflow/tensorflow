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
"""Tests for dense Bayesian layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import layers_dense_variational_impl as prob_layers_lib
from tensorflow.contrib.bayesflow.python.ops import layers_util as prob_layers_util
from tensorflow.contrib.distributions.python.ops import independent as independent_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import util as distribution_util
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
  """Monitors layer calls to the underlying distribution."""

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


class DenseVariational(test.TestCase):

  def _testKLPenaltyKernel(self, layer_class):
    with self.test_session():
      layer = layer_class(units=2)
      inputs = random_ops.random_uniform([2, 3], seed=1)

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
          units=2,
          bias_posterior_fn=prob_layers_util.default_mean_field_normal_fn(),
          bias_prior_fn=_make_normal)
      inputs = random_ops.random_uniform([2, 3], seed=1)

      # No keys.
      losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(losses), 0)
      self.assertListEqual(layer.losses, losses)

      _ = layer(inputs)

      # Yes keys.
      losses = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(losses), 2)
      self.assertListEqual(layer.losses, losses)

  def _testDenseSetUp(self, layer_class, batch_size, in_size, out_size,
                      **kwargs):
    seed = Counter()
    inputs = random_ops.random_uniform([batch_size, in_size], seed=seed())

    kernel_size = [in_size, out_size]
    kernel_posterior = MockDistribution(
        loc=random_ops.random_uniform(kernel_size, seed=seed()),
        scale=random_ops.random_uniform(kernel_size, seed=seed()),
        result_log_prob=random_ops.random_uniform(kernel_size, seed=seed()),
        result_sample=random_ops.random_uniform(kernel_size, seed=seed()))
    kernel_prior = MockDistribution(
        result_log_prob=random_ops.random_uniform(kernel_size, seed=seed()),
        result_sample=random_ops.random_uniform(kernel_size, seed=seed()))
    kernel_divergence = MockKLDivergence(
        result=random_ops.random_uniform(kernel_size, seed=seed()))

    bias_size = [out_size]
    bias_posterior = MockDistribution(
        result_log_prob=random_ops.random_uniform(bias_size, seed=seed()),
        result_sample=random_ops.random_uniform(bias_size, seed=seed()))
    bias_prior = MockDistribution(
        result_log_prob=random_ops.random_uniform(bias_size, seed=seed()),
        result_sample=random_ops.random_uniform(bias_size, seed=seed()))
    bias_divergence = MockKLDivergence(
        result=random_ops.random_uniform(bias_size, seed=seed()))

    layer = layer_class(
        units=out_size,
        kernel_posterior_fn=lambda *args: kernel_posterior,
        kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
        kernel_prior_fn=lambda *args: kernel_prior,
        kernel_divergence_fn=kernel_divergence,
        bias_posterior_fn=lambda *args: bias_posterior,
        bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
        bias_prior_fn=lambda *args: bias_prior,
        bias_divergence_fn=bias_divergence,
        **kwargs)

    outputs = layer(inputs)

    kl_penalty = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
    return (kernel_posterior, kernel_prior, kernel_divergence,
            bias_posterior, bias_prior, bias_divergence,
            layer, inputs, outputs, kl_penalty)

  def testKLPenaltyKernelReparameterization(self):
    self._testKLPenaltyKernel(prob_layers_lib.DenseReparameterization)

  def testKLPenaltyKernelLocalReparameterization(self):
    self._testKLPenaltyKernel(prob_layers_lib.DenseLocalReparameterization)

  def testKLPenaltyKernelFlipout(self):
    self._testKLPenaltyKernel(prob_layers_lib.DenseFlipout)

  def testKLPenaltyBothReparameterization(self):
    self._testKLPenaltyBoth(prob_layers_lib.DenseReparameterization)

  def testKLPenaltyBothLocalReparameterization(self):
    self._testKLPenaltyBoth(prob_layers_lib.DenseLocalReparameterization)

  def testKLPenaltyBothFlipout(self):
    self._testKLPenaltyBoth(prob_layers_lib.DenseFlipout)

  def testDenseReparameterization(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.test_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty) = self._testDenseSetUp(
           prob_layers_lib.DenseReparameterization,
           batch_size, in_size, out_size)

      expected_outputs = (
          math_ops.matmul(inputs, kernel_posterior.result_sample) +
          bias_posterior.result_sample)

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

  def testDenseLocalReparameterization(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.test_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty) = self._testDenseSetUp(
           prob_layers_lib.DenseLocalReparameterization,
           batch_size, in_size, out_size)

      expected_kernel_posterior_affine = normal_lib.Normal(
          loc=math_ops.matmul(inputs, kernel_posterior.result_loc),
          scale=math_ops.matmul(
              inputs**2., kernel_posterior.result_scale**2)**0.5)
      expected_kernel_posterior_affine_tensor = (
          expected_kernel_posterior_affine.sample(seed=42))
      expected_outputs = (expected_kernel_posterior_affine_tensor +
                          bias_posterior.result_sample)

      [
          expected_outputs_, actual_outputs_,
          expected_kernel_divergence_, actual_kernel_divergence_,
          expected_bias_, actual_bias_,
          expected_bias_divergence_, actual_bias_divergence_,
      ] = sess.run([
          expected_outputs, outputs,
          kernel_divergence.result, kl_penalty[0],
          bias_posterior.result_sample, layer.bias_posterior_tensor,
          bias_divergence.result, kl_penalty[1],
      ])

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
            None]],
          kernel_divergence.args)

      self.assertAllEqual(
          [[bias_posterior.distribution,
            bias_prior.distribution,
            bias_posterior.result_sample]],
          bias_divergence.args)

  def testDenseFlipout(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.test_session() as sess:
      (kernel_posterior, kernel_prior, kernel_divergence,
       bias_posterior, bias_prior, bias_divergence, layer, inputs,
       outputs, kl_penalty) = self._testDenseSetUp(
           prob_layers_lib.DenseFlipout,
           batch_size, in_size, out_size, seed=44)

      expected_kernel_posterior_affine = normal_lib.Normal(
          loc=array_ops.zeros_like(kernel_posterior.result_loc),
          scale=kernel_posterior.result_scale)
      expected_kernel_posterior_affine_tensor = (
          expected_kernel_posterior_affine.sample(seed=42))

      sign_input = random_ops.random_uniform(
          [batch_size, in_size],
          minval=0,
          maxval=2,
          dtype=dtypes.int32,
          seed=layer.seed)
      sign_input = math_ops.cast(2 * sign_input - 1, inputs.dtype)
      sign_output = random_ops.random_uniform(
          [batch_size, out_size],
          minval=0,
          maxval=2,
          dtype=dtypes.int32,
          seed=distribution_util.gen_new_seed(
              layer.seed, salt="dense_flipout"))
      sign_output = math_ops.cast(2 * sign_output - 1, inputs.dtype)
      perturbed_inputs = math_ops.matmul(
          inputs * sign_input, expected_kernel_posterior_affine_tensor)
      perturbed_inputs *= sign_output

      expected_outputs = math_ops.matmul(inputs, kernel_posterior.result_loc)
      expected_outputs += perturbed_inputs
      expected_outputs += bias_posterior.result_sample

      [
          expected_outputs_, actual_outputs_,
          expected_kernel_divergence_, actual_kernel_divergence_,
          expected_bias_, actual_bias_,
          expected_bias_divergence_, actual_bias_divergence_,
      ] = sess.run([
          expected_outputs, outputs,
          kernel_divergence.result, kl_penalty[0],
          bias_posterior.result_sample, layer.bias_posterior_tensor,
          bias_divergence.result, kl_penalty[1],
      ])

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
          [[kernel_posterior.distribution, kernel_prior.distribution, None]],
          kernel_divergence.args)

      self.assertAllEqual(
          [[bias_posterior.distribution,
            bias_prior.distribution,
            bias_posterior.result_sample]],
          bias_divergence.args)

  def testRandomDenseFlipout(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.test_session() as sess:
      seed = Counter()
      inputs = random_ops.random_uniform([batch_size, in_size], seed=seed())

      kernel_posterior = MockDistribution(
          loc=random_ops.random_uniform(
              [in_size, out_size], seed=seed()),
          scale=random_ops.random_uniform(
              [in_size, out_size], seed=seed()),
          result_log_prob=random_ops.random_uniform(
              [in_size, out_size], seed=seed()),
          result_sample=random_ops.random_uniform(
              [in_size, out_size], seed=seed()))
      bias_posterior = MockDistribution(
          loc=random_ops.random_uniform(
              [out_size], seed=seed()),
          scale=random_ops.random_uniform(
              [out_size], seed=seed()),
          result_log_prob=random_ops.random_uniform(
              [out_size], seed=seed()),
          result_sample=random_ops.random_uniform(
              [out_size], seed=seed()))
      layer_one = prob_layers_lib.DenseFlipout(
          units=out_size,
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          seed=44)
      layer_two = prob_layers_lib.DenseFlipout(
          units=out_size,
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          seed=45)

      outputs_one = layer_one(inputs)
      outputs_two = layer_two(inputs)

      outputs_one_, outputs_two_ = sess.run([
          outputs_one, outputs_two])

      self.assertLess(np.sum(np.isclose(outputs_one_, outputs_two_)), out_size)


if __name__ == "__main__":
  test.main()
