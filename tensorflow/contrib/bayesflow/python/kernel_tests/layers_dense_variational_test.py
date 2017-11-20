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

from tensorflow.contrib.bayesflow.python.ops import layers_dense_variational_impl as prob_layers_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
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


class MockDistribution(normal_lib.Normal):
  """Monitors DenseVariational calls to the underlying distribution."""

  def __init__(self, result_sample, result_log_prob, loc=None, scale=None):
    self.result_sample = result_sample
    self.result_log_prob = result_log_prob
    self.result_loc = loc
    self.result_scale = scale
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
  def loc(self):
    self.called_loc()
    return self.result_loc

  @property
  def scale(self):
    self.called_scale()
    return self.result_scale


class MockKLDivergence(object):
  """Monitors DenseVariational calls to the divergence implementation."""

  def __init__(self, result):
    self.result = result
    self.args = []
    self.called = Counter()

  def __call__(self, *args, **kwargs):
    self.called()
    self.args.append(args)
    return self.result


class DenseVariationalLocalReparametrization(test.TestCase):

  def testKLPenaltyKernel(self):
    with self.test_session():
      dense_vi = prob_layers_lib.DenseVariational(units=2)
      inputs = random_ops.random_uniform([2, 3], seed=1)

      # No keys.
      loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(loss_keys), 0)
      self.assertListEqual(dense_vi.losses, loss_keys)

      _ = dense_vi(inputs)

      # Yes keys.
      loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(loss_keys), 1)
      self.assertListEqual(dense_vi.losses, loss_keys)

  def testKLPenaltyBoth(self):
    def _make_normal(dtype, *args):  # pylint: disable=unused-argument
      return normal_lib.Normal(
          loc=dtype.as_numpy_dtype(0.), scale=dtype.as_numpy_dtype(1.))
    with self.test_session():
      dense_vi = prob_layers_lib.DenseVariational(
          units=2,
          bias_posterior_fn=prob_layers_lib.default_mean_field_normal_fn(),
          bias_prior_fn=_make_normal)
      inputs = random_ops.random_uniform([2, 3], seed=1)

      # No keys.
      loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(loss_keys), 0)
      self.assertListEqual(dense_vi.losses, loss_keys)

      _ = dense_vi(inputs)

      # Yes keys.
      loss_keys = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(loss_keys), 2)
      self.assertListEqual(dense_vi.losses, loss_keys)

  def testVariationalNonLocal(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.test_session() as sess:
      seed = Counter()
      inputs = random_ops.random_uniform([batch_size, in_size], seed=seed())

      kernel_size = [in_size, out_size]
      kernel_posterior = MockDistribution(
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

      expected_outputs = (
          math_ops.matmul(inputs, kernel_posterior.result_sample) +
          bias_posterior.result_sample)

      dense_vi = prob_layers_lib.DenseVariational(
          units=2,
          kernel_use_local_reparameterization=False,
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          kernel_prior_fn=lambda *args: kernel_prior,
          kernel_divergence_fn=kernel_divergence,
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          bias_prior_fn=lambda *args: bias_prior,
          bias_divergence_fn=bias_divergence)

      outputs = dense_vi(inputs)

      kl_penalty = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)

      [
          expected_outputs_, actual_outputs_,
          expected_kernel_, actual_kernel_,
          expected_kernel_divergence_, actual_kernel_divergence_,
          expected_bias_, actual_bias_,
          expected_bias_divergence_, actual_bias_divergence_,
      ] = sess.run([
          expected_outputs, outputs,
          kernel_posterior.result_sample, dense_vi.kernel.posterior_tensor,
          kernel_divergence.result, kl_penalty[0],
          bias_posterior.result_sample, dense_vi.bias.posterior_tensor,
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
          [[kernel_posterior, kernel_prior, kernel_posterior.result_sample]],
          kernel_divergence.args)

      self.assertAllEqual(
          [[bias_posterior, bias_prior, bias_posterior.result_sample]],
          bias_divergence.args)

  def testVariationalLocal(self):
    batch_size, in_size, out_size = 2, 3, 4
    with self.test_session() as sess:
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

      expected_kernel_posterior_affine = normal_lib.Normal(
          loc=math_ops.matmul(inputs, kernel_posterior.result_loc),
          scale=math_ops.matmul(
              inputs**2., kernel_posterior.result_scale**2)**0.5)
      expected_kernel_posterior_affine_tensor = (
          expected_kernel_posterior_affine.sample(seed=42))
      expected_outputs = (expected_kernel_posterior_affine_tensor +
                          bias_posterior.result_sample)

      dense_vi = prob_layers_lib.DenseVariational(
          units=2,
          kernel_use_local_reparameterization=True,
          kernel_posterior_fn=lambda *args: kernel_posterior,
          kernel_posterior_tensor_fn=lambda d: d.sample(seed=42),
          kernel_prior_fn=lambda *args: kernel_prior,
          kernel_divergence_fn=kernel_divergence,
          bias_posterior_fn=lambda *args: bias_posterior,
          bias_posterior_tensor_fn=lambda d: d.sample(seed=43),
          bias_prior_fn=lambda *args: bias_prior,
          bias_divergence_fn=bias_divergence)

      outputs = dense_vi(inputs)

      kl_penalty = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)

      [
          expected_outputs_, actual_outputs_,
          expected_kernel_divergence_, actual_kernel_divergence_,
          expected_bias_, actual_bias_,
          expected_bias_divergence_, actual_bias_divergence_,
      ] = sess.run([
          expected_outputs, outputs,
          kernel_divergence.result, kl_penalty[0],
          bias_posterior.result_sample, dense_vi.bias.posterior_tensor,
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
          [[kernel_posterior, kernel_prior, None]],
          kernel_divergence.args)

      self.assertAllEqual(
          [[bias_posterior, bias_prior, bias_posterior.result_sample]],
          bias_divergence.args)


if __name__ == "__main__":
  test.main()
